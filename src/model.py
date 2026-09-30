"""Monte Carlo race model: turns pace, grid, and reliability into win and podium probabilities.

Each simulated race, every driver gets a performance score (lower is better):

    score = beta * pace + gamma * (grid - 1) + noise,   noise ~ Normal(0, 1)

pace   qualifying gap to pole (%), pulled toward the teammate's gap by weight w,
       since teammates share a car and one lap is a noisy measure
grid   starting slot; gamma is how much track position matters beyond pace
noise  everything unmodeled: strategy, starts, incidents, luck

Each driver also retires with a probability based on their team's past
reliability. The best score among finishers wins. Parameters are chosen
walk forward: each race uses only settings that worked on earlier races.

Two versions answer the market making question:
  post_quali   uses qualifying and grid (priced after qualifying)
  pre_weekend  uses only recent form, before any running (priced before the weekend)
"""

import itertools

import numpy as np
import pandas as pd

N_SIMS = 10_000
BETAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
GAMMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
WEIGHTS = [1.0, 0.75, 0.5]           # own lap vs teammate average
DNF_PRIOR_STRENGTH = 10              # pseudo races pulling team rates toward the field average
FORM_HALFLIFE = 3                    # races
MIN_TRAIN_RACES = 10


# ------------------------------------------------------------------ features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Team averages, shrunk pace, pre weekend form, and team retirement rates.

    Everything that looks at other races uses only races before the current one.
    """
    df = df.sort_values(["date", "grid"]).copy()
    df["team_gap"] = df.groupby(["race_key", "constructor"])["quali_gap"].transform("mean")

    # Pre weekend form: exponentially weighted mean of the driver's past gaps,
    # falling back to the team's past gaps, then the field median
    order = df.drop_duplicates("race_key")["race_key"].tolist()
    idx = {k: i for i, k in enumerate(order)}
    df["race_idx"] = df["race_key"].map(idx)
    df["form"] = (df.groupby("driver_id")["quali_gap"]
                    .transform(lambda s: s.shift(1).ewm(halflife=FORM_HALFLIFE).mean()))
    team_form = (df.groupby(["constructor", "race_idx"])["quali_gap"].mean()
                   .groupby(level=0).transform(lambda s: s.shift(1).ewm(halflife=FORM_HALFLIFE).mean()))
    df["form"] = df["form"].fillna(df.set_index(["constructor", "race_idx"]).index.map(team_form).to_series(index=df.index))
    df["form"] = df["form"].fillna(df.groupby("race_key")["form"].transform("median")).fillna(1.5)

    # Team retirement rate from earlier races, shrunk toward the field average
    team_race = df.groupby(["constructor", "race_idx"])["dnf"].agg(["sum", "size"]).reset_index()
    team_race = team_race.sort_values("race_idx")
    team_race["prior_dnf"] = team_race.groupby("constructor")["sum"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    team_race["prior_n"] = team_race.groupby("constructor")["size"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    field = df.groupby("race_idx")["dnf"].mean().shift(1).expanding().mean().fillna(0.12)
    team_race["p_dnf"] = ((team_race["prior_dnf"] + DNF_PRIOR_STRENGTH * team_race["race_idx"].map(field))
                          / (team_race["prior_n"] + DNF_PRIOR_STRENGTH))
    df = df.merge(team_race[["constructor", "race_idx", "p_dnf"]], on=["constructor", "race_idx"], how="left")
    return df


# ------------------------------------------------------------------ simulation
def simulate(pace, grid, p_dnf, beta, gamma, noise, retire_draw):
    """Win and podium probabilities for one race.

    noise and retire_draw are pre drawn (N_SIMS x drivers) so every parameter
    setting sees the same random races, which makes comparisons fair and fast.
    """
    score = beta * pace + gamma * (grid - 1) + noise
    score = np.where(retire_draw < p_dnf, np.inf, score)
    order = np.argsort(score, axis=1)
    n = len(pace)
    p_win = np.bincount(order[:, 0], minlength=n) / len(score)
    finished_enough = np.isfinite(np.take_along_axis(score, order[:, :3], axis=1))
    # a retired car can't take a podium spot even if fewer than 3 finish
    podium_counts = np.zeros(n)
    for k in range(3):
        np.add.at(podium_counts, order[finished_enough[:, k], k], 1)
    return p_win, podium_counts / len(score)


def race_draws(n_drivers: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_SIMS, n_drivers)), rng.random((N_SIMS, n_drivers))


def score_all_settings(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """For every race and every parameter setting: probabilities and log loss.

    Returns long table: race_key, setting, driver rows with p_win and p_podium.
    """
    rows = []
    settings = (list(itertools.product(BETAS, GAMMAS, WEIGHTS)) if version == "post_quali"
                else [(b, 0.0, 1.0) for b in BETAS])
    for i, (key, race) in enumerate(df.groupby("race_key", sort=False)):
        noise, retire = race_draws(len(race), seed=i)
        grid = race["grid"].to_numpy(float)
        p_dnf = race["p_dnf"].to_numpy()
        for beta, gamma, w in settings:
            if version == "post_quali":
                pace = w * race["quali_gap"].to_numpy() + (1 - w) * race["team_gap"].to_numpy()
                g = grid
            else:
                pace, g = race["form"].to_numpy(), np.ones_like(grid)
            p_win, p_pod = simulate(pace, g, p_dnf, beta, gamma, noise, retire)
            rows.append(pd.DataFrame({"race_key": key, "beta": beta, "gamma": gamma, "w": w,
                                      "driver_id": race["driver_id"].to_numpy(),
                                      "p_win": p_win, "p_podium": p_pod,
                                      "won": race["won"].to_numpy(), "podium": race["podium"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def log_loss_winner(p_win: pd.Series, won: pd.Series) -> float:
    """Minus log probability given to the actual winner (floored at one simulation)."""
    return float(-np.log(max(p_win[won.to_numpy()].sum(), 1 / N_SIMS)))


def walk_forward(scored: pd.DataFrame, race_order: list[str]) -> pd.DataFrame:
    """Pick settings for each race using only earlier races, return its predictions."""
    loss = (scored.groupby(["race_key", "beta", "gamma", "w"])
                  .apply(lambda g: log_loss_winner(g["p_win"], g["won"]), include_groups=False)
                  .rename("loss").reset_index())
    out = []
    for i, key in enumerate(race_order):
        if i < MIN_TRAIN_RACES:
            continue
        past = loss[loss["race_key"].isin(race_order[:i])]
        best = past.groupby(["beta", "gamma", "w"])["loss"].mean().idxmin()
        pred = scored[(scored["race_key"] == key) & (scored["beta"] == best[0])
                      & (scored["gamma"] == best[1]) & (scored["w"] == best[2])]
        out.append(pred)
    return pd.concat(out, ignore_index=True)
