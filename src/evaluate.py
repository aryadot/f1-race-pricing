"""Score walk forward predictions: accuracy, calibration, and a grid position baseline."""

import numpy as np
import pandas as pd

from model import MIN_TRAIN_RACES, N_SIMS

CAL_BINS = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]


def grid_baseline(df: pd.DataFrame, race_order: list[str]) -> pd.DataFrame:
    """Win probability from starting slot alone, learned from earlier races.

    Each slot's historical win rate (with a small smoothing count), renormalized
    so the field sums to one. The model has to beat this to be worth anything.
    """
    out = []
    for i, key in enumerate(race_order):
        if i < MIN_TRAIN_RACES:
            continue
        past = df[df["race_key"].isin(race_order[:i])]
        wins = past.groupby("grid")["won"].sum()
        starts = past.groupby("grid")["won"].size()
        race = df[df["race_key"] == key]
        rate = ((race["grid"].map(wins).fillna(0) + 0.05)
                / (race["grid"].map(starts).fillna(0) + 1.0))
        out.append(pd.DataFrame({"race_key": key, "driver_id": race["driver_id"].to_numpy(),
                                 "p_win": (rate / rate.sum()).to_numpy(),
                                 "won": race["won"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def per_race_scores(pred: pd.DataFrame) -> pd.DataFrame:
    def one(g):
        p_actual = max(g.loc[g["won"], "p_win"].sum(), 1 / N_SIMS)
        return pd.Series({"log_loss": -np.log(p_actual),
                          "brier": ((g["p_win"] - g["won"]) ** 2).sum(),
                          "p_actual_winner": p_actual})
    return pred.groupby("race_key").apply(one, include_groups=False)


def compare(a: pd.DataFrame, b: pd.DataFrame, metric: str = "log_loss",
            n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean difference a minus b over shared races, with a bootstrap 95% interval.

    Negative means a is better (lower loss).
    """
    joined = per_race_scores(a)[[metric]].join(per_race_scores(b)[[metric]], lsuffix="_a", rsuffix="_b",
                                              how="inner")
    diff = (joined[f"{metric}_a"] - joined[f"{metric}_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, (n_boot, len(diff))).mean(axis=1)
    return {"races": len(diff), "mean_a": joined[f"{metric}_a"].mean(), "mean_b": joined[f"{metric}_b"].mean(),
            "diff": diff.mean(), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)}


def calibration(pred: pd.DataFrame) -> pd.DataFrame:
    """Do drivers given ~X% chance actually win ~X% of the time?"""
    bins = pd.cut(pred["p_win"], CAL_BINS, include_lowest=True)
    return pred.groupby(bins, observed=True).agg(predicted=("p_win", "mean"),
                                                 observed=("won", "mean"),
                                                 drivers=("won", "size"),
                                                 wins=("won", "sum"))
