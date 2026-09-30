"""Compare the model with Polymarket: accuracy, the qualifying reaction, and a backtest.

Two snapshots per race, both taken from hourly prices:
  pre_weekend   00:00 UTC three days before the race (before first practice)
  post_quali    00:00 UTC on race day (after qualifying, before the start)
Every race in the sample fits these windows: qualifying ends Saturday by
about 22:00 UTC and the earliest race starts about 04:00 UTC Sunday.

Market prices are normalized so the field sums to one, which strips out the
market's built in margin. Only 2025 onward is used: 2024 markets listed just
the front runners, so their normalized prices aren't comparable.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIRST_SEASON = 2025


def market_snapshots(data_dir: str | Path, races: pd.DataFrame) -> pd.DataFrame:
    """One row per race and driver: normalized market probability at each snapshot."""
    data_dir = Path(data_dir)
    prices = pd.read_parquet(data_dir / "pm_prices.parquet")
    race_match = pd.read_csv(data_dir / "race_match.csv").dropna(subset=["season"])
    driver_match = pd.read_csv(data_dir / "driver_match.csv").dropna(subset=["driver_id"])

    race_match["race_key"] = (race_match["season"].astype(int).astype(str) + "_"
                              + race_match["round"].astype(int).astype(str).str.zfill(2))
    prices = (prices.merge(race_match[["event_id", "race_key"]], on="event_id")
                    .merge(driver_match[["pm_driver", "driver_id"]], left_on="driver", right_on="pm_driver"))

    dates = races.drop_duplicates("race_key").set_index("race_key")["date"]
    rows = []
    for key, g in prices.groupby("race_key"):
        race_day = pd.Timestamp(dates[key]).tz_localize("UTC")
        for name, t in [("pre_weekend", race_day - pd.Timedelta(days=3)), ("post_quali", race_day)]:
            last = (g[g["ts"] <= t].sort_values("ts").groupby("driver_id")["price"].last())
            rows.append(pd.DataFrame({"race_key": key, "snapshot": name,
                                      "driver_id": last.index, "raw_price": last.to_numpy()}))
    snap = pd.concat(rows, ignore_index=True)
    snap = snap.pivot_table(index=["race_key", "driver_id"], columns="snapshot", values="raw_price").reset_index()
    snap.columns.name = None

    out = races[["race_key", "season", "driver_id", "won"]].merge(snap, on=["race_key", "driver_id"], how="inner")
    out = out[out["season"] >= FIRST_SEASON]
    for col in ["pre_weekend", "post_quali"]:
        if col in out:
            out[f"mkt_{col}"] = out[col] / out.groupby("race_key")[col].transform("sum")
    return out


def as_pred(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reshape a probability column into the (race_key, driver_id, p_win, won) format evaluate uses."""
    sub = df.dropna(subset=[col])
    # keep only races where every listed driver has a price at this snapshot
    complete = sub.groupby("race_key")[col].transform("size") == df.groupby("race_key")["driver_id"].transform("size").loc[sub.index]
    sub = sub[complete]
    return sub[["race_key", "driver_id", "won"]].assign(p_win=sub[col].to_numpy())


def blend_curve(model_p: pd.Series, market_p: pd.Series, won: pd.Series, race: pd.Series) -> pd.DataFrame:
    """Log loss of a linear blend: weight on the model from 0 (pure market) to 1 (pure model).

    If the best weight is above zero, the model carries information the market missed.
    """
    rows = []
    for lam in np.linspace(0, 1, 11):
        p = lam * model_p + (1 - lam) * market_p
        p_winner = p[won.to_numpy()].groupby(race[won.to_numpy()]).sum().clip(lower=1e-4)
        rows.append({"model_weight": round(lam, 1), "log_loss": float(-np.log(p_winner).mean())})
    return pd.DataFrame(rows)


def qualifying_reaction(df: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """Does the market's post qualifying move have the right size?

    Regress the outcome on the pre weekend price and the qualifying move:
        won = a + b1 * pre + b2 * (post - pre)
    Efficient updating gives b2 close to b1. b2 above b1 means the market
    under reacts to qualifying; below means it over reacts. The interval
    comes from resampling whole races.
    """
    d = df.dropna(subset=["mkt_pre_weekend", "mkt_post_quali"]).copy()
    d["move"] = d["mkt_post_quali"] - d["mkt_pre_weekend"]

    def fit(sample):
        X = np.column_stack([np.ones(len(sample)), sample["mkt_pre_weekend"], sample["move"]])
        coef, *_ = np.linalg.lstsq(X, sample["won"].astype(float), rcond=None)
        return coef

    coef = fit(d)
    rng = np.random.default_rng(seed)
    keys = d["race_key"].unique()
    groups = {k: g for k, g in d.groupby("race_key")}
    boots = np.array([fit(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(n_boot)])
    ratio = boots[:, 2] / boots[:, 1]
    return {"races": len(keys), "b_pre": coef[1], "b_move": coef[2],
            "move_to_pre_ratio": coef[2] / coef[1],
            "ratio_ci_low": np.percentile(ratio, 2.5), "ratio_ci_high": np.percentile(ratio, 97.5)}


def backtest(df: pd.DataFrame, model_col: str, market_col: str, edge: float = 0.03,
             cost: float = 0.01, n_boot: int = 5000, seed: int = 0) -> dict:
    """Buy 1 unit of 'yes' on every driver where the model beats the market price by `edge`.

    Fills are assumed at the market price plus `cost` (no order book history is
    available, so this stands in for the bid ask spread). Profit per bet is
    1/price - 1 if the driver wins, else -1. Interval from resampling races.
    """
    d = df.dropna(subset=[model_col, market_col]).copy()
    d["fill"] = d[market_col] + cost
    bets = d[d[model_col] > d["fill"] + edge].copy()
    if bets.empty:
        return {"bets": 0}
    bets["pnl"] = np.where(bets["won"], 1 / bets["fill"] - 1, -1.0)
    per_race = bets.groupby("race_key")["pnl"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_race), (n_boot, len(per_race)))
    roi_boot = per_race["sum"].to_numpy()[idx].sum(1) / per_race["size"].to_numpy()[idx].sum(1)
    return {"bets": len(bets), "races": len(per_race), "wins": int(bets["won"].sum()),
            "roi": bets["pnl"].sum() / len(bets),
            "roi_ci_low": np.percentile(roi_boot, 2.5), "roi_ci_high": np.percentile(roi_boot, 97.5)}


def blend_walk_forward(df: pd.DataFrame, model_col: str, market_col: str,
                       min_races: int = 8) -> pd.DataFrame:
    """Blend weight chosen for each race using only earlier races.

    The in sample blend curve picks its best weight after seeing every result.
    This version can't: each race's weight comes from races before it, so its
    loss is an honest out of sample number to compare against the market alone.
    """
    order = df.drop_duplicates("race_key").sort_values("race_key")["race_key"].tolist()
    lams = np.linspace(0, 1, 11)

    def winner_loss(g, lam):
        p = lam * g[model_col] + (1 - lam) * g[market_col]
        return float(-np.log(max(p[g["won"].to_numpy()].sum(), 1e-4)))

    groups = {k: g for k, g in df.groupby("race_key")}
    table = pd.DataFrame({lam: [winner_loss(groups[k], lam) for k in order] for lam in lams}, index=order)
    rows = []
    for i, key in enumerate(order):
        if i < min_races:
            continue
        lam = table.iloc[:i].mean().idxmin()
        rows.append({"race_key": key, "model_weight": lam,
                     "blend_loss": table.loc[key, lam], "market_loss": table.loc[key, 0.0]})
    return pd.DataFrame(rows)
