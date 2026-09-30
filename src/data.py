"""Turn raw Jolpica files into one clean driver-race table.

One row per driver per race, with everything the model needs:
qualifying gap to pole, grid slot, team, whether the driver retired,
and the outcome (won, podium).
"""

from pathlib import Path

import numpy as np
import pandas as pd

FINISHED = {"Finished", "Lapped"}  # classified at the end; everything else counts as a retirement


def lap_seconds(t) -> float:
    """'1:29.179' -> 89.179; missing -> NaN."""
    if not isinstance(t, str) or ":" not in t:
        return np.nan
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def load_races(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    res = pd.read_csv(data_dir / "results.csv")
    qual = pd.read_csv(data_dir / "qualifying.csv")

    # Each driver's best qualifying lap across Q1 to Q3
    for q in ["q1", "q2", "q3"]:
        qual[q + "_s"] = qual[q].map(lap_seconds)
    qual["best_s"] = qual[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    qual = qual[["season", "round", "driver_id", "best_s"]]

    df = res.merge(qual, on=["season", "round", "driver_id"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    df["race_key"] = df["season"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)

    # Gap to pole in percent; drivers with no time go to the back with a small margin
    pole = df.groupby("race_key")["best_s"].transform("min")
    df["quali_gap"] = (df["best_s"] / pole - 1) * 100
    worst = df.groupby("race_key")["quali_gap"].transform("max")
    df["quali_gap"] = df["quali_gap"].fillna(worst + 0.5).clip(upper=7.0)

    # Grid 0 means a pit lane start: treat as starting last
    n = df.groupby("race_key")["driver_id"].transform("size")
    df["grid"] = np.where(df["grid"] == 0, n, df["grid"])

    # Jolpica has no qualifying for a few races (2025 Miami). For those, impute
    # each driver's gap from the typical gap at their grid slot in other races.
    typical = df.groupby("grid")["quali_gap"].median()
    df["quali_imputed"] = df["quali_gap"].isna()
    df["quali_gap"] = df["quali_gap"].fillna(df["grid"].map(typical))

    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["won"] = df["position"] == 1
    df["podium"] = df["position"] <= 3
    df["dnf"] = ~df["status"].isin(FINISHED)

    cols = ["race_key", "season", "round", "race_name", "circuit_id", "date", "driver_id",
            "constructor", "grid", "quali_gap", "quali_imputed", "position", "won", "podium", "dnf"]
    return df[cols].sort_values(["date", "grid"]).reset_index(drop=True)
