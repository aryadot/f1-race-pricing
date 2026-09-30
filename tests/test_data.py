import numpy as np
import pandas as pd

from data import lap_seconds, load_races


def test_lap_seconds_parses_and_handles_missing():
    assert lap_seconds("1:29.179") == 89.179
    assert np.isnan(lap_seconds(None))
    assert np.isnan(lap_seconds("DNF"))


def test_pole_sitter_has_zero_gap(data_dir):
    df = load_races(data_dir)
    pole = df[(df["race_key"] == "2025_01") & (df["driver_id"] == "d1")]
    assert pole["quali_gap"].iloc[0] == 0.0
    assert (df["quali_gap"] >= 0).all()


def test_one_winner_per_race_and_retirements_flagged(data_dir):
    df = load_races(data_dir)
    assert (df.groupby("race_key")["won"].sum() == 1).all()
    assert df.loc[df["driver_id"] == "d6", "dnf"].all()          # status "Retired"
    assert not df.loc[df["driver_id"] == "d1", "dnf"].any()


def test_pit_lane_start_goes_to_back(data_dir):
    res = pd.read_csv(data_dir / "results.csv")
    res.loc[(res["round"] == 1) & (res["driver_id"] == "d1"), "grid"] = 0
    res.to_csv(data_dir / "results.csv", index=False)
    df = load_races(data_dir)
    row = df[(df["race_key"] == "2025_01") & (df["driver_id"] == "d1")]
    assert row["grid"].iloc[0] == 6


def test_missing_qualifying_is_imputed_from_grid(data_dir):
    qual = pd.read_csv(data_dir / "qualifying.csv")
    qual[qual["round"] != 2].to_csv(data_dir / "qualifying.csv", index=False)
    df = load_races(data_dir)
    race2 = df[df["race_key"] == "2025_02"]
    assert race2["quali_imputed"].all()
    assert race2["quali_gap"].notna().all()
