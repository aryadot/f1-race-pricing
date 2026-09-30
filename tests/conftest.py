"""Shared test setup: make src/ and scripts/ importable and build tiny fake datasets.

Tests never touch the real data/ folder, so they run anywhere, including CI.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TEAMS = ["red_bull", "mclaren", "ferrari"]


def fake_season(n_races: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six drivers on three teams; driver d1 always takes pole and wins."""
    res, qual = [], []
    for rnd in range(1, n_races + 1):
        for i in range(6):
            driver, team = f"d{i + 1}", TEAMS[i // 2]
            base = dict(season=2025, round=rnd, race_name=f"Race {rnd}", circuit_id=f"c{rnd}",
                        date=f"2025-0{min(rnd, 9)}-1{rnd % 10}", driver_id=driver, code=driver.upper(),
                        given_name="Given", family_name=f"Driver{i + 1}", constructor=team)
            res.append({**base, "position": i + 1, "grid": i + 1,
                        "status": "Finished" if i < 5 else "Retired", "laps": 50})
            lap = 90.0 + 0.3 * i
            qual.append({**base, "position": i + 1, "q1": f"1:{lap:06.3f}", "q2": None, "q3": None})
    return pd.DataFrame(res), pd.DataFrame(qual)


@pytest.fixture
def data_dir(tmp_path):
    res, qual = fake_season()
    res.to_csv(tmp_path / "results.csv", index=False)
    qual.to_csv(tmp_path / "qualifying.csv", index=False)
    return tmp_path
