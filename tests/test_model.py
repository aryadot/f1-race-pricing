import numpy as np
import pandas as pd

import model
from data import load_races


def draws(n):
    return model.race_draws(n, seed=0)


def test_probabilities_sum_correctly():
    noise, retire = draws(4)
    pace, grid, p_dnf = np.array([0.0, 0.5, 1.0, 1.5]), np.arange(1, 5.0), np.zeros(4)
    p_win, p_pod = model.simulate(pace, grid, p_dnf, 2.0, 0.1, noise, retire)
    assert np.isclose(p_win.sum(), 1.0)
    assert np.isclose(p_pod.sum(), 3.0)          # nobody retires, so exactly 3 podium spots


def test_faster_driver_wins_more_often():
    noise, retire = draws(3)
    p_win, _ = model.simulate(np.array([0.0, 1.0, 2.0]), np.ones(3), np.zeros(3), 2.0, 0.0, noise, retire)
    assert p_win[0] > p_win[1] > p_win[2]


def test_certain_retirement_means_no_win_or_podium():
    noise, retire = draws(4)
    p_dnf = np.array([1.0, 0.0, 0.0, 0.0])
    p_win, p_pod = model.simulate(np.zeros(4), np.arange(1, 5.0), p_dnf, 1.0, 0.1, noise, retire)
    assert p_win[0] == 0 and p_pod[0] == 0


def test_team_reliability_uses_only_earlier_races(data_dir):
    """Changing a race's own result must not change the retirement rate used to predict it."""
    before = model.add_features(load_races(data_dir))
    res = pd.read_csv(data_dir / "results.csv")
    res.loc[res["round"] == 3, "status"] = "Retired"
    res.to_csv(data_dir / "results.csv", index=False)
    after = model.add_features(load_races(data_dir))
    key = ["race_key", "driver_id"]
    b = before.set_index(key)["p_dnf"]
    a = after.set_index(key)["p_dnf"]
    assert np.allclose(b.xs("2025_03"), a.xs("2025_03"))       # same race: unchanged
    assert not np.allclose(b.xs("2025_04"), a.xs("2025_04"))   # later race: updated


def test_form_uses_only_earlier_races(data_dir):
    df = model.add_features(load_races(data_dir))
    first = df[df["race_key"] == "2025_01"]
    assert (first["form"] == 1.5).all()                         # no history yet: default


def test_walk_forward_picks_settings_from_past_only():
    """Setting A is best on early races, B only on the last; the last race must get A."""
    rows = []
    races = [f"r{i:02d}" for i in range(model.MIN_TRAIN_RACES + 1)]
    for i, r in enumerate(races):
        last = i == len(races) - 1
        for beta, p in [(1.0, 0.1 if last else 0.9), (2.0, 0.9 if last else 0.1)]:
            rows.append({"race_key": r, "beta": beta, "gamma": 0.0, "w": 1.0, "driver_id": "a",
                         "p_win": p, "p_podium": p, "won": True, "podium": True})
            rows.append({"race_key": r, "beta": beta, "gamma": 0.0, "w": 1.0, "driver_id": "b",
                         "p_win": 1 - p, "p_podium": 1 - p, "won": False, "podium": False})
    out = model.walk_forward(pd.DataFrame(rows), races)
    assert set(out["race_key"]) == {races[-1]}                  # early races are training only
    assert (out["beta"] == 1.0).all()
