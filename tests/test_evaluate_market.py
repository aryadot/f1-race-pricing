import numpy as np
import pandas as pd

import evaluate
import market


def preds(p_winner):
    """Two races, two drivers each; the listed probability goes to the actual winner."""
    rows = []
    for r, p in enumerate(p_winner):
        rows += [{"race_key": f"r{r}", "driver_id": "a", "p_win": p, "won": True},
                 {"race_key": f"r{r}", "driver_id": "b", "p_win": 1 - p, "won": False}]
    return pd.DataFrame(rows)


def test_compare_is_negative_when_first_model_is_better():
    out = evaluate.compare(preds([0.8, 0.9]), preds([0.5, 0.5]), n_boot=200)
    assert out["diff"] < 0
    assert np.isclose(out["mean_b"], -np.log(0.5))


def test_calibration_counts_every_driver():
    cal = evaluate.calibration(preds([0.8, 0.9]))
    assert cal["drivers"].sum() == 4


def test_blend_weight_zero_equals_market():
    d = preds([0.6, 0.7])
    curve = market.blend_curve(d["p_win"] * 0 + 0.5, d["p_win"], d["won"], d["race_key"])
    assert np.isclose(curve.loc[curve["model_weight"] == 0, "log_loss"].iloc[0],
                      -np.log([0.6, 0.7]).mean())


def test_backtest_profit_arithmetic():
    d = pd.DataFrame({"race_key": ["r0", "r0", "r1"], "won": [True, False, False],
                      "model": [0.6, 0.6, 0.6], "market": [0.4, 0.4, 0.4]})
    out = market.backtest(d, "model", "market", edge=0.03, cost=0.0, n_boot=100)
    # three bets at 0.4: one pays 1/0.4 - 1 = 1.5, two lose 1 each
    assert out["bets"] == 3
    assert np.isclose(out["roi"], (1.5 - 2) / 3)


def test_backtest_skips_when_no_edge():
    d = pd.DataFrame({"race_key": ["r0"], "won": [True], "model": [0.40], "market": [0.40]})
    assert market.backtest(d, "model", "market", edge=0.03)["bets"] == 0
