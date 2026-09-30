"""Run the full analysis end to end and save every table and figure.

    python scripts/run_analysis.py            (reads data/, writes reports/)

Steps: build features, simulate every race under every parameter setting,
pick settings walk forward, score against a grid baseline, then compare
with Polymarket: accuracy, blend test, qualifying reaction, and backtest.
Takes a few minutes, mostly the simulations.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate  # noqa: E402
import market  # noqa: E402
import model  # noqa: E402
from data import load_races  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def clean(d: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating, int, np.integer)) else v)
            for k, v in d.items()}


def main() -> None:
    races = model.add_features(load_races(DATA))
    order = races.drop_duplicates("race_key")["race_key"].tolist()
    print(f"{len(order)} races, {races['driver_id'].nunique()} drivers")

    print("simulating (post qualifying model)...")
    post = model.walk_forward(model.score_all_settings(races, "post_quali"), order)
    print("simulating (pre weekend model)...")
    pre = model.walk_forward(model.score_all_settings(races, "pre_weekend"), order)
    base = evaluate.grid_baseline(races, order)

    results = {
        "model_vs_grid_baseline": {m: clean(evaluate.compare(post, base, m)) for m in ["log_loss", "brier"]},
        "pre_weekend_vs_post_quali_model": clean(evaluate.compare(pre, post)),
        "settings_chosen": post[["race_key", "beta", "gamma", "w"]].drop_duplicates("race_key")
                               .set_index("race_key").to_dict("index"),
    }
    cal_model = evaluate.calibration(post)
    cal_model.to_csv(REPORTS / "calibration_model.csv")

    # ---- market comparison (2025 onward)
    m = market.market_snapshots(DATA, races)
    m = (m.merge(post[["race_key", "driver_id", "p_win"]].rename(columns={"p_win": "model_post"}),
                 on=["race_key", "driver_id"])
          .merge(pre[["race_key", "driver_id", "p_win"]].rename(columns={"p_win": "model_pre"}),
                 on=["race_key", "driver_id"]))
    for c in ["model_post", "model_pre"]:  # compare over the same drivers the market lists
        m[c] = m[c] / m.groupby("race_key")[c].transform("sum")

    def pred(col):
        return m[["race_key", "driver_id", "won"]].assign(p_win=m[col].to_numpy())

    results["market"] = {
        "races": int(m["race_key"].nunique()),
        "post_quali_model_vs_market": {k: clean(evaluate.compare(pred("model_post"), pred("mkt_post_quali"), k))
                                       for k in ["log_loss", "brier"]},
        "pre_weekend_model_vs_market": {k: clean(evaluate.compare(pred("model_pre"), pred("mkt_pre_weekend"), k))
                                        for k in ["log_loss", "brier"]},
        "blend_in_sample": market.blend_curve(m["model_post"], m["mkt_post_quali"], m["won"], m["race_key"])
                                 .to_dict("records"),
        "qualifying_reaction": clean(market.qualifying_reaction(m)),
        "backtest_edge_0.03_primary": clean(market.backtest(m, "model_post", "mkt_post_quali", edge=0.03)),
        "backtest_edge_sensitivity": {str(e): clean(market.backtest(m, "model_post", "mkt_post_quali", edge=e))
                                      for e in [0.0, 0.05, 0.10]},
    }
    wf_blend = market.blend_walk_forward(m, "model_post", "mkt_post_quali")
    diff = (wf_blend["blend_loss"] - wf_blend["market_loss"]).to_numpy()
    boots = np.random.default_rng(0).choice(diff, (5000, len(diff))).mean(1)
    results["market"]["blend_walk_forward"] = clean({
        "races": len(diff), "blend_loss": wf_blend["blend_loss"].mean(),
        "market_loss": wf_blend["market_loss"].mean(), "diff": diff.mean(),
        "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)})
    cal_market = evaluate.calibration(pred("mkt_post_quali"))
    cal_market.to_csv(REPORTS / "calibration_market.csv")
    m.to_csv(REPORTS / "market_panel.csv", index=False)

    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=str))
    # same 39 races for both curves, so the chart is a fair head to head
    cal_model_market_races = evaluate.calibration(pred("model_post"))
    cal_model_market_races.to_csv(REPORTS / "calibration_model_market_races.csv")
    make_figures(cal_model_market_races, cal_market, m)
    print(json.dumps({k: v for k, v in results.items() if k != "settings_chosen"}, indent=2, default=str))


def make_figures(cal_model: pd.DataFrame, cal_market: pd.DataFrame, m: pd.DataFrame) -> None:
    # 1. Calibration: model and market
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    for cal, label, color in [(cal_model, "Model (post qualifying)", "#1a73e8"),
                              (cal_market, "Polymarket (post qualifying)", "#e8710a")]:
        ax.scatter(cal["predicted"], cal["observed"], s=np.sqrt(cal["drivers"]) * 6, color=color,
                   alpha=0.8, label=label)
        ax.plot(cal["predicted"], cal["observed"], color=color, alpha=0.5)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Win probability calibration, 39 races (2025 to 2026)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png", dpi=200)

    # 2. Qualifying reaction: size of the market's move vs what happened
    d = m.dropna(subset=["mkt_pre_weekend", "mkt_post_quali"]).copy()
    d["move"] = d["mkt_post_quali"] - d["mkt_pre_weekend"]
    d["surprise"] = d["won"].astype(float) - d["mkt_pre_weekend"]
    d = d[d["move"].abs() > 0.01]
    bins = pd.qcut(d["move"], 6, duplicates="drop")
    g = d.groupby(bins, observed=True)[["move", "surprise"]].mean()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = max(g.abs().max().max() * 1.2, 0.1)
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=1, label="Market moved the right amount")
    ax.scatter(g["move"], g["surprise"], color="#1a73e8", s=50, label="Drivers grouped by market move")
    ax.set_xlabel("Market move after qualifying (probability)")
    ax.set_ylabel("Outcome minus pre weekend price")
    ax.set_title("Does the market move enough after qualifying?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "qualifying_reaction.png", dpi=200)
    plt.close("all")


if __name__ == "__main__":
    main()
