"""Result figures for our copy of the Day 2 slides, from the full-panel run (code_full).

Reads `results/day2_full/` (configs and the leaderboard), writes the numbers the slides quote
to `full_results.json` and the result charts to `img/res_*.png`, in the style of
`make_figures.py` (which still draws the theory figures; its own `res_*` charts are the v1
deck's results and are not run here).

    .venv/bin/python day2_neural/slides_full/make_result_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RUN = ROOT / "results" / "day2_full"
OUT = HERE / "img"

INK, ORANGE, BLUE, GREEN, GREY, SKY = "#0B2A3A", "#D55E00", "#0072B2", "#009E73", "#9AA7AE", "#56B4E9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold",
    "axes.titlesize": 16, "axes.titlecolor": INK, "savefig.dpi": 160,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


def cfg(name: str) -> dict:
    return json.loads((RUN / "configs" / f"{name}.json").read_text())


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def barh(ax, d: dict, colors=None, fmt="{:.3f}", ref=None):
    names, vals = list(d), list(d.values())
    ax.barh(names, vals, color=colors or INK)
    for k, v in enumerate(vals):
        ax.text(v, k, " " + fmt.format(v), va="center", fontsize=12, color=INK)
    if ref is not None:
        ax.axvline(ref, color=GREY, ls="--", lw=1)
    ax.invert_yaxis()


# ------------------------------------------------------------------ the numbers
def collect() -> dict:
    board = pd.read_csv(RUN / "leaderboard.csv").set_index("model")
    row = lambda m: {"median": round(float(board.loc[m, "median_rmsse"]), 3),
                     "mean": round(float(board.loc[m, "mean_rmsse"]), 3),
                     "wape": round(float(board.loc[m, "wape"]), 3)}
    c01, c03, c04, c05 = cfg("01_recipe"), cfg("03_features"), cfg("04_price"), cfg("05_selected")
    c05c, c06, c07, c07b = cfg("05c_input_clip"), cfg("06_clusters"), cfg("07_foundation"), \
        cfg("07b_new_foundation")
    c08, c09, c10, c11 = cfg("08_reconcile"), cfg("09_intervals"), cfg("10_final"), \
        cfg("11_all_series")
    response = {r["model"]: r for r in c10["price_response"]}
    standings = {r["model"]: r for r in c10["standings"]}
    return {
        "noise_floor": round(cfg("02_shortlist")["noise_floor"], 4),
        "day1": {m: row(m) for m in ["SeasonalNaive", "Naive", "AutoETS", "LightGBM"]},
        "lab01": {"median": {k: round(v, 3) for k, v in c01["median_rmsse"].items()},
                  "seed_spread": c01["seed_spread"], "minutes": c01["minutes_per_fit"]},
        "lab03": {"valid": c03["median_rmsse"]["valid"], "test": c03["median_rmsse"]["test"]},
        "lab04_price": {"test": {k: round(v, 3) for k, v in c04["median_rmsse"].items()},
                        "valid": {k: round(v, 3) for k, v in c04["median_rmsse_valid"].items()},
                        "e20": {k: round(c04["elasticity"][k]["median_e20"], 2)
                                for k in ["A_features", "B_cpi", "C_cpi_over"]},
                        "empirical": round(c04["empirical_all"], 2)},
        "lab05_hpo": {"n_trials": c05["n_trials"], "test": round(c05["median_rmsse"], 3),
                      "confirm": [{k: r[k] for k in ["role", "valid_rmsse", "test_rmsse",
                                                     "valid_e20", "test_e20", "selected"]}
                                  for r in c05["confirm"]]},
        "lab05c": {"valid": c05c["valid"], "explosions": c05c["explosions"],
                   "use": c05c["use"], "recommended": c05c["recommended"]},
        "lab06_clusters": c06["median_rmsse"],
        "lab07_fm": {"valid": c07b["median_rmsse_valid"], "test": c07b["median_rmsse_test"],
                     "seconds_2k": {k: round(v, 1) for k, v in c07b["seconds"].items()
                                    if k.endswith("_price_test")},
                     "history": c07b["history"], "gfm": c07["median_rmsse_test"]["GFM_selected"],
                     "lightgbm": c07["median_rmsse_test"]["LightGBM"]},
        "lab08": {"levels": list(c08["levels"]), "gfm": c08["median_rmsse_by_level"]["GFM"],
                  "chronos": c08["median_rmsse_by_level"]["Chronos"], "chosen": c08["chosen"],
                  "upper": c08["upper_model"]},
        "lab09": c09["scores"],
        "lab10": {"final": c10["final_model"], "top3": c10["ensembles"]["top3"],
                  "held_out": c10["ensembles"]["held_out"],
                  "standings": {m: {k: standings[m][k] for k in ["median_rmsse", "mean_rmsse",
                                                                 "avg_win_rate", "bradley_terry"]}
                                for m in list(standings)[:8]},
                  "response": {m: {k: response[m][k] for k in ["cut20 lift", "right sign"]}
                               for m in response if response[m]["cut20 lift"] is not None}},
        "lab11": {m: {k: round(v, 3) for k, v in c11["scores_all"][m].items()}
                  for m in ["ENS_mean_top3", "T0gamma_zs_cov", "Chronos2_zs_cov",
                            "GFM_selected", "Timestral_zs_cov", "NHITS_recipe", "LightGBM",
                            "SeasonalNaive"]},
    }


# ------------------------------------------------------------------ the figures
def res_day1(R):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    c = [GREY, GREY, GREY, BLUE]
    barh(axes[0], {m: v["mean"] for m, v in R["day1"].items()}, c, ref=1.0)
    axes[0].set_title("Mean RMSSE: pulled up by a few bad series")
    barh(axes[1], {m: v["median"] for m, v in R["day1"].items()}, c, ref=1.0)
    axes[1].set_title("Median RMSSE: the typical series")
    for ax in axes:
        ax.set_xlim(0, 1.9)
    save(fig, "res_day1")


def res_lab01(R):
    l1 = R["lab01"]
    names = {"NBEATSx_off": "NBEATSx, library defaults", "NBEATSx_recipe": "NBEATSx + recipe",
             "NHITS_recipe": "NHITS + recipe"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    vals = {names[k]: l1["median"][k] for k in names}
    barh(axes[0], vals, [GREY, BLUE, INK])
    lgb = R["day1"]["LightGBM"]["median"]
    axes[0].axvline(lgb, color=GREEN, ls="--", lw=1.5)
    axes[0].text(lgb, 2.45, f" LightGBM {lgb:.3f}", color=GREEN, fontsize=12)
    axes[0].set_xlim(0.4, 0.85); axes[0].set_title("Test median RMSSE (3 seeds)")
    spread = [l1["seed_spread"][k] for k in names]
    axes[1].bar(range(3), spread, color=[GREY, BLUE, INK])
    for k, v in enumerate(spread):
        axes[1].text(k, v, f"{v:.3f}", ha="center", va="bottom", fontsize=13)
    axes[1].set_xticks(range(3), ["defaults", "recipe", "NHITS"])
    axes[1].set_title("Seed spread (max − min)"); axes[1].set_ylim(0, max(spread) * 1.25)
    save(fig, "res_lab01")


def res_lab02(R):
    h = R["lab05_hpo"]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    labels = {"accuracy_optimal": "accuracy-optimal (chosen)", "knee_gain": "knee (gain)",
              "knee_distance": "knee (distance)", "response_optimal": "most price-responsive",
              "incumbent": "incumbent (04, arm C)"}
    for r in h["confirm"]:
        col = BLUE if r["selected"] else INK
        ax.scatter(r["valid_rmsse"], r["valid_e20"], s=200, color=col, zorder=3)
        ax.text(r["valid_rmsse"] + 0.0006, r["valid_e20"], " " + labels.get(r["role"], r["role"]),
                va="center", fontsize=12, color=col)
    best = min(r["valid_rmsse"] for r in h["confirm"])
    ax.axvspan(best, best + R["noise_floor"], color=SKY, alpha=0.25, label="within the noise floor")
    ax.set_xlabel("validation median RMSSE (lower is better)")
    ax.set_ylabel("price elasticity at ±20%")
    ax.set_title(f"{h['n_trials']} trials, 3 seeds per finalist: the response barely moves")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    xs = [r["valid_rmsse"] for r in h["confirm"]]
    ax.set_xlim(min(xs) - 0.002, max(xs) + 0.012)
    save(fig, "res_lab02")


def res_lab03(R):
    l4, l3 = R["lab04_price"], R["lab03"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"wspace": 0.75})
    acc = {"calendar + hierarchy (03)": l3["test"]["NBEATSx_cal_hier"],
           "A: price features": l4["test"]["A_features"], "B: + CPI deflation": l4["test"]["B_cpi"],
           "C: + oversample promos": l4["test"]["C_cpi_over"]}
    barh(axes[0], acc, [GREY, INK, INK, BLUE]); axes[0].set_xlim(0.4, 0.7)
    axes[0].set_title("Test median RMSSE (C chosen on validation)")
    resp = R["lab10"]["response"]
    names = {"GFM_selected": "GFM (selected)", "Timestral_zs_cov": "Timestral",
             "Chronos2_zs_cov": "Chronos-2", "T0gamma_zs_cov": "t0-gamma", "LightGBM": "LightGBM"}
    lift = {"implied by the data": 1.25, **{names[m]: resp[m]["cut20 lift"] for m in names}}
    barh(axes[1], lift, [GREEN] + [BLUE, INK, INK, INK, INK], fmt="{:+.0%}")
    axes[1].set_xlim(0, 1.45); axes[1].set_title("Lift for a one-week 20% cut")
    save(fig, "res_lab03")


def res_lab05c(R):
    s = R["lab05c"]
    arms = {"05": "old model (raw prices)", "floor": "floor", "clean": "clean",
            "clean_floor": "clean + floor", "clean_floor_clip": "clean + floor + clip"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"wspace": 0.75})
    acc = {arms[a]: s["valid"][a]["median_rmsse"] for a in arms}
    barh(axes[0], acc, [GREY, INK, ORANGE, INK, BLUE]); axes[0].set_xlim(0.5, 0.63)
    axes[0].set_title("Validation median RMSSE")
    worst = {arms[a]: s["explosions"][a]["largest multiple"] for a in arms}
    ax = axes[1]
    ax.barh(list(worst), list(worst.values()), color=[GREY, INK, ORANGE, INK, BLUE])
    for k, (a, v) in enumerate(worst.items()):
        n = s["explosions"][[x for x in arms if arms[x] == a][0]]["items"]
        ax.text(v, k, f" {v:,.0f}× ({n:.0f} items)", va="center", fontsize=12, color=INK)
    ax.set_xscale("log"); ax.invert_yaxis(); ax.set_xlim(1, 1e11)
    ax.set_title("Worst forecast ÷ item's largest week")
    save(fig, "res_lab05c")


def res_lab04(R):
    f = R["lab07_fm"]
    models = {"Chronos2": "Chronos-2", "T0gamma": "t0-gamma", "Timestral": "Timestral"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    w, x = 0.38, np.arange(len(models))
    none = [f["test"][m]["none"] for m in models]; price = [f["test"][m]["price"] for m in models]
    ax.bar(x - w / 2, none, w, color=GREY, label="history only")
    ax.bar(x + w / 2, price, w, color=BLUE, label="+ price and promotion plan")
    for k in range(len(models)):
        ax.text(x[k] + w / 2, price[k] + 0.005, f"{price[k]:.3f}", ha="center", va="bottom",
                fontsize=11)
    ax.axhline(f["gfm"], color=ORANGE, ls="--", lw=1.5, label=f"GFM selected ({f['gfm']:.3f})")
    ax.set_xticks(x, list(models.values())); ax.set_ylim(0.4, 0.7)
    ax.set_ylabel("test median RMSSE"); ax.set_title("Zero-shot: the covariates are the lever")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax = axes[1]
    for m, lab in models.items():
        sec = f["seconds_2k"][f"{m}_price_test"] / 2
        ax.scatter(sec, f["test"][m]["price"], s=160, color=BLUE, zorder=3)
        ax.text(sec * 1.12, f["test"][m]["price"], lab, va="center", fontsize=12)
    for k, col, v in [("GFM selected", ORANGE, f["gfm"]), ("LightGBM", GREEN, f["lightgbm"])]:
        ax.axhline(v, color=col, ls="--", lw=1.2)
        ax.text(0.55, v + 0.002, k, color=col, fontsize=11, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.5, 20)
    ax.set_xlabel("GPU seconds per 1,000 series (log)"); ax.set_ylabel("test median RMSSE")
    ax.set_title("No training, seconds to forecast")
    save(fig, "res_lab04")


def res_lab05(R):
    h = R["lab08"]
    lv = h["levels"]
    base = np.array([h["gfm"][l]["Base (incoherent)"] for l in lv])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    styles = {"BottomUp": BLUE, "TopDown": ORANGE, "MinT-wls_var": GREEN, "MinT-shrink-sub": INK}
    for name, c in styles.items():
        ax.plot(lv, np.array([h["gfm"][l][name] for l in lv]) / base, marker="o", color=c, lw=2.5,
                label=name)
    ax.axhline(1, color=GREY, lw=1, ls="--")
    ax.text(len(lv) - 0.95, 1.0, "base", va="center", fontsize=12)
    ax.set_ylabel("RMSSE ÷ base (lower is better)")
    ax.set_title("GFM items, ETS above: the sum of the items is hard to beat")
    ax.legend(frameon=False, ncol=4, fontsize=12, loc="upper left")
    save(fig, "res_lab05")


def res_lab06(R):
    s = R["lab09"]
    names = {"GFM_quantile": "GFM, quantile loss", "GFM_conf_step": "GFM, conformal by week",
             "GFM_conf_class": "GFM, conformal by class × week",
             "ETS_conf": "AutoETS, conformal"}
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for k, (m, lab) in enumerate(names.items()):
        ax.plot([s[m]["coverage_80"]], [k - 0.12], "o", ms=12, color=BLUE,
                label="80% interval" if k == 0 else None)
        ax.plot([s[m]["coverage_95"]], [k + 0.12], "s", ms=12, color=ORANGE,
                label="95% interval" if k == 0 else None)
    ax.axvline(0.8, color=BLUE, ls="--", lw=1.2); ax.axvline(0.95, color=ORANGE, ls="--", lw=1.2)
    ax.set_yticks(range(len(names)), [f"{lab}\ninterval score 80%: {s[m]['interval_score_80']:.1f}"
                                      for m, lab in names.items()])
    ax.invert_yaxis(); ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("share of test weeks inside the interval (dashed = promised)")
    ax.set_title("The quantile GFM is sharpest; conformal keeps its 80% promise")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    save(fig, "res_lab06")


def res_lab07(R):
    names = {"ENS_mean_top3": "average of top 3", "SEL_by_class": "best model per class",
             "T0gamma_zs_cov": "t0-gamma", "Chronos2_zs_cov": "Chronos-2",
             "GFM_selected": "GFM selected", "Timestral_zs_cov": "Timestral"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"wspace": 0.8})
    held = {lab: R["lab10"]["held_out"][m]["median_rmsse"] for m, lab in names.items()}
    barh(axes[0], held, [GREEN, GREY, BLUE, INK, ORANGE, INK]); axes[0].set_xlim(0.45, 0.55)
    axes[0].set_title("Held-out half: averaging beats picking")
    a = R["lab11"]
    alln = {"ENS_mean_top3": "average of top 3", "T0gamma_zs_cov": "t0-gamma",
            "Chronos2_zs_cov": "Chronos-2", "GFM_selected": "GFM selected",
            "Timestral_zs_cov": "Timestral", "NHITS_recipe": "NHITS + recipe",
            "LightGBM": "LightGBM"}
    barh(axes[1], {lab: a[m]["median_rmsse"] for m, lab in alln.items()},
         [GREEN, BLUE, INK, ORANGE, INK, INK, GREY]); axes[1].set_xlim(0.4, 0.62)
    axes[1].set_title("All 17,238 series, capped")
    save(fig, "res_lab07")


if __name__ == "__main__":
    R = collect()
    (HERE / "full_results.json").write_text(json.dumps(R, indent=1))
    for f in [res_day1, res_lab01, res_lab02, res_lab03, res_lab05c, res_lab04, res_lab05,
              res_lab06, res_lab07]:
        f(R)
    print("wrote full_results.json and 9 result figures to", OUT)
