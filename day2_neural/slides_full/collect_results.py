"""Collect the reference-run numbers that the slides' result figures (make_figures.py) and the website use.

    python slides/collect_results.py RUN_DIR slides/reference_results.json

RUN_DIR holds the executed notebooks saved as out_<notebook>.ipynb and the run's results/day2 folder under
public/results/day2 (the leaderboard). The numbers are read from the notebooks' printed outputs.
"""
import json, re, sys, types
import pandas as pd

# ---------------------------------------------------------------- reading printed outputs

def outputs(nb_path):
    nb = json.load(open(nb_path))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            t = "".join(o.get("text", "")) or "".join(o.get("data", {}).get("text/plain", ""))
            if t:
                yield t

def find(nb_path, *must):
    for t in outputs(nb_path):
        if all(m in t for m in must):
            return t
    raise KeyError(f"{must} not in {nb_path}")

def table(text):
    """Parse a (possibly wrapped) pandas DataFrame repr into {index: {col: value}}."""
    blocks = [b for b in re.split(r"\n\s*\n", text.strip("\n")) if b.strip()]
    out, order = {}, []
    for b in blocks:
        lines = [l for l in b.split("\n") if l.strip()]
        cols = [c for c in re.split(r"\s{2,}", lines[0].strip().rstrip("\\").strip()) if c]
        for l in lines[1:]:
            parts = [p for p in re.split(r"\s{2,}", l.strip()) if p]
            if len(parts) == 1:          # index-name line
                continue
            idx, vals = parts[0], parts[1:]
            if len(vals) != len(cols):
                # index and first value separated by one space only
                continue
            if idx not in out:
                out[idx] = {}; order.append(idx)
            for c, v in zip(cols, vals):
                try: v = float(v)
                except ValueError: pass
                out[idx][c] = v
    return {k: out[k] for k in order}

def series(text):
    """Parse a pandas Series repr ('name   value' lines)."""
    out = {}
    for l in text.strip().split("\n"):
        m = re.match(r"^(\S+)\s+(-?[\d.]+)$", l.strip())
        if m: out[m.group(1)] = float(m.group(2))
    return out

P = types.SimpleNamespace(outputs=outputs, find=find, table=table, series=series)

# ---------------------------------------------------------------- the numbers
R = sys.argv[1]; nb = lambda n: f"{R}/out_{n}.ipynb"
lb = pd.read_csv(f"{R}/public/results/day2/leaderboard.csv")
fam = lambda lab, m: ("ens" if lab.startswith("07") else "fm" if lab.startswith("04") else "gbm" if m.startswith("LightGBM") else
                      "stat" if lab.startswith("05") or m == "AutoETS" else "base" if m in ("Naive", "SeasonalNaive") else "nn")
res = {"n_series": int(lb.n_series.max())}
res["lb"] = [[r.model, round(r.mean_rmsse, 3), round(r.median_rmsse, 3), round(r.wape, 3), fam(r.lab, r.model)]
             for r in lb.sort_values("mean_rmsse").itertuples()]
g = lb.set_index("model")
m = lambda k: round(float(g.loc[k, "mean_rmsse"]), 3)
sec = lambda k: round(float(g.loc[k, "fit_seconds"]), 1)
# Lab 01
res["l1"] = [[k, sec(k), m(k), "gbm" if k == "LightGBM" else "nn"] for k in ["DLinear", "NBEATS_interpretable", "NBEATS_generic", "NHITS", "LightGBM"]]
cv = re.findall(r"(\d{4}-\d\d-\d\d)\s+NHITS\s+([\d.]+)", P.find(nb("01_lab_gfm_default_training"), "cutoff"))
res["cv"] = [[d, round(float(v), 3)] for d, v in cv]
# Lab 02
t2 = "\n".join(P.outputs(nb("02_lab_hpo")))
sp = {}
for t in P.outputs(nb("02_lab_hpo")):
    s = P.series(t)
    if set(s) >= {"loss", "input_size", "learning_rate"}: sp = s
res["l2_spread"] = sorted([[k, round(v, 3)] for k, v in sp.items()], key=lambda x: -x[1])
res["l2_seed"] = round(float(re.search(r"seed spread ([\d.]+)", t2).group(1)), 4)
res["l2_loss"] = {k: round(float(v), 3) for k, v in re.findall(r"loss\s+(\S+)\s+RMSSE ([\d.]+)", t2)}
res["l2_trials_s"] = int(re.search(r"(\d+) trials in (\d+)s", t2).group(2))
res["l2_fanova"] = re.search(r"fANOVA importance: (.*)", t2).group(1) if "fANOVA" in t2 else None
res["l2_cost"] = [["default", sec("NHITS"), m("NHITS")], ["tuned", res["l2_trials_s"], m("NHITS_tuned")]]
# Lab 03
t3 = P.table(P.find(nb("03_lab_exogenous_response"), "median_elasticity"))
raw = float(re.search(r"lift \(train weeks.*?\): ([\d.]+)", P.find(nb("03_lab_exogenous_response"), "raw empirical")).group(1))
res["l3_lift"] = [["raw", raw]] + [[k, round(t3[k]["median_promo_lift"], 3)] for k in ["LightGBMx", "LightGBMx_mono", "NHITSx"]]
res["l3_resp"] = {k: {"perverse": v["share_perverse (e>0.05)"], "monotone": v["share_monotone_decreasing"], "elasticity": v["median_elasticity"]} for k, v in t3.items()}
cmp = lb.set_index("model").loc[["NHITS_noexog", "NHITSx"], ["mean_rmsse", "median_rmsse", "wape"]].round(3)
res["l3_cov"] = {k: cmp.loc[k].tolist() for k in cmp.index}
res["l3_better"] = float(re.search(r"NHITSx better on (\d+)%", P.find(nb("03_lab_exogenous_response"), "NHITSx better on")).group(1)) / 100
# Lab 04 (foundation models): validation table with cost, test table, validation RMSSE by demand type
n4 = nb("04_lab_foundation_models")
v4 = P.table(P.find(n4, "seconds_per_1k_series"))
t4 = P.table(P.find(n4, "n_series", "Chronos2_ft"))
d4 = P.table(P.find(n4, "demand_type", "Chronos2_zs_cov"))
res["l4"] = {"valid": {v["model"]: [v["mean_rmsse"], v["median_rmsse"], v["wape"], round(v["seconds_per_1k_series"], 1)] for v in v4.values()},
             "test": {v["model"]: [v["mean_rmsse"], v["median_rmsse"]] for v in t4.values()},
             "types": {t: {k: v for k, v in row.items()} for t, row in d4.items()},
             "ft_gain_valid": float(re.search(r"RMSSE gain ([-+][\d.]+)", P.find(n4, "fine-tuning: RMSSE gain")).group(1))}
# Lab 05
t5 = P.table(P.find(nb("05_lab_hierarchy_coherence"), "MinT-wls_var", "level"))
lv = list(t5); cols = list(t5[lv[0]])
res["l5"] = {"levels": lv, "base": [t5[l]["Base (incoherent)"] for l in lv],
             "m": {c: [t5[l][c] for l in lv] for c in cols if c != "Base (incoherent)"}}
h5 = re.search(r"\{'total': 1, 'department': (\d+), 'class': (\d+), 'subclass': (\d+), 'item': (\d+)\}", "\n".join(P.outputs(nb("05_lab_hierarchy_coherence"))))
res["hier5"] = [["Total", 1], ["Departments", int(h5.group(1))], ["Classes", int(h5.group(2))], ["Subclasses", int(h5.group(3))], ["Items", int(h5.group(4))]]
# Lab 06
t6 = P.table(P.find(nb("06_demo_uncertainty"), "coverage_80"))
res["l6"] = {k: [v["coverage_80"], v["coverage_95"], v["scaled_interval_score_80"], v["scaled_crps"]] for k, v in t6.items()}
# Lab 07
dtab = P.table(P.find(nb("07_lab_model_selection"), "demand_type", "smooth", "intermittent"))
bt = P.table(P.find(nb("07_lab_model_selection"), "bradley_terry"))
res["l7_bt"] = [[k, round(v["bradley_terry"]), v["mean_rmsse"], v["avg_win_rate"]] for k, v in bt.items()]
models6 = [c for c in next(iter(dtab.values())) if c != "n"]
res["l7_types"] = {"models": models6, "rows": [[t, [dtab[t][c] for c in models6], int(dtab[t]["n"])] for t in ["smooth", "erratic", "lumpy", "intermittent"]]}
mean = P.series(P.find(nb("07_lab_model_selection"), "SeasonalNaive", "dtype: float64"))
res["l7_mean"] = sorted([[k, v] for k, v in mean.items()], key=lambda x: x[1])
ev = P.table(P.find(nb("07_lab_model_selection"), "SEL_by_class", "median_rmsse", "wape"))
res["l7_eval"] = {v["model"]: v["mean_rmsse"] for v in ev.values()}
single = {k: v for k, v in res["l7_eval"].items() if not k.startswith(("ENS", "SEL"))}
best = min(single, key=single.get)
res["l7_heldout"] = {"top3": round(res["l7_eval"]["ENS_mean_top3"], 3), "sel": round(res["l7_eval"]["SEL_by_class"], 3),
                     "best_single": [best, round(single[best], 3)], "lightgbm": round(res["l7_eval"].get("LightGBM", float("nan")), 3)}
t7 = "\n".join(P.outputs(nb("07_lab_model_selection")))
res["l7_top3"] = eval(re.search(r"top-3 on selection set: (\[.*\])", t7).group(1))
res["l7_sel"] = eval(re.search(r"per-class choice: (\{.*\})", t7).group(1))
# share of summed RMSSE carried by intermittent series, per model (from the demand-type table)
tot = {c: sum(dtab[t][c] * dtab[t]["n"] for t in dtab) for c in models6}
res["l7_int_share"] = {c: round(dtab["intermittent"][c] * dtab["intermittent"]["n"] / tot[c], 3) for c in models6}
res["l7_int_n_share"] = round(dtab["intermittent"]["n"] / sum(dtab[t]["n"] for t in dtab), 3)
# Full metric suite on the test window (written by Lab 07) and the leaderboard's bias column
ms = pd.read_csv(f"{R}/public/results/day2/metrics_test.csv")
res["suite"] = {r["model"]: {k: (round(float(v), 4) if isinstance(v, (int, float)) and v == v else None)
                             for k, v in r.items() if k != "model"} for r in ms.to_dict("records")}
res["lb_bias"] = {r.model: round(float(r.bias_pct), 4) for r in lb.itertuples() if r.bias_pct == r.bias_pct}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "lb"}, indent=None)[:4000])
