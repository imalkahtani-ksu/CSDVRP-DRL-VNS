"""
analyze.py — aggregate the revised experiments and run the statistical tests.

    python analyze.py

Reads results_v2/runs_*.csv (+ .jsonl) and results_v2/exact_milp*.csv and
writes the tables used in the paper to results_v2/:

  audit.csv                every hard-constraint violation count (must be 0)
  per_instance.csv         mean / median / sd / min over the 10 runs
  summary.csv              per size class and method: improvement over CW
  statistics.csv           Wilcoxon signed-rank tests with Holm correction
  friedman.csv             Friedman test and average ranks
  policy_seeds.csv         DRL-VNS split by the policy it used
  exact_gaps.csv           gaps against the MILP optima
  case_study.csv           the Riyadh instance
  action_usage.csv         action use by phase, stagnation and size class
  action_success.csv       improvement rate per action
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_v2"
sys.stdout.reconfigure(encoding="utf-8")

METHODS = ["VNS", "ALNS+", "DRL-ALNS", "VNS-9", "Fixed-small", "Random-9",
           "Roulette-9", "Marginal-9", "DRL-VNS"]
PROPOSED = "DRL-VNS"
# the eighteen-action family; VNS is the shared external baseline
METHODS18 = ["VNS", "VNS-18", "Fixed-dominant", "Random-18", "Roulette-18",
             "Marginal-18", "DRL-VNS18"]
PROPOSED18 = "DRL-VNS18"
ALL = METHODS + [m for m in METHODS18 if m not in METHODS]
CLASSES = ["S", "M", "L", "XL"]
VIOL = ["demand_viol", "cap_viol", "n1_viol", "n2_viol",
        "dup_visit_viol", "zero_qty_stops"]


def load(pattern):
    frames = []
    for f in glob.glob(str(RES / pattern)):
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue          # file created but not yet written
        if len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_exact():
    """All exact runs, keeping for each instance the proven optimum if one
    exists and otherwise the longest attempt."""
    ex = pd.concat([load("exact_milp_shard*.csv"),
                    load("exact_milp_n12_shard*.csv"),
                    load("exact_milp_n12long_s*.csv")], ignore_index=True)
    if ex.empty:
        return ex
    ex["proven"] = ex["proven"].astype(bool)
    ex = ex.sort_values(["proven", "time_s"], ascending=[False, False])
    return ex.drop_duplicates(subset="instance", keep="first").reset_index(drop=True)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        prev = max(prev, val)
        adj[i] = min(1.0, prev)
    return adj


def paired_tests(piv, scope, rows, methods=None, proposed=None):
    methods = methods or METHODS
    proposed = proposed or PROPOSED
    ps, labels = [], []
    for m in methods:
        if m == proposed or m not in piv:
            continue
        pair = piv[[m, proposed]].dropna()
        if len(pair) < 5:
            continue
        a, b = pair[m].values, pair[proposed].values
        d = a - b
        if np.all(np.abs(d) < 1e-9):
            p, stat = 1.0, 0.0
        else:
            stat, p = wilcoxon(a, b, alternative="two-sided")
        nz = d[np.abs(d) > 1e-9]
        z = 0.0
        if len(nz):
            n = len(nz)
            z = abs(stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        rows.append({"scope": scope, "comparison": f"{proposed} vs {m}",
                     "N": len(d), "wins": int((d > 1e-9).sum()),
                     "losses": int((d < -1e-9).sum()),
                     "ties": int((np.abs(d) <= 1e-9).sum()),
                     "median_diff_pct": round(float(np.median(d / a * 100)), 3),
                     "p_raw": p, "effect_r": round(float(z / np.sqrt(max(len(nz), 1))), 3)})
        ps.append(p)
        labels.append(len(rows) - 1)
    if ps:
        for i, adj in zip(labels, holm(np.array(ps))):
            rows[i]["p_holm"] = adj
            rows[i]["sig"] = ("***" if adj < 0.001 else "**" if adj < 0.01
                              else "*" if adj < 0.05 else "n.s.")


def main():
    runs = pd.concat([load("runs_main_exact_case_shard*.csv"),
                      load("runs_family18_shard*.csv")], ignore_index=True)
    if runs.empty:
        print("no main results yet")
        return
    RES.mkdir(exist_ok=True)

    # 1. feasibility audit over every run of every suite
    allr = pd.concat([runs, load("runs_poc_shard*.csv"),
                      load("runs_sens_csens_shard*.csv")], ignore_index=True)
    aud = allr.groupby("method")[VIOL].sum()
    aud["runs"] = allr.groupby("method").size()
    aud.to_csv(RES / "audit.csv")
    print("=== feasibility audit ===")
    print(f"{len(allr)} runs, total violations {int(allr[VIOL].sum().sum())}, "
          f"max load/Q {allr.max_util.max():.4f}")

    main_df = runs[runs.suite.isin(["main", "family18"])]
    # 2. per instance statistics over the 10 runs
    per = (main_df.groupby(["instance", "size_class", "n", "method"])
           .agg(mean_Z=("Z_best", "mean"), median_Z=("Z_best", "median"),
                sd_Z=("Z_best", "std"), min_Z=("Z_best", "min"),
                runs=("Z_best", "size"), Z_cw=("Z_init", "mean"),
                mean_cpu=("cpu_s", "mean"), mean_iters=("iters", "mean"))
           .reset_index())
    per["cv_pct"] = per.sd_Z / per.mean_Z * 100
    per["impr_pct"] = (per.Z_cw - per.mean_Z) / per.Z_cw * 100
    per.to_csv(RES / "per_instance.csv", index=False)

    # 3. summary per class
    rows = []
    for sc in CLASSES + ["Overall"]:
        sub = per if sc == "Overall" else per[per.size_class == sc]
        for m in ALL:
            s = sub[sub.method == m]
            if not len(s):
                continue
            rows.append({"scope": sc, "method": m, "instances": len(s),
                         "mean_Z": round(s.mean_Z.mean(), 2),
                         "impr_over_CW_pct": round(s.impr_pct.mean(), 3),
                         "sd_across_instances": round(s.impr_pct.std(), 3),
                         "mean_run_cv_pct": round(s.cv_pct.mean(), 3),
                         "best_of_10_impr_pct": round(
                             ((s.Z_cw - s.min_Z) / s.Z_cw * 100).mean(), 3),
                         "mean_iters": int(s.mean_iters.mean())})
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "summary.csv", index=False)
    print("\n=== improvement over Clarke-Wright (%, mean of 10 runs) ===")
    tbl = summ.pivot(index="scope", columns="method",
                     values="impr_over_CW_pct").reindex(CLASSES + ["Overall"])
    print(tbl[[m for m in ALL if m in tbl]].round(2).to_string())

    # 4. paired tests on per-instance means
    piv = per.pivot(index=["instance", "size_class"], columns="method",
                    values="mean_Z").reset_index()
    stat_rows = []
    paired_tests(piv, "Overall", stat_rows)
    for sc in CLASSES:
        sub = piv[piv.size_class == sc]
        if len(sub) >= 5:
            paired_tests(sub, sc, stat_rows)
    if PROPOSED18 in piv:
        paired_tests(piv, "Overall", stat_rows, METHODS18, PROPOSED18)
        for sc in CLASSES:
            sub = piv[piv.size_class == sc]
            if len(sub) >= 5:
                paired_tests(sub, sc, stat_rows, METHODS18, PROPOSED18)
    st = pd.DataFrame(stat_rows)
    st.to_csv(RES / "statistics.csv", index=False)
    print("\n=== DRL-VNS vs each method (Wilcoxon, Holm-corrected) ===")
    print(st[st.scope == "Overall"][["comparison", "wins", "losses", "ties",
                                     "median_diff_pct", "p_holm", "sig"]]
          .to_string(index=False))

    # 5. Friedman test across all methods
    frows = []
    for sc in CLASSES + ["Overall"]:
        sub = piv if sc == "Overall" else piv[piv.size_class == sc]
        cols = [m for m in ALL if m in sub]
        data = [sub[m].values for m in cols]
        if len(sub) >= 5:
            chi, p = friedmanchisquare(*data)
            ranks = sub[cols].rank(axis=1).mean()
            frows.append({"scope": sc, "N": len(sub), "chi2": round(chi, 2),
                          "p": p, **{f"rank_{m}": round(ranks[m], 2) for m in cols}})
    pd.DataFrame(frows).to_csv(RES / "friedman.csv", index=False)

    # 6. variation across the five trained policies
    pol = (main_df[main_df.method.isin(["DRL-VNS", "DRL-ALNS", "DRL-VNS18"])]
           .groupby(["method", "policy_seed", "instance"])
           .agg(Z=("Z_best", "mean"), cw=("Z_init", "mean")).reset_index())
    pol["impr"] = (pol.cw - pol.Z) / pol.cw * 100
    pol.groupby(["method", "policy_seed"])["impr"].agg(
        ["mean", "std", "count"]).round(3).to_csv(RES / "policy_seeds.csv")

    # 7. exact gaps
    ex = load_exact()
    er = runs[runs.suite == "exact"]
    if not ex.empty and not er.empty:
        ref = ex.set_index("instance")
        g = (er.groupby(["instance", "n", "method"])
             .agg(mean_Z=("Z_best", "mean"), best_Z=("Z_best", "min"),
                  sd_Z=("Z_best", "std")).reset_index())
        g["MILP_Z"] = g.instance.map(ref.Z)
        g["MILP_LB"] = g.instance.map(ref.LB)
        g["proven"] = g.instance.map(ref.proven)
        base = np.where(g.proven, g.MILP_Z, g.MILP_LB)
        g["mean_gap_pct"] = (g.mean_Z - base) / base * 100
        g["best_gap_pct"] = (g.best_Z - base) / base * 100
        g.to_csv(RES / "exact_gaps.csv", index=False)
        pr = g[g.proven == True]
        worst = pr.mean_gap_pct.min() if len(pr) else 0.0
        if worst < -0.01:
            bad = pr[pr.mean_gap_pct < -0.01]
            print(f"\nWARNING: {len(bad)} heuristic means below a proven optimum "
                  f"(min {worst:.3f}%) — the heuristics and the MILP would not be "
                  f"solving the same problem:\n{bad[['instance','method','mean_gap_pct']]}")
        print("\n=== gap to proven optimum (%) ===")
        gp = pr.pivot_table(index="n", columns="method", values="mean_gap_pct")
        print(gp[[m for m in ALL if m in gp]].round(2).to_string())

    # 8. case study
    cs = runs[runs.suite == "case"]
    if not cs.empty:
        c = (cs.groupby("method").agg(mean_Z=("Z_best", "mean"),
                                      median_Z=("Z_best", "median"),
                                      sd_Z=("Z_best", "std"),
                                      best_Z=("Z_best", "min"),
                                      cw=("Z_init", "mean")).reset_index())
        c["impr_pct"] = (c.cw - c.mean_Z) / c.cw * 100
        c.round(3).to_csv(RES / "case_study.csv", index=False)
        print("\n=== Riyadh case study ===")
        print(c[["method", "mean_Z", "sd_Z", "best_Z", "impr_pct"]]
              .sort_values("mean_Z").round(2).to_string(index=False))

    # 9. action usage and success, from the jsonl side files
    inst_class = dict(zip(main_df.instance, main_df.size_class))
    usage = {}
    succ = {}
    for f in glob.glob(str(RES / "runs_main_exact_case_shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if "usage" not in d or d["instance"] not in inst_class:
                continue
            key = (d["method"], inst_class[d["instance"]])
            usage[key] = usage.get(key, 0) + np.asarray(d["usage"])
            if "wins" in d:
                s = succ.setdefault(key, [np.zeros(9), np.zeros(9)])
                s[0] += np.asarray(d["uses"])
                s[1] += np.asarray(d["wins"])
    rows = []
    names = [f"{o}-{s}" for o in ("random", "related", "worst")
             for s in ("small", "medium", "large")]
    for (m, sc), u in usage.items():
        tot = u.sum()
        for ph in range(3):
            for sb in range(3):
                for a in range(9):
                    if u[ph, sb, a]:
                        rows.append({"method": m, "size_class": sc,
                                     "phase": ("early", "middle", "late")[ph],
                                     "stagnation": ("0-4", "5-19", "20+")[sb],
                                     "action": names[a],
                                     "share": u[ph, sb, a] / tot})
    pd.DataFrame(rows).to_csv(RES / "action_usage.csv", index=False)
    rows = []
    for (m, sc), (u, wn) in succ.items():
        for a in range(9):
            if u[a]:
                rows.append({"method": m, "size_class": sc, "action": names[a],
                             "uses": int(u[a]), "improvements": int(wn[a]),
                             "success_rate": wn[a] / u[a]})
    pd.DataFrame(rows).to_csv(RES / "action_success.csv", index=False)
    print("\nwrote tables to", RES)


if __name__ == "__main__":
    main()
