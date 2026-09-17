"""
run_experiments.py — all heuristic experiments of the revised study.

    python run_experiments.py --suites main,exact,case --shard 0 --nshards 6

Suites
  main  : 42 benchmark instances (7 sizes x instance seeds 0-5), every method,
          10 runs each (search seeds 0-9).
  exact : the 20 small instances of the exact study (n = 6, 8, 10, 12;
          seeds 1-5), every method, 10 runs, 5 CPU seconds.
  case  : the 30-hospital Riyadh instance, every method, 10 runs, 30 s.
  poc   : price of control. Sizes 15, 30, 50, 75 x seeds 1-5 x
          {FULL, no-N1, no-N2, no-N3, SDVRP}; VNS, 5 runs.
  sens  : parameter sensitivity on sizes 30 and 75 (seeds 0-2) for
          beta_Q, mu, delta and U_max; VNS and DRL-VNS, 5 runs.
  csens : delta and U_max sensitivity on the Riyadh instance; VNS and
          DRL-VNS, 5 runs.

Paired design: run r of every method uses search seed r. Learned methods use
the policy trained with seed (r mod 5) + 1, so the 10 runs span five
independently trained policies. Marginal-9 run r samples actions from the
validation action frequencies of that same DRL-VNS policy.

The full job list is shuffled with a fixed seed and split into shards; each
shard appends to results_v2/runs_<suites>_shard<k>.csv and skips jobs that are
already there. Per-run action statistics and convergence histories go to a
matching .jsonl file.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(1)

from src.gen_benchmark import make_instance
from src.gen_exact_instance import make_exact_instance
from src.instance import CSDVRPInstance
from src.clarke_wright import clarke_wright
from src.vns import VNS
from src.alns_plus import ClassicalALNSPlus, DRLALNSPlus, N_ACTIONS_PLUS, STATE_DIM_PLUS
from src.drl_vns import VNS9, DRLVNS, N_ACTIONS_VNS, STATE_DIM_VNS
from src.ppo import PPOAgent
from src.budgets import BUDGET, EXACT_BUDGET, CASE_BUDGET

RES = ROOT / "results_v2"
MODELS = ROOT / "models_v2"
CASE_DIR = ROOT / "data" / "case_study_riyadh"

ALL_METHODS = ["VNS", "ALNS+", "DRL-ALNS", "DRL-VNS", "VNS-9",
               "Fixed-small", "Random-9", "Roulette-9", "Marginal-9"]
# the eighteen-action family, whose action set contains the moves of the
# standard VNS baseline (both insertion operators)
METHODS18 = ["DRL-VNS18", "VNS-18", "Fixed-dominant", "Random-18",
             "Roulette-18", "Marginal-18"]
GRID = [("S", 15), ("S", 20), ("M", 30), ("M", 50),
        ("L", 75), ("L", 100), ("XL", 150)]
N_RUNS = 10
N_POLICIES = 5

FIELDS = ["suite", "instance", "size_class", "n", "inst_seed", "config",
          "method", "run", "policy_seed", "budget_s", "Z_best", "Z_init",
          "routing", "emission", "split", "cpu_s", "wall_s", "iters",
          "routes", "split_customers", "extra_visits", "max_util",
          "demand_viol", "cap_viol", "n1_viol", "n2_viol",
          "dup_visit_viol", "zero_qty_stops"]


# instances
def case_instance(delta=None, umax=None):
    nodes = pd.read_csv(CASE_DIR / "nodes.csv")
    veh = pd.read_csv(CASE_DIR / "vehicles.csv")
    params = json.load(open(CASE_DIR / "params.json"))
    params["n"] = int((nodes["node_type"] == "customer").sum())
    if delta is not None:
        params["delta"] = float(delta)
    if umax is not None:
        m = (nodes.node_type == "customer") & (nodes.U_max > 1)
        nodes.loc[m, "U_max"] = int(umax)
    return CSDVRPInstance(nodes, veh, params)


def poc_variant(inst, mode):
    """Relax one control. Split-eligibility is stored as U_max > 1."""
    nodes = inst.nodes.copy()
    params = dict(inst.params)
    cust = nodes["node_type"] == "customer"
    elig = cust & (nodes["U_max"] > 1)
    u = int(inst.params["U_max"])
    if mode == "no-N1":                  # everyone eligible, cap kept
        nodes.loc[cust, "U_max"] = u
    elif mode == "no-N2":                # eligible customers uncapped
        nodes.loc[elig, "U_max"] = 999
    elif mode == "no-N3":
        params["delta"] = 0.0
    elif mode == "SDVRP":
        nodes.loc[cust, "U_max"] = 999
        params["delta"] = 0.0
    nodes.loc[cust, "alpha"] = (nodes.loc[cust, "U_max"] > 1).astype(int)
    return CSDVRPInstance(nodes, inst.vehicles.copy(), params)


def build_jobs(suites):
    """Job = (suite, instance spec, config, method, run, budget)."""
    jobs = []
    for suite in suites:
        if suite == "main":
            for sc, n in GRID:
                for s in range(6):
                    for m in ALL_METHODS:
                        for r in range(N_RUNS):
                            jobs.append((suite, ("bench", sc, n, s), "base", m, r, BUDGET[n]))
        elif suite == "exact":
            for n in (6, 8, 10, 12):
                for s in range(1, 6):
                    for m in ALL_METHODS:
                        for r in range(N_RUNS):
                            jobs.append((suite, ("exact", "S", n, s), "base", m, r, EXACT_BUDGET))
        elif suite == "case":
            for m in ALL_METHODS:
                for r in range(N_RUNS):
                    jobs.append((suite, ("case",), "base", m, r, CASE_BUDGET))
        elif suite == "family18":
            for sc, n in GRID:
                for s in range(6):
                    for m in METHODS18:
                        for r in range(N_RUNS):
                            jobs.append((suite, ("bench", sc, n, s), "base", m, r, BUDGET[n]))
            for n in (6, 8, 10, 12):
                for s in range(1, 6):
                    for m in METHODS18:
                        for r in range(N_RUNS):
                            jobs.append((suite, ("exact", "S", n, s), "base", m, r, EXACT_BUDGET))
            for m in METHODS18:
                for r in range(N_RUNS):
                    jobs.append((suite, ("case",), "base", m, r, CASE_BUDGET))
        elif suite == "poc":
            for sc, n in [("S", 15), ("M", 30), ("L", 50), ("L", 75)]:
                for s in range(1, 6):
                    for cfg in ["FULL", "no-N1", "no-N2", "no-N3", "SDVRP"]:
                        for r in range(5):
                            jobs.append((suite, ("bench", sc, n, s), cfg, "VNS", r, BUDGET.get(n, 15)))
        elif suite == "sens":
            settings = ([f"beta={b}" for b in (0.3, 0.5, 0.7)] +
                        [f"mu={x}" for x in (0.0, 0.2)] +
                        [f"delta={d}" for d in (0.0, 5.0, 30.0)] +
                        [f"umax={u}" for u in (2, 4)])
            for sc, n in [("M", 30), ("L", 75)]:
                for s in range(3):
                    for cfg in settings:
                        for m in ("VNS", "DRL-VNS"):
                            for r in range(5):
                                jobs.append((suite, ("bench", sc, n, s), cfg, m, r, BUDGET[n]))
        elif suite == "csens":
            settings = ([f"delta={d}" for d in (0, 5, 10, 20, 25, 30)] +
                        [f"umax={u}" for u in (1, 2)] + ["base"])
            for cfg in settings:
                for m in ("VNS", "DRL-VNS"):
                    for r in range(5):
                        jobs.append((suite, ("case",), cfg, m, r, CASE_BUDGET))
        else:
            raise ValueError(suite)
    order = np.random.default_rng(2026).permutation(len(jobs))
    return [jobs[i] for i in order]


def make_inst(spec, cfg):
    kind = spec[0]
    if kind == "case":
        if cfg.startswith("delta="):
            return case_instance(delta=float(cfg[6:])), 0
        if cfg.startswith("umax="):
            return case_instance(umax=int(cfg[5:])), 0
        return case_instance(), 0
    _, sc, n, s = spec
    if kind == "exact":
        return make_exact_instance(n, seed=s), s
    kw = {}
    if cfg.startswith("beta="):
        kw["beta_Q"] = float(cfg[5:])
    if cfg.startswith("umax="):
        kw["u_max"] = int(cfg[5:])
    inst = make_instance(n, sc, seed=s, **kw)
    if cfg.startswith("mu="):
        inst.mu = float(cfg[3:]); inst.params["mu"] = inst.mu
    if cfg.startswith("delta="):
        inst.delta = float(cfg[6:]); inst.params["delta"] = inst.delta
    if cfg in ("FULL", "no-N1", "no-N2", "no-N3", "SDVRP") and cfg != "FULL":
        inst = poc_variant(inst, cfg)
    return inst, s


class Policies:
    def __init__(self):
        self.vns, self.alns, self.freq = {}, {}, {}
        self.vns18, self.freq18, self._dom = {}, {}, None

    def vns18_agent(self, k):
        if k not in self.vns18:
            a = PPOAgent(n_actions=18, state_dim=STATE_DIM_VNS)
            a.load(str(MODELS / f"vns18_seed{k}_best.pt"), for_inference=True)
            self.vns18[k] = a
        return self.vns18[k]

    def marginal18(self, k):
        if k not in self.freq18:
            info = json.load(open(MODELS / f"vns18_seed{k}_info.json"))
            p = np.asarray(info["valid_action_freq"], float)
            self.freq18[k] = p / p.sum()
        return self.freq18[k]

    def dominant18(self):
        """The action the trained policies play most often on the validation
        instances; the control that always uses it."""
        if self._dom is None:
            fr = np.mean([self.marginal18(k) for k in range(1, N_POLICIES + 1)], axis=0)
            self._dom = int(np.argmax(fr))
        return self._dom

    def vns_agent(self, k):
        if k not in self.vns:
            a = PPOAgent(n_actions=N_ACTIONS_VNS, state_dim=STATE_DIM_VNS)
            a.load(str(MODELS / f"vns_seed{k}_best.pt"), for_inference=True)
            self.vns[k] = a
        return self.vns[k]

    def alns_agent(self, k):
        if k not in self.alns:
            a = PPOAgent(n_actions=N_ACTIONS_PLUS, state_dim=STATE_DIM_PLUS)
            a.load(str(MODELS / f"alns_seed{k}_best.pt"), for_inference=True)
            self.alns[k] = a
        return self.alns[k]

    def marginal(self, k):
        if k not in self.freq:
            info = json.load(open(MODELS / f"vns_seed{k}_info.json"))
            p = np.asarray(info["valid_action_freq"], float)
            self.freq[k] = p / p.sum()
        return self.freq[k]


def make_solver(method, k, pol):
    if method == "VNS":
        return VNS()
    if method == "ALNS+":
        return ClassicalALNSPlus()
    if method == "DRL-ALNS":
        return DRLALNSPlus(pol.alns_agent(k))
    if method == "DRL-VNS":
        return DRLVNS(pol.vns_agent(k))
    if method == "VNS-9":
        return VNS9("cyclic")
    if method == "Fixed-small":
        return VNS9("small")
    if method == "Random-9":
        return VNS9("random")
    if method == "Roulette-9":
        return VNS9("roulette")
    if method == "Marginal-9":
        return VNS9("marginal", marginal=pol.marginal(k))
    if method == "DRL-VNS18":
        return DRLVNS(pol.vns18_agent(k), n_actions=18)
    if method == "VNS-18":
        return VNS9("cyclic", n_actions=18)
    if method == "Fixed-dominant":
        return VNS9("fixed", n_actions=18, fixed_action=pol.dominant18())
    if method == "Random-18":
        return VNS9("random", n_actions=18)
    if method == "Roulette-18":
        return VNS9("roulette", n_actions=18)
    if method == "Marginal-18":
        return VNS9("marginal", n_actions=18, marginal=pol.marginal18(k))
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    suites = args.suites.split(",")
    jobs = build_jobs(suites)
    if args.list:
        cpu = sum(j[5] for j in jobs)
        print(f"{len(jobs)} jobs, {cpu/3600:.1f} CPU hours")
        return
    jobs = jobs[args.shard::args.nshards]

    RES.mkdir(exist_ok=True)
    tag = "_".join(suites)
    out_csv = RES / f"runs_{tag}_shard{args.shard}.csv"
    out_js = RES / f"runs_{tag}_shard{args.shard}.jsonl"
    done = set()
    if out_csv.exists():
        for r in csv.DictReader(open(out_csv, newline="")):
            done.add((r["suite"], r["instance"], r["config"], r["method"], int(r["run"])))
    new = not out_csv.exists()
    fc = open(out_csv, "a", newline="")
    w = csv.DictWriter(fc, fieldnames=FIELDS)
    if new:
        w.writeheader()
    fj = open(out_js, "a", encoding="utf-8")
    pol = Policies()
    cache = {}
    t0 = time.time()
    for i, (suite, spec, cfg, method, run, budget) in enumerate(jobs):
        key_inst = (spec, cfg)
        if key_inst not in cache:
            cache.clear()
            cache[key_inst] = make_inst(spec, cfg)
        inst, s = cache[key_inst]
        if (suite, inst.name, cfg, method, run) in done:
            continue
        k = run % N_POLICIES + 1
        uses_policy = method in ("DRL-VNS", "DRL-ALNS", "Marginal-9",
                                 "DRL-VNS18", "Marginal-18")
        solver = make_solver(method, k, pol)
        res = solver.solve(inst, seed=run, time_limit=budget)
        best = res["best"]
        comp = best.components()
        aud = best.audit()
        row = {"suite": suite, "instance": inst.name,
               "size_class": spec[1] if spec[0] != "case" else "case",
               "n": inst.n, "inst_seed": s, "config": cfg,
               "method": method, "run": run,
               "policy_seed": k if uses_policy else "",
               "budget_s": budget, "Z_best": round(res["Z_best"], 4),
               "Z_init": round(res["Z_init"], 4),
               "routing": round(comp["routing"], 4),
               "emission": round(comp["emission"], 4),
               "split": round(comp["split"], 4),
               "cpu_s": round(res["cpu_s"], 3), "wall_s": round(res["wall_s"], 3),
               "iters": res["iters"]}
        row.update({k2: aud[k2] for k2 in ("routes", "split_customers",
                    "extra_visits", "max_util", "demand_viol", "cap_viol",
                    "n1_viol", "n2_viol", "dup_visit_viol", "zero_qty_stops")})
        w.writerow(row)
        fc.flush()
        extra = {"suite": suite, "instance": inst.name, "config": cfg,
                 "method": method, "run": run,
                 "history": [(round(a, 3), round(b, 4)) for a, b in res["history"]]}
        for key in ("usage", "uses", "wins"):
            if key in res:
                extra[key] = np.asarray(res[key]).tolist()
        fj.write(json.dumps(extra) + "\n")
        fj.flush()
        if (i + 1) % 20 == 0:
            print(f"shard {args.shard}: {i+1}/{len(jobs)} jobs, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    print(f"shard {args.shard} finished {len(jobs)} jobs", flush=True)


if __name__ == "__main__":
    main()
