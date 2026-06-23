"""
run_benchmark.py — Main benchmark, EQUAL WALL-CLOCK time budgets.

Every metaheuristic gets the same per-instance time budget (scaled by size);
we record the best objective found within budget. Removes the
per-iteration-work confound and is the rigorous comparison for the paper.

Methods: CW, VNS, ClassicalALNS (basic), ClassicalALNSPlus (strong LS),
DRL-ALNS+ (proposed, trained v4 model).

Resumable: each (instance, method, run) row is flushed immediately.
"""
import sys, os, csv, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gen_benchmark import make_instance
from src.alns          import ClassicalALNS
from src.alns_plus     import ClassicalALNSPlus, DRLALNSPlus, N_ACTIONS_PLUS
from src.vns           import VNS
from src.drl_vns       import DRLVNS, N_ACTIONS_VNS, STATE_DIM_VNS
from src.ppo           import PPOAgent
from src.clarke_wright import clarke_wright

BASE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(BASE, "results", "benchmark.csv")
MODEL = os.path.join(BASE, "models", "ppo_drl_alns.pt")
MODEL_VNS = os.path.join(BASE, "models", "ppo_drl_vns.pt")

# (size_class, n, time_budget_s)
GRID  = [("S",15,5),("S",20,6),("M",30,10),("M",50,15),
         ("L",75,25),("L",100,35),("XL",150,50)]
SEEDS = list(range(6))     # 6 instances per (class, n)
N_RUNS = 2
BIG = 10_000_000
FIELDS = ["instance","size_class","n","n_split_eligible","method","run",
          "Z_best","Z_init","time_s","budget_s"]


def done_keys(path):
    keys=set()
    if os.path.exists(path):
        with open(path,newline="") as f:
            for r in csv.DictReader(f):
                keys.add((r["instance"],r["method"],int(r["run"])))
    return keys


def main():
    agent = PPOAgent(n_actions=N_ACTIONS_PLUS)
    if os.path.exists(MODEL):
        agent.load(MODEL, for_inference=True); print("Loaded", MODEL, flush=True)
    else:
        print("WARNING: v4 model missing; DRL+ uses untrained agent", flush=True)
    agent_vns = PPOAgent(n_actions=N_ACTIONS_VNS, state_dim=STATE_DIM_VNS)
    if os.path.exists(MODEL_VNS):
        agent_vns.load(MODEL_VNS, for_inference=True); print("Loaded", MODEL_VNS, flush=True)
    else:
        print("WARNING: DRL-VNS model missing; uses untrained agent", flush=True)

    done = done_keys(OUT)
    new = not os.path.exists(OUT)
    fh = open(OUT,"a",newline=""); w=csv.DictWriter(fh,fieldnames=FIELDS)
    if new: w.writeheader(); fh.flush()

    def emit(inst,sc,n,method,run,Zb,Zi,t,tb):
        w.writerow({"instance":inst.name,"size_class":sc,"n":n,
                    "n_split_eligible":inst.params["n_split_eligible"],
                    "method":method,"run":run,"Z_best":round(Zb,4),
                    "Z_init":round(Zi,4),"time_s":round(t,2),"budget_s":tb})
        fh.flush()

    methods = [
        ("CW",    lambda inst,r,tb: (lambda s: {"Z_best":s,"Z_init":s,"time_s":0.0})(clarke_wright(inst,seed=0).objective())),
        ("VNS",   lambda inst,r,tb: VNS().solve(inst,I_max=BIG,seed=r,time_limit=tb)),
        ("ALNS",  lambda inst,r,tb: ClassicalALNS().solve(inst,I_max=BIG,seed=r,time_limit=tb)),
        ("ALNS+", lambda inst,r,tb: ClassicalALNSPlus().solve(inst,I_max=BIG,seed=r,time_limit=tb)),
        ("DRL+",  lambda inst,r,tb: DRLALNSPlus().solve(inst,agent,I_max=BIG,seed=r,train=False,time_limit=tb)),
        ("DRL-VNS",lambda inst,r,tb: DRLVNS().solve(inst,agent_vns,I_max=BIG,seed=r,train=False,time_limit=tb)),
    ]

    for sc,n,tb in GRID:
        for seed in SEEDS:
            inst = make_instance(n,size_class=sc,seed=seed)
            for mname,fn in methods:
                runs = 1 if mname=="CW" else N_RUNS
                for run in range(runs):
                    if (inst.name,mname,run) in done: continue
                    res = fn(inst,run,tb)
                    emit(inst,sc,n,mname,run,res["Z_best"],res["Z_init"],res["time_s"],tb)
            print(f"done {inst.name} (elig={inst.params['n_split_eligible']}/{n})",flush=True)
    fh.close()
    print("Benchmark v4 complete ->",OUT,flush=True)


if __name__ == "__main__":
    main()
