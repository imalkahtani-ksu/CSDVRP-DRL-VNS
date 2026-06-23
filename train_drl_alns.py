"""
train_drl_alns.py — Train the enhanced DRL-ALNS+ agent.

Action space = 18 (destroy x repair x removal-size tier); the agent learns
to adapt the perturbation strength as well as the operator pair, on top of
strong RVND local search. Saves models/ppo_drl_alns.pt.
"""
import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

_LOG = Path(__file__).parent / "train_drl_alns.log"
_fh  = open(_LOG, "w", encoding="utf-8", buffering=1)
def log(m): print(m, flush=True); _fh.write(m+"\n"); _fh.flush()

from src.gen_benchmark import make_instance
from src.ppo           import PPOAgent
from src.alns_plus     import DRLALNSPlus, N_ACTIONS_PLUS
from src.alns          import _imax_for_n

TRAIN_EPISODES = 240
VALID_EVERY    = 30
TRAIN_IMAX_CAP = 800            # strong LS => fewer iterations needed
MODEL_PATH     = Path(__file__).parent / "models" / "ppo_drl_alns.pt"
MODEL_PATH.parent.mkdir(exist_ok=True)

# Train on small/medium (strong LS is costly); state is size-normalized so the
# policy transfers to larger n at evaluation time.
TRAIN_SIZES = [("S",10),("S",15),("S",20),("M",30),("M",50)]
VALID_SIZES = [("S",15),("M",30),("M",50)]
MASTER = 42

def main():
    log("="*60)
    log("  DRL-ALNS+ Training v4 (18 actions, strong RVND local search)")
    log(f"  episodes={TRAIN_EPISODES} imax_cap={TRAIN_IMAX_CAP}")
    log("="*60)
    agent  = PPOAgent(n_actions=N_ACTIONS_PLUS)
    solver = DRLALNSPlus()
    valid  = [make_instance(n, sc, seed=9000+i*13) for i,(sc,n) in enumerate(VALID_SIZES)]
    best   = -1e9; t_start=time.perf_counter()
    for ep in range(TRAIN_EPISODES):
        sc, n = TRAIN_SIZES[ep % len(TRAIN_SIZES)]
        try:
            inst  = make_instance(n, size_class=sc, seed=MASTER+ep*97+3)
            I_max = min(_imax_for_n(n), TRAIN_IMAX_CAP)
            agent.net.train()
            t0=time.perf_counter()
            res = solver.solve(inst, agent, I_max=I_max, seed=MASTER+ep, train=True)
            el=time.perf_counter()-t0
            gap=(res["Z_best"]-res["Z_init"])/max(res["Z_init"],1e-6)*100
        except Exception as e:
            log(f"  Ep {ep+1:4d} ERROR ({n}): {e}"); continue
        if (ep+1)%VALID_EVERY==0:
            agent.net.eval(); vg=[]
            for vi in valid:
                vr=solver.solve(vi, agent, I_max=min(_imax_for_n(vi.n),TRAIN_IMAX_CAP),
                                seed=0, train=False)
                vg.append((vr["Z_init"]-vr["Z_best"])/max(vr["Z_init"],1e-6)*100)
            av=float(np.mean(vg)); mark=""
            if av>best: best=av; agent.save(str(MODEL_PATH)); mark=" *** SAVED"
            log(f"  Ep {ep+1:4d}/{TRAIN_EPISODES} | n={n:3d} | gap={gap:6.2f}% "
                f"| valid={av:6.2f}% | {el:.1f}s | total {(time.perf_counter()-t_start)/60:.1f}min{mark}")
        elif (ep+1)%5==0:
            log(f"  Ep {ep+1:4d}/{TRAIN_EPISODES} | n={n:3d} | gap={gap:6.2f}% | {el:.1f}s")
    if not MODEL_PATH.exists(): agent.save(str(MODEL_PATH))
    log(f"\n  Done in {(time.perf_counter()-t_start)/60:.1f} min. Best valid: {best:.2f}%")
    _fh.close()

if __name__ == "__main__":
    main()
