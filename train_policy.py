"""
train_policy.py — train the PPO policy of DRL-VNS or DRL-ALNS.

    python train_policy.py --algo vns  --seed 1
    python train_policy.py --algo alns --seed 1

Protocol
  * one episode = one solve of a freshly generated training instance under
    the CPU-time budget of its size class (the same budgets as the benchmark);
    the policy is updated once at the end of every episode;
  * training sizes cycle through S-15, S-20, M-30, M-50, L-75; the benchmark
    sizes L-100 and XL-150 are never used for training;
  * training instance seeds are 100000 + 1000 * seed + episode, disjoint from
    the validation seeds (9000-9005) and the test seeds (0-5);
  * every VALID_EVERY episodes the policy is evaluated on six validation
    instances (S-20, M-50, L-75, two seeds each, search seed 0); the
    checkpoint with the highest mean improvement over Clarke-Wright is kept
    as <algo>_seed<k>_best.pt, the last one as <algo>_seed<k>_final.pt.
  * after training, the selected policy's action frequencies on the
    validation instances are saved (used by the Marginal-9 control).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(1)

from src.gen_benchmark import make_instance
from src.ppo import PPOAgent
from src.drl_vns import DRLVNS, N_ACTIONS_VNS, STATE_DIM_VNS
from src.alns_plus import DRLALNSPlus, N_ACTIONS_PLUS, STATE_DIM_PLUS
from src.budgets import BUDGET

EPISODES = 250
VALID_EVERY = 25
TRAIN_SIZES = [("S", 15), ("S", 20), ("M", 30), ("M", 50), ("L", 75)]
VALID_SET = [("S", 20, 9000), ("S", 20, 9001), ("M", 50, 9002),
             ("M", 50, 9003), ("L", 75, 9004), ("L", 75, 9005)]


def build(algo, n_actions=9):
    if algo == "vns":
        agent = PPOAgent(n_actions=n_actions, state_dim=STATE_DIM_VNS)
        return agent, DRLVNS(agent, n_actions=n_actions)
    agent = PPOAgent(n_actions=N_ACTIONS_PLUS, state_dim=STATE_DIM_PLUS)
    return agent, DRLALNSPlus(agent)


def validate(solver, valid):
    gains, counts = [], None
    for inst, n in valid:
        r = solver.solve(inst, seed=0, time_limit=BUDGET[n])
        gains.append((r["Z_init"] - r["Z_best"]) / r["Z_init"] * 100)
        if "uses" in r:
            counts = r["uses"] if counts is None else counts + r["uses"]
    return float(np.mean(gains)), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["vns", "alns"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out", default="models_v2")
    ap.add_argument("--valid-every", type=int, default=VALID_EVERY)
    ap.add_argument("--actions", type=int, default=9, choices=[9, 18])
    args = ap.parse_args()

    out = Path(__file__).parent / args.out
    out.mkdir(exist_ok=True)
    tag = f"{args.algo}{'' if args.actions == 9 else args.actions}_seed{args.seed}"
    logf = open(out / f"{tag}_train.log", "w", encoding="utf-8", buffering=1)
    curve = []

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent, solver = build(args.algo, args.actions)
    n_params = sum(p.numel() for p in agent.net.parameters())
    valid = [(make_instance(n, sc, seed=s), n) for sc, n, s in VALID_SET]
    log(f"train {tag}: {args.episodes} episodes, network parameters {n_params}")

    best_v, best_ep = -1e9, -1
    steps = 0
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        sc, n = TRAIN_SIZES[ep % len(TRAIN_SIZES)]
        inst = make_instance(n, sc, seed=100000 + 1000 * args.seed + ep)
        agent.net.train()
        r = solver.solve(inst, seed=args.seed * 100000 + ep,
                         time_limit=BUDGET[n], train=True)
        steps += r["iters"]
        gain = (r["Z_init"] - r["Z_best"]) / r["Z_init"] * 100
        rec = {"episode": ep + 1, "n": n, "steps": r["iters"],
               "train_gain": round(gain, 3),
               "entropy": round(r["train_stats"].get("entropy", 0.0), 4)}
        if (ep + 1) % args.valid_every == 0:
            agent.net.eval()
            v, _ = validate(solver, valid)
            rec["valid_gain"] = round(v, 3)
            mark = ""
            if v > best_v:
                best_v, best_ep = v, ep + 1
                agent.save(str(out / f"{tag}_best.pt"))
                mark = " (saved)"
            log(f"ep {ep+1:4d} n={n:3d} steps={r['iters']:5d} gain={gain:6.2f}% "
                f"valid={v:6.2f}% elapsed={(time.perf_counter()-t0)/60:.1f} min{mark}")
        curve.append(rec)
    agent.save(str(out / f"{tag}_final.pt"))

    # action frequencies of the selected policy on the validation set
    agent.load(str(out / f"{tag}_best.pt"), for_inference=True)
    v, counts = validate(solver, valid)
    info = {"algo": args.algo, "seed": args.seed, "episodes": args.episodes,
            "env_steps": steps, "network_parameters": n_params,
            "best_episode": best_ep, "best_valid_gain": best_v,
            "train_minutes": round((time.perf_counter() - t0) / 60, 2)}
    if counts is not None:
        info["valid_action_freq"] = (counts / counts.sum()).tolist()
    with open(out / f"{tag}_info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(out / f"{tag}_curve.json", "w") as f:
        json.dump(curve, f)
    log(f"done {tag}: best episode {best_ep}, valid {best_v:.2f}%, "
        f"{steps} environment steps, {info['train_minutes']} min")


if __name__ == "__main__":
    main()
