"""
run_pool.py — run a list of commands in parallel, one per CPU.

    python tools/run_pool.py --jobs jobs.txt --cpus 0,2,4,6,8,10 --logdir logs_v2/pool

Each line of the jobs file is one command (run from the repository root).
Every child process is pinned to one logical CPU and runs at below-normal
priority. Budgets are measured in process CPU time, so other programs on the
machine do not change the results. While the pool runs, Windows is asked not
to go to sleep (SetThreadExecutionState).
"""
import argparse, ctypes, os, shlex, subprocess, sys, time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent


def keep_awake(on: bool):
    if os.name != "nt":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--cpus", required=True)
    ap.add_argument("--logdir", required=True)
    args = ap.parse_args()

    cpus = [int(c) for c in args.cpus.split(",")]
    jobs = [l.strip() for l in open(args.jobs, encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    summary = open(logdir / "pool_summary.txt", "a", encoding="utf-8", buffering=1)
    summary.write(f"{time.ctime()} start {len(jobs)} jobs on cpus {cpus}\n")

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    keep_awake(True)
    running = {}          # cpu -> (Popen, job index, file)
    queue = list(enumerate(jobs))
    try:
        while queue or running:
            for cpu in cpus:
                if cpu in running or not queue:
                    continue
                idx, cmd = queue.pop(0)
                fh = open(logdir / f"job{idx:03d}.log", "w", encoding="utf-8")
                fh.write(f"# {cmd}\n# cpu {cpu}\n")
                fh.flush()
                argv = shlex.split(cmd, posix=False)
                if argv and argv[0] == "python":
                    argv[0] = sys.executable
                p = subprocess.Popen(argv, cwd=ROOT,
                                     stdout=fh, stderr=subprocess.STDOUT, env=env)
                try:
                    ps = psutil.Process(p.pid)
                    ps.cpu_affinity([cpu])
                    if os.name == "nt":
                        ps.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except psutil.Error:
                    pass
                running[cpu] = (p, idx, fh)
                summary.write(f"{time.ctime()} start job{idx:03d} cpu{cpu}: {cmd}\n")
            time.sleep(2)
            for cpu in list(running):
                p, idx, fh = running[cpu]
                if p.poll() is not None:
                    fh.close()
                    summary.write(f"{time.ctime()} end job{idx:03d} "
                                  f"exit={p.returncode}\n")
                    del running[cpu]
            keep_awake(True)
    finally:
        keep_awake(False)
    summary.write(f"{time.ctime()} all jobs finished\n")


if __name__ == "__main__":
    main()
