#!/usr/bin/env python
"""Chunked concurrent driver for tools/reeval_lr0.py.

Runs pending (file, problem, lr0) evals with a worker pool, skipping
outputs that already exist (checkpointing), and exits cleanly when the
wall-time budget is reached so it can be re-invoked to resume.

Usage:
    pixi run python tools/reeval_lr0_driver.py \
        --files 'runs/schedule_only_5hr/iter_*.py' \
        --problem results/problem_dei_n50.json --lr0 240 \
        --outdir results/lr0_rd_reeval/train \
        --minutes 7 --workers 6
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REEVAL = os.path.join(PROJECT_ROOT, "tools", "reeval_lr0.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True, help="Glob of schedule files")
    p.add_argument("--only", default=None,
                   help="Comma-separated basenames (no ext) to restrict to, "
                        "e.g. iter_192,iter_179")
    p.add_argument("--problem", required=True)
    p.add_argument("--lr0", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--minutes", type=float, default=7.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--job-timeout", type=int, default=180)
    args = p.parse_args()

    files = sorted(glob.glob(args.files))
    if args.only:
        keep = set(args.only.split(","))
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in keep]
    os.makedirs(args.outdir, exist_ok=True)

    pending = []
    for f in files:
        out = os.path.join(
            args.outdir, os.path.splitext(os.path.basename(f))[0] + ".json")
        if not os.path.exists(out):
            pending.append((f, out))

    t0 = time.time()
    deadline = t0 + args.minutes * 60
    running = {}  # proc -> (script, out, start)
    done = failed = 0
    launched_all = False

    def reap():
        nonlocal done, failed
        for proc in list(running):
            script, out, start = running[proc]
            if proc.poll() is not None:
                del running[proc]
                if os.path.exists(out):
                    try:
                        r = json.load(open(out))
                        if "error" in r:
                            failed += 1
                        else:
                            done += 1
                    except Exception:
                        failed += 1
                else:
                    failed += 1
                    with open(out, "w") as fh:
                        json.dump({"file": os.path.basename(script),
                                   "error": f"driver: exited rc={proc.returncode} "
                                            f"without output"}, fh)
            elif time.time() - start > args.job_timeout:
                proc.kill()
                del running[proc]
                failed += 1
                with open(out, "w") as fh:
                    json.dump({"file": os.path.basename(script),
                               "error": f"driver: timeout {args.job_timeout}s"}, fh)

    i = 0
    while True:
        reap()
        while (i < len(pending) and len(running) < args.workers
               and time.time() < deadline):
            script, out = pending[i]
            i += 1
            proc = subprocess.Popen(
                [sys.executable, REEVAL, script,
                 "--problem", args.problem, "--lr0", args.lr0, "--out", out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=PROJECT_ROOT)
            running[proc] = (script, out, time.time())
        if not running and (i >= len(pending) or time.time() >= deadline):
            launched_all = i >= len(pending)
            break
        time.sleep(2)

    remaining = len(pending) - i
    print(json.dumps({
        "outdir": args.outdir, "lr0": args.lr0,
        "total_files": len(files), "already_done": len(files) - len(pending),
        "completed_this_chunk": done, "failed_this_chunk": failed,
        "remaining": remaining, "all_launched": launched_all,
        "elapsed_s": round(time.time() - t0, 1),
    }))


if __name__ == "__main__":
    main()
