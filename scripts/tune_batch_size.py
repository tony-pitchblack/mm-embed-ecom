"""Binary-search the largest batch_size_per_device that fits on the GPU
by launching train.py with PROBE_STEPS and reading torch.cuda.max_memory_allocated
from the subprocess.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path

import yaml


PROBE_RE = re.compile(
    r"PROBE_RESULT peak_mib=(\d+) reserved_mib=(\d+) bs=(\d+)"
)
OOM_RE = re.compile(r"(out of memory|CUDA out of memory|CUDA_ERROR_OUT_OF_MEMORY)", re.I)


def probe(
    base_cfg: dict,
    bs: int,
    work_dir: Path,
    gpu_idx: int,
    pairs_limit: int,
    probe_steps: int,
    timeout_s: float,
):
    cfg = dict(base_cfg)
    cfg["batch_size_per_device"] = int(bs)
    cfg["epochs"] = 1
    cfg["limit_train_pos_pairs_per_query"] = pairs_limit
    cfg["limit_val_pos_pairs_per_query"] = pairs_limit
    cfg["limit_test_pos_pairs_per_query"] = pairs_limit

    tmp_cfg = work_dir / f"bs{bs}.yml"
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(cfg, f)

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = ""
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    env["PROBE_STEPS"] = str(probe_steps)
    env["PYTHONUNBUFFERED"] = "1"

    log_path = work_dir / f"bs{bs}.log"
    with open(log_path, "wb") as log_f:
        try:
            proc = subprocess.run(
                ["python", "-u", "train.py", "--config", str(tmp_cfg)],
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = -1
            timed_out = True

    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = PROBE_RE.search(text)
    if m:
        peak_mib = int(m.group(1))
        reserved_mib = int(m.group(2))
        actual_bs = int(m.group(3))
        return "ok", peak_mib, reserved_mib, actual_bs
    if OOM_RE.search(text):
        return "oom", 0, 0, 0
    if timed_out:
        return "timeout", 0, 0, 0
    if rc != 0:
        return "error", 0, 0, 0
    return "no_probe", 0, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min-bs", type=int, default=4)
    ap.add_argument("--max-bs", type=int, default=2048)
    ap.add_argument("--pairs-limit", type=int, default=20)
    ap.add_argument("--probe-steps", type=int, default=3)
    ap.add_argument("--headroom-mib", type=int, default=1024)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--total-mib", type=int, default=24576)
    ap.add_argument("--work-dir", default="logs/bs_tune")
    args = ap.parse_args()

    base_cfg_path = Path(args.config)
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    results = []

    def run(bs):
        status, peak, reserved, actual_bs = probe(
            base_cfg,
            bs,
            work,
            args.gpu,
            args.pairs_limit,
            args.probe_steps,
            args.timeout_s,
        )
        results.append((bs, status, peak, reserved, actual_bs))
        print(
            f"bs={bs:>5d}  actual={actual_bs:>5d}  status={status:<8s}  "
            f"peak={peak:>6d} MiB  reserved={reserved:>6d} MiB",
            flush=True,
        )
        return status, peak

    lo_ok = 0
    hi_fail = None
    bs = args.min_bs
    while bs <= args.max_bs:
        s, p = run(bs)
        if s == "ok":
            lo_ok = bs
            bs *= 2
        else:
            hi_fail = bs
            break

    if hi_fail is None:
        print(f"\nno OOM up to bs={args.max_bs}; best ok = {lo_ok}")
    else:
        while hi_fail - lo_ok > 1:
            mid = (lo_ok + hi_fail) // 2
            s, p = run(mid)
            if s == "ok":
                lo_ok = mid
            else:
                hi_fail = mid

    print("\n--- summary ---")
    for bs_, s, p, r, ab in sorted(results):
        print(
            f"bs={bs_:>5d}  actual={ab:>5d}  {s:<8s}  peak={p:>6d} MiB  reserved={r:>6d} MiB"
        )
    print(f"\nmax fitting bs = {lo_ok}")
    ok = [(bs_, p) for bs_, s, p, _, _ in results if s == "ok"]
    if ok:
        safe = [bs_ for bs_, p in ok if (args.total_mib - p) >= args.headroom_mib]
        if safe:
            print(
                f"recommended bs (>= {args.headroom_mib} MiB free of {args.total_mib}): "
                f"{max(safe)}"
            )


if __name__ == "__main__":
    main()
