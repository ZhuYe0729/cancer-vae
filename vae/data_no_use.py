"""
    构造数据集(弃用)
    使用命令：./cancer_gillespie_simulation_no_display -o results/3/
"""
import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple


# default values (can be overridden with command line args)
start_index = 3
total_samples = 1


def find_next_index(results_dir: Path, min_index: int = 0) -> int:
    """Find the next numeric index under results_dir. Returns max(existing)+1 or min_index if none."""
    if not results_dir.exists():
        return min_index
    nums = []
    for child in results_dir.iterdir():
        if child.is_dir() and re.fullmatch(r"\d+", child.name):
            try:
                nums.append(int(child.name))
            except ValueError:
                continue
    if not nums:
        return min_index
    return max(max(nums) + 1, min_index)


def run_simulation_for_index(repo_root: Path, idx: int, execute: bool = False, force: bool = False) -> Tuple[int, str]:
    """Prepare results directory for idx, run the simulation (if execute=True) and save logs.

    Returns a tuple (rc, message). rc is the subprocess return code (0 means success). message is a short status string.
    """
    results_dir = repo_root / "results"
    target = results_dir / str(idx)
    target.mkdir(parents=True, exist_ok=True)

    # if target exists and not empty and not force, skip
    if any(target.iterdir()) and not force:
        return 0, f"[SKIP] results/{idx} already exists and is not empty (use --force to override)"

    exe = repo_root / "cancer_gillespie_simulation_no_display"
    if not exe.exists() or not os.access(exe, os.X_OK):
        raise FileNotFoundError(f"Executable not found or not executable: {exe}")

    cmd = [str(exe), "-o", str(target) + os.sep]

    log_file = target / "run.log"
    if not execute:
        # dry-run: just write the planned command to run.log
        with log_file.open("w") as f:
            f.write("DRY RUN\n")
            f.write(" ".join(cmd) + "\n")
        return 0, f"[DRY-RUN] planned: {cmd} -> {log_file}"

    with log_file.open("w") as f:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = proc.stdout.decode(errors="replace")
        f.write(out)
        rc = proc.returncode
    if rc != 0:
        return rc, f"[ERROR] index={idx} exited with code {rc}, see {log_file}"
    return 0, f"[OK] index={idx} finished, logs -> {log_file}"


def main():
    parser = argparse.ArgumentParser(description="Run cancer_gillespie_simulation_no_display multiple times and store outputs in results/<num>/")
    parser.add_argument("--start", type=int, default=None, help="minimum start index (will be used if greater than detected next index)")
    parser.add_argument("--n", type=int, default=total_samples, help="number of samples to generate")
    parser.add_argument("--execute", action="store_true", help="Actually run the simulation. By default this script does a dry-run and only creates directories and run.log entries.")
    parser.add_argument("--force", action="store_true", help="Overwrite / run even if target results/<num> is non-empty")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    min_start = args.start if args.start is not None else start_index
    next_idx = find_next_index(results_dir, min_start)

    # try to use tqdm for a progress bar; fall back gracefully if not installed
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    header = f"Repository root: {repo_root}\nDetected next start index: {next_idx}\nWill generate {args.n} samples (indexes {next_idx} .. {next_idx + args.n - 1})"
    if tqdm:
        tqdm.write(header)
    else:
        print(header)

    it = range(next_idx, next_idx + args.n)
    if tqdm:
        it = tqdm(it, desc="Generating samples", unit="sample")

    for i in it:
        try:
            rc, msg = run_simulation_for_index(repo_root, i, execute=args.execute, force=args.force)
        except Exception as e:
            rc = 1
            msg = f"[EXCEPTION] index={i}: {e}"

        # print or use tqdm.write so the progress bar is not corrupted
        if tqdm:
            tqdm.write(msg)
        else:
            print(msg)

        if rc != 0:
            stop_msg = f"Stopping on failure at index {i} (rc={rc})"
            if tqdm:
                tqdm.write(stop_msg)
            else:
                print(stop_msg)
            break


if __name__ == "__main__":
    main()



