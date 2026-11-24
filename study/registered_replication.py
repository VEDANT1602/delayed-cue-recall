"""Execute the 32-pair external protocol, or recheck completed trajectories."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aggregate import boot_interval

SEEDS = range(11900, 11932)
MODES = ("cue", "auxiliary")
HEADER = ("seed", "mode", "update", "correct", "n")


def summarize(rows, seeds):
    """Validate one complete panel per seed/arm; distinguish an event at 200."""
    histories = {(s, m): {} for s in seeds for m in MODES}
    for row in rows:
        seed, mode = int(row["seed"]), row["mode"]
        step, correct, n = (int(row[k]) for k in ("update", "correct", "n"))
        if (seed, mode) not in histories or n != 512 or not 0 <= correct <= n:
            raise ValueError(f"Unexpected panel at {seed}/{mode}/{step}")
        h = histories[seed, mode]
        if step in h or not 1 <= step <= 200:
            raise ValueError(f"Duplicate or out-of-range update at {seed}/{mode}/{step}")
        h[step] = correct

    pairs = []
    for s in seeds:
        record = {"seed": s}
        for mode in MODES:
            h = histories[s, mode]
            if len(h) != 200:
                raise ValueError(f"Incomplete history for {s}/{mode}")
            streak = 0
            time, event = 200, 0
            for t in range(1, 201):
                streak = streak + 1 if h[t] >= 461 else 0
                if streak == 11:
                    time, event = t, 1
                    break
            record[mode] = time
            record[mode + "_event"] = event
        record["difference"] = record["auxiliary"] - record["cue"]
        pairs.append(record)
    delta = np.array([p["difference"] for p in pairs])
    lo, hi = boot_interval(delta)
    summary = {"n_pairs": len(pairs), "difference": float(delta.mean()),
               "interval_lower": float(lo), "interval_upper": float(hi),
               "slower": int(sum(delta > 0)), "tied": int(sum(delta == 0)),
               "faster": int(sum(delta < 0))}
    for m in MODES:
        summary[m + "_mean"] = float(np.mean([p[m] for p in pairs]))
        summary[m + "_events"] = sum(p[m + "_event"] for p in pairs)
    return pairs, summary


def write_summary(folder):
    def rows():
        for seed in SEEDS:
            with (folder / f"seed-{seed}.csv").open() as f:
                yield from csv.DictReader(f)
    pairs, summary = summarize(rows(), SEEDS)
    with (folder / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pairs[0])
        writer.writeheader()
        writer.writerows(pairs)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run(folder, registration):
    from fine_resolution import trajectory
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder.mkdir(parents=True, exist_ok=True)
    run_record = folder / "execution.json"
    if run_record.exists():
        info = json.loads(run_record.read_text())
        if info["registration"] != registration:
            raise ValueError("Existing output directory points to a different registration")
    else:
        if any(folder.glob("seed-*.csv")):
            raise ValueError("Existing seed files have no execution record")
        info = {"registration": registration, "start_utc": datetime.now(timezone.utc).isoformat(),
                "torch": torch.__version__, "numpy": np.__version__}
        run_record.write_text(json.dumps(info, indent=2) + "\n")
    for seed in SEEDS:
        result = folder / f"seed-{seed}.csv"
        if result.exists():
            continue
        partial = result.with_suffix(".partial")
        with partial.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for mode in MODES:
                writer.writerows(trajectory(seed, mode))
        partial.replace(result)
    write_summary(folder)
    if "finish_utc" not in info:
        info["finish_utc"] = datetime.now(timezone.utc).isoformat()
        run_record.write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run", help="run after external registration")
    train.add_argument("--registration", required=True, help="archived registration URL")
    train.add_argument("folder", type=Path)
    check = commands.add_parser("summarize", help="reanalyze all 32 completed pairs without training")
    check.add_argument("folder", type=Path)
    a = parser.parse_args()
    if a.command == "run":
        run(a.folder, a.registration)
    else:
        write_summary(a.folder)
