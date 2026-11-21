"""Check all overlapping fine/coarse observations and enumerate paired acquisition times."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate import boot_interval

ROOT = Path(__file__).resolve().parent
SEEDS = range(7900, 7916)
RULES = ("ten_update_grid", "endpoint_pair_ten_apart", "eleven_uninterrupted", "two_uninterrupted")


def records(filename):
    with (ROOT / filename).open() as f:
        return list(csv.DictReader(f))


def event_time(history, rule):
    """Return confirmation time plus event flag; history indexed by completed update."""
    if rule == "ten_update_grid":
        checks = range(20, 201, 10)
    elif rule == "two_uninterrupted":
        checks = range(2, 201)
    else:
        checks = range(11, 201)
    for t in checks:
        if rule in ("ten_update_grid", "endpoint_pair_ten_apart"):
            confirmed = history[t] and history[t - 10]
        elif rule == "eleven_uninterrupted":
            confirmed = all(history[t - 10:t + 1])
        elif rule == "two_uninterrupted":
            confirmed = history[t] and history[t - 1]
        else:
            raise ValueError(rule)
        if confirmed:
            return t, 1
    return 200, 0


def write(filename, rows):
    with (ROOT / filename).open("w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def main():
    fine = records("fine_resolution_curve.csv")
    if len(fine) != 32 * 200 or any(int(r["n"]) != 512 for r in fine):
        raise ValueError("Missing fine visits or wrong panel")
    observed = {(int(r["seed"]), r["mode"], int(r["update"])): int(r["correct"]) for r in fine}
    coarse = [r for r in records("curve_confirm_GRU.csv") if int(r["delay"]) == 16]
    if len(coarse) != 640:
        raise ValueError("Missing archived primary visits")
    for r in coarse:
        key = int(r["seed"]), r["mode"], int(r["update"])
        if observed[key] != int(r["correct"]) or int(r["n"]) != 512:
            raise ValueError(f"Trajectory differs at {key}")
    pairs, summary = [], []
    for rule in RULES:
        conditions = defaultdict(list)
        flags = defaultdict(list)
        for seed in SEEDS:
            for mode in ("cue", "auxiliary"):
                passed = [False] + [10 * observed[seed, mode, t] >= 9 * 512 for t in range(1, 201)]
                time, event = event_time(passed, rule)
                conditions[mode].append(time)
                flags[mode].append(event)
            pairs.append(dict(rule=rule, seed=seed, cue=conditions["cue"][-1],
                              auxiliary=conditions["auxiliary"][-1],
                              difference=conditions["auxiliary"][-1] - conditions["cue"][-1],
                              cue_event=flags["cue"][-1], auxiliary_event=flags["auxiliary"][-1]))
        delta = np.array(conditions["auxiliary"]) - np.array(conditions["cue"])
        low, high = boot_interval(delta)
        summary.append(dict(rule=rule, pairs=len(SEEDS), cue_events=sum(flags["cue"]),
                            auxiliary_events=sum(flags["auxiliary"]), cue_mean=np.mean(conditions["cue"]),
                            auxiliary_mean=np.mean(conditions["auxiliary"]), difference=np.mean(delta),
                            interval_lower=low, interval_upper=high, slower=int(sum(delta > 0)),
                            tied=int(sum(delta == 0)), faster=int(sum(delta < 0))))
    write("fine_resolution_pairs.csv", pairs)
    write("fine_resolution_summary.csv", summary)
    print("All 640 delay-16 archived visit counts matched.")


if __name__ == "__main__":
    main()
