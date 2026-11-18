"""Check the archived delayed-cue experiment without training a model or importing PyTorch."""
import csv
import itertools
import math
from pathlib import Path
from statistics import fmean

MODES = ("choice", "texture", "shuffled", "cue_trace")
DELAYS = (2,4,8,16,32)
SEEDS = range(12)
UPDATES = (20,60,220)


def compare(label, value, expected):
    if not math.isclose(value, expected, rel_tol=0, abs_tol=.00501):
        raise ValueError(f"{label}: observed {value}, expected {expected}")


def main():
    cases = {}
    with Path(__file__).with_name("results.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row["seed"]),row["condition"],int(row["delay"]),int(row["update"])
            if key in cases:
                raise ValueError(f"Duplicate result {key}")
            cases[key] = row
    expected = set(itertools.product(SEEDS, MODES, DELAYS, UPDATES))
    if cases.keys() != expected:
        raise ValueError("Missing or unexpected seed, objective, delay or checkpoint")
    for (seed,mode,delay,update), row in cases.items():
        trials, correct, erased = (int(row[f]) for f in ("trials","correct","correct_erased"))
        if trials != 1024 or erased * 2 != trials or not 0 <= correct <= trials:
            raise ValueError(f"Invalid decision counts for {seed}, {mode}, {delay}, {update}")
        if update == 220 and correct != trials:
            raise ValueError(f"Final checkpoint below decision ceiling for {seed}, {mode}, {delay}")
    def accuracies(mode, delay, update):
        batch = [cases[s,mode,delay,update] for s in SEEDS]
        return [100 * int(r["correct"]) / int(r["trials"]) for r in batch]
    expected_table = {
        2: (100,99.10,100,92.47),
        8: (94.57,80.40,89.31,74.00),
        16: (95.15,79.17,88.24,66.69),
        32: (91.79,79.17,85.92,66.64),
    }
    for delay, targets in expected_table.items():
        for mode,target in zip(MODES,targets):
            compare(f"{mode}, delay {delay}",fmean(accuracies(mode,delay,60)),target)
    treatment = accuracies("texture",16,60)
    control = accuracies("choice",16,60)
    paired = fmean(t-c for t,c in zip(treatment,control))
    compare("paired percentage-point difference",paired,-15.97)
    counts = [int(cases[s,"texture",16,60]["correct"]) for s in SEEDS]
    if counts.count(1024) != 7 or sum(c in (512,513) for c in counts) != 5:
        raise ValueError("Unexpected seed-level distribution at 60 updates and delay 16")
    prediction = 100*fmean(float(cases[s,"texture",16,220]["next_texture_acc"]) for s in SEEDS)
    compare("final texture accuracy",prediction,79.89)
    print("Verified 720 rows, 48 final models and exact cue-erasure balance.")
    print(f"Delay 16 after 60 updates: {fmean(control):.2f}% choice only, {fmean(treatment):.2f}% next texture.")
    print(f"Paired recall difference {paired:.2f} points; final texture accuracy {prediction:.2f}%.")


if __name__ == "__main__":
    main()
