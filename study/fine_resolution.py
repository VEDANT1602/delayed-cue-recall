"""Measure delay-16 symbolic recall after every optimizer update on the archived GRU seeds."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from timecourse import CueNet, sample

ROOT = Path(__file__).resolve().parent
SEEDS = range(7900, 7916)
HORIZON = 200


def trajectory(seed, mode):
    test_x, test_y, _ = sample(np.random.default_rng(2_000_000 + 41 * seed + 16), 16, 256, None)
    torch.manual_seed(seed + 100)
    model = CueNet("GRU", 4)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    training_rng = np.random.default_rng(40000 + seed)
    for step in range(1, HORIZON + 1):
        delay = int(training_rng.choice((2, 4, 8)))
        x, y, z = sample(training_rng, delay, 24, None)
        cue_logits, forecast_logits = model(x)
        loss = F.cross_entropy(cue_logits, y)
        if mode == "auxiliary":
            loss += 0.5 * F.cross_entropy(forecast_logits.flatten(0, 1), z.flatten())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            correct = int((model(test_x)[0].argmax(1) == test_y).sum())
        yield seed, mode, step, correct, len(test_y)


def run(output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mode", "update", "correct", "n"])
        for seed in SEEDS:
            for mode in ("cue", "auxiliary"):
                writer.writerows(trajectory(seed, mode))
            f.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "fine_resolution_curve.csv")
    run(p.parse_args().out)
