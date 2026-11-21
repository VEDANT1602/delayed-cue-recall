"""Reproduce a paired, synthetic delayed-choice experiment on CPU.

Run with Python 3.11, NumPy 1.26 and PyTorch 2.9: python run.py
"""
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
TRAIN_DELAYS = (2, 4, 8)
TEST_DELAYS = (2, 4, 8, 16, 32)
CONDITIONS = ("choice", "texture", "shuffled", "cue_trace")
SEEDS = range(12)
STEPS = 220


def batch(rng, pairs, delay):
    """Counterbalance each corridor's symbols across roles of the initial cue."""
    textures = np.empty((pairs, delay + 1), dtype=np.int64)
    textures[:, 0] = rng.integers(4, size=pairs)
    for step in range(1, delay + 1):
        turn = np.where(rng.random(pairs) < .8, 1, -1)
        textures[:, step] = (textures[:, step - 1] + turn) % 4
    textures = torch.from_numpy(np.tile(textures, (2, 1)))
    cues = torch.repeat_interleave(torch.arange(2), pairs)
    x = torch.zeros((pairs * 2, delay + 1, 6))
    x[:, 1:, :4] = F.one_hot(textures[:, :delay], 4).float()
    x[:, 0, 4:] = F.one_hot(cues, 2).float()
    return x, cues, textures


class ChoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.RNN(6, 12, nonlinearity="tanh", batch_first=True)
        self.choice = nn.Linear(12, 2)
        self.aux = nn.Linear(12, 4)

    def forward(self, x):
        states, _ = self.encoder(x)
        return self.choice(states[:, -1]), self.aux(states[:, 1:])


def fit(seed, mode):
    torch.manual_seed(100 + seed)
    rng = np.random.default_rng(20000 + seed)
    model = ChoiceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    snapshots = {}
    for update in range(1, STEPS+1):
        delay = int(rng.choice(TRAIN_DELAYS))
        x, cue, tex = batch(rng, 24, delay)
        shuffled = tex[rng.permutation(len(cue)), 1:]
        decision, predicted = model(x)
        loss = F.cross_entropy(decision, cue)
        if mode != "choice":
            targets = {"texture": tex[:, 1:], "shuffled": shuffled,
                       "cue_trace": cue[:, None].expand(-1, delay)}[mode]
            loss = loss + .5 * F.cross_entropy(predicted.reshape(-1, 4), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if update in (20, 60, STEPS):
            snapshots[update] = {key: tensor.detach().clone() for key,tensor in model.state_dict().items()}
    return model, snapshots


@torch.no_grad()
def evaluate(model, seed, delay):
    rng = np.random.default_rng(54000 + 100 * seed + delay)
    x, labels, textures = batch(rng, 512, delay)
    choice, aux = model(x)
    correct = int((choice.argmax(-1) == labels).sum().item())
    next_correct = (aux.argmax(-1) == textures[:, 1:]).float().mean().item()
    x[:, 0, 4:] = 0
    erased = int((model(x)[0].argmax(-1) == labels).sum().item())
    assert erased == 512
    return correct, 1024, erased, next_correct


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for seed, mode in itertools.product(SEEDS, CONDITIONS):
        model, snapshots = fit(seed, mode)
        for update, state in snapshots.items():
            model.load_state_dict(state)
            for delay in TEST_DELAYS:
                rows.append((seed, mode, delay, *evaluate(model, seed, delay), update))
    with (HERE / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "condition", "delay", "correct", "trials", "correct_erased", "next_texture_acc","update"])
        writer.writerows(rows)
    for update,delay in itertools.product((20,60,STEPS), TEST_DELAYS):
        scores = {m: np.array([r[3] / r[4] for r in rows if r[1:3] == (m, delay) and r[-1]==update]) for m in CONDITIONS}
        print(update, delay, {m: round(float(v.mean()), 4) for m, v in scores.items()})
        for m in CONDITIONS[1:]:
            diff = scores[m] - scores["choice"]
            sampler = np.random.default_rng(7400 + delay)
            boot = sampler.choice(diff, (20000, len(diff)), replace=True).mean(1)
            print(m, "paired mean difference and percentile interval", diff.mean(), np.quantile(boot, [.025, .975]))


if __name__ == "__main__":
    main()
