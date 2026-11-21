"""Fresh-seed and loss-weight checks using the pinned public reference implementation."""
import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path
import platform

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PIN = "fce228350f9c08560043d3ec5c33f04f987956de5e47f8e29b2be10f871e7598"


def load_study(source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != PIN:
        raise ValueError("Reference run.py differs from the pinned version")
    spec = importlib.util.spec_from_file_location("corridor_study", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def early_fit(study, seed, weight):
    torch.manual_seed(100 + seed)
    generator = np.random.default_rng(20000 + seed)
    model = study.ChoiceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(60):
        delay = int(generator.choice(study.TRAIN_DELAYS))
        inputs, cue, texture = study.batch(generator, 24, delay)
        generator.permutation(len(cue))  # Keep input-stream consumption matched to the reference.
        decision, predicted = model(inputs)
        loss = F.cross_entropy(decision, cue)
        if weight:
            loss = loss + weight * F.cross_entropy(predicted.reshape(-1,4),texture[:,1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    return study.evaluate(model, seed, 16)


def write_csv(name, rows):
    with (HERE/name).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path, help="reference repository run.py")
    study=load_study(parser.parse_args().reference)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows=[]
    for seed in range(1000,1024):
        for weight in (0,.125,.5,2.):
            correct, trials, erased, prediction=early_fit(study,seed,weight)
            assert trials==1024 and erased==512
            rows.append(dict(seed=seed,weight=weight,correct=correct,trials=trials,
                             erased_correct=erased,texture_accuracy=prediction))
    write_csv("fresh_seeds.csv",rows)
    controls=[100*r["correct"]/r["trials"] for r in rows if r["weight"]==0]
    summary=[]
    for weight in (.125,.5,2.):
        scores=[100*r["correct"]/r["trials"] for r in rows if r["weight"]==weight]
        delta=np.array(scores)-controls
        rng=np.random.default_rng(902491)
        lo, hi=np.quantile(rng.choice(delta,(20000,len(delta))).mean(1),[.025,.975])
        summary.append(dict(weight=weight,control_pct=np.mean(controls),texture_pct=np.mean(scores),
                            difference_pp=delta.mean(),lower_pp=lo,upper_pp=hi,negative=int((delta<0).sum()),
                            tied=int((delta==0).sum()),positive=int((delta>0).sum())))
    write_csv("summary.csv",summary)
    print(platform.python_version(), np.__version__, torch.__version__)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
