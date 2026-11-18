"""Rebuild the published figure from this repository's results.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TEAL, ORANGE, NAVY, SMOKE = "#006b68", "#ab4b26", "#223c57", "#717d87"
plt.rcParams.update({"font.size":9.5, "axes.spines.top":False, "axes.spines.right":False,
                     "figure.dpi":130, "savefig.dpi":250, "font.family":"DejaVu Sans"})


def read_results():
    with (ROOT / "results.csv").open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig):
    fig.tight_layout(pad=0.7)
    fig.savefig(ROOT / "figure.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def memory():
    tab=[r for r in read_results() if r["delay"]=="16"]
    fig,ax=plt.subplots(figsize=(6.3,2.55))
    for cond,label,color,marker in [("choice","Choice only",NAVY,"o"),("texture","Next texture",ORANGE,"s"),
            ("shuffled","Shuffled target",SMOKE,"^"),("cue_trace","Cue target",TEAL,"D")]:
        means=[]
        for step in (20,60,220):
            select=[r for r in tab if r["condition"]==cond and int(r["update"])==step]
            means.append(100*sum(int(r["correct"]) for r in select)/sum(int(r["trials"]) for r in select))
        ax.plot([20,60,220],means,label=label,color=color,marker=marker,ms=4)
    ax.set(xlabel="Training updates",ylabel="Correct choices (%)",xticks=[20,60,220],ylim=(44,104))
    ax.legend(frameon=False,ncol=2,fontsize=8)
    save_figure(fig)


if __name__ == "__main__":
    memory()
