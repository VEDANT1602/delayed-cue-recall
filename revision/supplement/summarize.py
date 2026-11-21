"""Summarize matched seeds; estimates and intervals use complete seed pairs."""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
def summarize(filename):
    grouped=defaultdict(lambda:defaultdict(dict))
    with (HERE/filename).open() as file:
        for row in csv.DictReader(file):
            key=(int(row["width"]),float(row["lr"]),row["optimizer"])
            grouped[key][row["mode"]][int(row["seed"])]=row
    out=[]
    for (width,lr,optimizer),conditions in grouped.items():
        for candidate in ("texture","gate"):
            if candidate not in conditions:
                continue
            for control in (["choice"] if candidate=="texture" else ["choice","texture"]):
                treatment=conditions[candidate]
                base=conditions[control]
                assert treatment.keys()==base.keys()
                seeds=sorted(base)
                def pct(row):
                    return 100*int(row["correct"])/int(row["trials"])
                a=np.array([pct(treatment[s]) for s in seeds])
                b=np.array([pct(base[s]) for s in seeds])
                d=a-b
                boot=np.random.default_rng(42117).choice(d,(20000,len(d)),replace=True).mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                out.append(dict(width=width,lr=lr,optimizer=optimizer,mode=candidate,control=control,
                                n=len(d),treatment_accuracy=a.mean(),control_accuracy=b.mean(),
                                difference_pp=d.mean(),lower_pp=lo,upper_pp=hi,
                                fewer=(d<0).sum(),tied=(d==0).sum(),more=(d>0).sum(),
                                cosine=(np.mean([float(v["mean_cosine"]) for v in treatment.values()]) if "mean_cosine" in treatment[seeds[0]] else ""),
                                fraction_negative=(np.mean([float(v["negative_fraction"]) for v in treatment.values()]) if "negative_fraction" in treatment[seeds[0]] else "")))
    return out

if __name__=="__main__":
    for fn in ["corridor","har","confirm"]:
        rows=summarize(fn+"_runs.csv")
        with (HERE/(fn+"_summary.csv")).open("w",newline="") as file:
            wr=csv.DictWriter(file,fieldnames=rows[0].keys())
            wr.writeheader()
            wr.writerows(rows)
