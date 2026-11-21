"""Execute the publicly committed follow-up seed block without tuning."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import torch
from experiment import train

HERE=Path(__file__).resolve().parent

if __name__=="__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print("Started UTC",datetime.now(timezone.utc).isoformat(),flush=True)
    with (HERE/"confirm_runs.csv").open("w",newline="") as file:
        rows=csv.writer(file)
        rows.writerow(["width","lr","optimizer","seed","mode","correct","trials"])
        for seed in range(4000,4024):
            for mode in ("choice","texture"):
                correct,total,_=train(seed,mode,12,.003,"Adam",None)
                rows.writerow([12,.003,"Adam",seed,mode,correct,total])
    print("Completed UTC",datetime.now(timezone.utc).isoformat(),flush=True)
