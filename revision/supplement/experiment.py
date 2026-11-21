"""Run paired early learning experiments. The HAR archive can be downloaded once and reused."""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import original_model as ref

ROOT = Path(__file__).resolve().parent
HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
HAR_SHA = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
MODES = ("choice","texture","gate")


class Network(nn.Module):
    def __init__(self, width, human):
        super().__init__()
        self.encoder=nn.RNN(3 if human else 6, width, nonlinearity="tanh",batch_first=True)
        self.choice=nn.Linear(width,6 if human else 2)
        self.aux=nn.Linear(width,3 if human else 4)

    def forward(self,x):
        states,_=self.encoder(x)
        return self.choice(states[:,-1]), self.aux(states)


def load_har(path):
    if not path.exists():
        urllib.request.urlretrieve(HAR_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=HAR_SHA:
        raise ValueError("Unexpected public HAR data digest")
    with zipfile.ZipFile(path) as outer:
        payload=outer.read("UCI HAR Dataset.zip")
    data={}
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        for split in ("train","test"):
            prefix="UCI HAR Dataset/"+split+"/"
            acc=[np.loadtxt(inner.open(prefix+f"Inertial Signals/total_acc_{axis}_{split}.txt"),dtype=np.float32)[:,::4] for axis in "xyz"]
            x=torch.tensor(np.stack(acc,axis=2))
            y=torch.tensor(np.loadtxt(inner.open(prefix+f"y_{split}.txt"),dtype=np.int64)-1)
            people=np.loadtxt(inner.open(prefix+f"subject_{split}.txt"),dtype=np.int64)
            data[split]=(x,y,people)
    assert not set(data["train"][2]) & set(data["test"][2])
    x=data["train"][0]
    mean=x.mean((0,1))
    sd=x.std((0,1),correction=0)
    return {split:((x-mean)/sd,y,person) for split,(x,y,person) in data.items()}


def cosine(loss_a, loss_b, parameters):
    ga=torch.cat([x.flatten() for x in torch.autograd.grad(loss_a,parameters,retain_graph=True)])
    gb=torch.cat([x.flatten() for x in torch.autograd.grad(loss_b,parameters,retain_graph=True)])
    return F.cosine_similarity(ga[None],gb[None],dim=1,eps=1e-12).item()


def train(seed, mode, width, lr, optimizer, data):
    human=data is not None
    torch.manual_seed(100+seed)
    rng=np.random.default_rng(20000+seed)
    net=Network(width,human)
    if optimizer=="Adam":
        opt=torch.optim.Adam(net.parameters(), lr=lr)
    else:
        opt=torch.optim.SGD(net.parameters(), lr=lr,momentum=.9)
    alignment=[]
    for _ in range(60):
        if human:
            xt,yt,_=data["train"]
            ids=rng.integers(len(yt),size=48)
            x,y=xt[ids],yt[ids]
            out, predictions=net(x)
            auxiliary=F.mse_loss(predictions[:,:-1],x[:,1:])
        else:
            delay=int(rng.choice(ref.TRAIN_DELAYS))
            x,y,tex=ref.batch(rng,24,delay)
            rng.permutation(len(y))
            out,predictions=net(x)
            auxiliary=F.cross_entropy(predictions[:,1:].reshape(-1,4),tex[:,1:].reshape(-1))
        main=F.cross_entropy(out,y)
        c=cosine(main,auxiliary,tuple(net.encoder.parameters()))
        alignment.append(c)
        use=(mode=="texture" or mode=="gate" and c>0)
        loss=main+.5*auxiliary if use else main
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
    with torch.no_grad():
        if human:
            x,y,_=data["test"]
            predicted=torch.cat([net(xb)[0].argmax(1) for xb in x.split(256)])
        else:
            gen=np.random.default_rng(54000+100*seed+16)
            x,y,_=ref.batch(gen,512,16)
            predicted=net(x)[0].argmax(1)
            x[:,0,4:]=0
            assert int((net(x)[0].argmax(1)==y).sum())==512
        return int((predicted==y).sum()),len(y),alignment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("task",choices=["corridor","har"])
    parser.add_argument("--har-archive",type=Path, default=ROOT/"har.zip")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.task=="har":
        data=load_har(args.har_archive)
        configs=[(12,.003,"Adam")]
        seeds=range(3000,3008)
    else:
        data=None
        configs=itertools.product((6,12,24),(.001,.003,.01),("Adam","SGD"))
        seeds=range(2000,2012)
    with (ROOT/f"{args.task}_runs.csv").open("w",newline="") as out, (ROOT/f"{args.task}_gradients.csv").open("w",newline="") as trace:
        results=csv.writer(out)
        gradients=csv.writer(trace)
        keys=["width","lr","optimizer","seed","mode"]
        results.writerow(keys+["correct","trials","mean_cosine","negative_fraction"])
        gradients.writerow(keys+["update","cosine"])
        for w,lr,opt in configs:
            for seed,mode in itertools.product(seeds,MODES):
                k=[w,lr,opt,seed,mode]
                correct,trials,cos=train(seed,mode,w,lr,opt,data)
                results.writerow(k+[correct,trials,np.mean(cos),np.mean(np.array(cos)<0)])
                gradients.writerows(k+[i,c] for i,c in enumerate(cos,1))
            out.flush()
            trace.flush()


if __name__=="__main__":
    main()
