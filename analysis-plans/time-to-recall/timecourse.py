"""Train paired terminal-cue models, retain the complete scheduled assessment series."""
import argparse,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
CELLS={'RNN':nn.RNN,'GRU':nn.GRU,'LSTM':nn.LSTM}
DELAYS=(8,16,32)

class CueNet(nn.Module):
    def __init__(self,cell,channels,width=12):
        super().__init__()
        self.encoder=CELLS[cell](channels+2,width,batch_first=True)
        self.cue=nn.Linear(width,2)
        self.aux=nn.Linear(width,channels)

    def forward(self,x):
        h,_=self.encoder(x)
        return self.cue(h[:,-1]),self.aux(h[:,1:])

def symbolic(rng,delay,pairs):
    initial=rng.integers(0,4,size=(pairs,1))
    turns=np.where(rng.random((pairs,delay))<.8,1,-1)
    path=np.concatenate([initial,(initial+np.cumsum(turns,axis=1))%4],axis=1)
    labels=torch.from_numpy(path)
    return F.one_hot(labels[:,:-1],4).float(),labels[:,1:]

def sample(rng,delay,pairs,source):
    if source is None:
        sequence,aux=symbolic(rng,delay,pairs)
    else:
        # Every selected measured window contributes contiguous observations.
        which=rng.integers(len(source),size=pairs)
        start=rng.integers(33-delay,size=pairs)
        sequence=source[which[:,None],start[:,None]+np.arange(delay)[None,:]]
        aux=sequence[:,1:]
    channels=sequence.shape[-1]
    x=torch.zeros((2*pairs,delay+1,channels+2))
    x[:,1:,:channels]=sequence.repeat(2,1,1)
    y=torch.arange(2).repeat_interleave(pairs)
    x[:,0,channels:]=F.one_hot(y,2).float()
    return x,y,aux.repeat((2,)+(1,)*(aux.ndim-1))

def datasets(cache):
    a=np.load(cache)
    tr=torch.from_numpy(a['train'][:,:,:3]);te=torch.from_numpy(a['test'][:,:,:3])
    mu,sd=tr.mean((0,1)),tr.std((0,1),correction=0)
    return (tr-mu)/sd,(te-mu)/sd

def run_pair(task,cell,seed,sources,writer):
    channels=4 if task=='corridor' else 3
    tr,te=sources
    tests={}
    for delay in DELAYS:
        tests[delay]=sample(np.random.default_rng(2_000_000+41*seed+delay),delay,256,te)[:2]
    for mode in ('cue','auxiliary'):
        torch.manual_seed(seed+100)
        model=CueNet(cell,channels)
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        rng=np.random.default_rng(40000+seed)
        for step in range(1,201):
            delay=int(rng.choice((2,4,8)))
            x,y,z=sample(rng,delay,24,tr)
            main,pred=model(x)
            loss=F.cross_entropy(main,y)
            if mode=='auxiliary':
                if task=='corridor': second=F.cross_entropy(pred.flatten(0,1),z.flatten())
                else: second=F.mse_loss(pred[:,:-1],z)
                loss=loss+.5*second
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%10==0:
                with torch.no_grad():
                    for d,(xt,yt) in tests.items():
                        out=model(xt)[0].argmax(1)
                        hits=int((out==yt).sum())
                        if step==200:
                            zero=xt.clone();zero[:,0,-2:]=0
                            erased=int((model(zero)[0].argmax(1)==yt).sum())
                            assert erased==len(yt)//2
                        else: erased=''
                        writer.writerow([task,cell,seed,mode,d,step,hits,len(yt),erased])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['corridor','measured'])
    p.add_argument('cell',choices=CELLS)
    p.add_argument('--cache',default='har.npz')
    p.add_argument('--out',type=Path,default=ROOT)
    a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    seeds=range(5100,5120) if a.task=='corridor' else range(6500,6512)
    source=(None,None) if a.task=='corridor' else datasets(a.cache)
    with (a.out/f'curve_{a.task}_{a.cell}.csv').open('w',newline='') as out:
        wr=csv.writer(out)
        wr.writerow(['task','cell','seed','mode','delay','update','correct','n','erased_correct'])
        for seed in seeds:
            run_pair(a.task,a.cell,seed,source,wr);out.flush()

if __name__=='__main__': main()
