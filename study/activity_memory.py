"""Retain a verified activity label across a counterbalanced measured distractor segment."""
import argparse,csv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from activity import ROOT,load
class MemoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.GRU(10,48,batch_first=True)
        self.decision=nn.Linear(48,6);self.aux=nn.Linear(48,9)
    def forward(self,x):
        h,_=self.enc(x)
        return self.decision(h[:,-1]),self.aux(h[:,16:-1])

class Sampler:
    def __init__(self,fold):
        self.x,self.y,self.ids=fold
        self.classes=[np.flatnonzero(self.y.numpy()==k) for k in range(6)]
    def batch(self,rng,pairs):
        # One measured suffix per group, reused across all six possible activity labels.
        prefix_id=np.concatenate([rng.choice(c,pairs) for c in self.classes])
        suffix_id=rng.integers(len(self.x),size=pairs)
        prefix_start=rng.integers(17,size=len(prefix_id));suffix_start=rng.integers(17,size=pairs)
        pre=self.x[prefix_id[:,None],prefix_start[:,None]+np.arange(16)[None,:]]
        post=self.x[suffix_id[:,None],suffix_start[:,None]+np.arange(16)[None,:]].repeat(6,1,1)
        v=torch.zeros(len(pre),32,10);v[:,:16,:9]=pre;v[:,16:,:9]=post;v[:,:16,9]=1.
        y=self.y[prefix_id];assert torch.equal(y,torch.arange(6).repeat_interleave(pairs))
        return v,y
def main():
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('--cache',default='har.npz');p.add_argument('--out',type=type(ROOT),default=ROOT)
    args=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folds=load(args.cache)
    tr,va,te=[Sampler(folds[k]) for k in ('fit','valid','test')]
    xv,yv=va.batch(np.random.default_rng(97111),128);xt,yt=te.batch(np.random.default_rng(97313),128)
    with (args.out/f'activity_memory_{args.seed}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','mode','update','val_correct','val_n','test_correct','test_n','erased_correct'])
        for mode in ('activity','auxiliary'):
            torch.manual_seed(args.seed);net=MemoryNet();opt=torch.optim.Adam(net.parameters(),lr=.002)
            rng=np.random.default_rng(12000+args.seed)
            for t in range(1,1501):
                x,y=tr.batch(rng,6);dec,aux=net(x)
                loss=F.cross_entropy(dec,y)
                if mode=='auxiliary':loss=loss+.5*F.mse_loss(aux,x[:,17:,:9])
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                if t%50:continue
                with torch.no_grad():
                    v=int((net(xv)[0].argmax(1)==yv).sum())
                    test=erased=n=''
                    if t==1500:
                        test=int((net(xt)[0].argmax(1)==yt).sum());n=len(yt)
                        masked=xt.clone();masked[:,:16,:9]=0.
                        erased=int((net(masked)[0].argmax(1)==yt).sum());assert erased==n//6
                    w.writerow([args.seed,mode,t,v,len(yv),test,n,erased]);f.flush()
if __name__=='__main__':main()
