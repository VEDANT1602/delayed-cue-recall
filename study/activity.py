"""Subject-disjoint activity classification with model selection on validation subjects alone."""
import argparse,copy,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent

class ActivityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.GRU(9,48,batch_first=True)
        self.readout=nn.Linear(48,6)
        self.forecast=nn.Linear(48,9)
    def forward(self,x):
        h,_=self.enc(x)
        return self.readout(h.mean(1)),self.forecast(h[:,:-1])

def summary_features(a):
    return np.concatenate([a.mean(1),a.std(1),a.min(1),a.max(1),
                           np.abs(np.diff(a,axis=1)).mean(1),
                           np.abs(np.fft.rfft(a,axis=1))[:,1:5].mean(1),
                           np.abs(np.fft.rfft(a,axis=1))[:,5:12].mean(1)],axis=1)

def load(cache):
    a=np.load(cache)
    people=np.unique(a['train_person'])
    val=people[[3,8,13,18]]
    mask=np.isin(a['train_person'],val)
    assert not set(a['test_person']).intersection(a['train_person'])
    folds={}
    for name,subset in [('fit',~mask),('valid',mask)]:
        folds[name]=(a['train'][subset],a['train_y'][subset],a['train_person'][subset])
    folds['test']=(a['test'],a['test_y'],a['test_person'])
    mean=folds['fit'][0].mean((0,1))
    sd=folds['fit'][0].std((0,1))
    return {name:(torch.from_numpy((x-mean)/sd),torch.from_numpy(y),person)
            for name,(x,y,person) in folds.items()}

@torch.no_grad()
def predict(net,x):
    return torch.cat([net(b)[0].argmax(1) for b in x.split(256)]).numpy()

def stats(y,p):
    return accuracy_score(y,p),f1_score(y,p,average='macro',labels=np.arange(6),zero_division=0)

def write_scores(wr,mode,seed,epoch,fold,preds):
    _,y,groups=fold
    for person in [-1]+list(np.unique(groups)):
        m=np.ones(len(y),bool) if person==-1 else groups==person
        acc,f1=stats(y[m],preds[m])
        wr.writerow([mode,seed,epoch,person,len(y[m]),acc,f1])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('seed',type=int)
    parser.add_argument('--cache',default='har.npz')
    parser.add_argument('--out',type=Path,default=ROOT)
    a=parser.parse_args();seed=a.seed
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folds=load(a.cache);x,y,_=folds['fit'];xv,yv,_=folds['valid']
    with (a.out/f'activity_curve_{seed}.csv').open('w',newline='') as traces,(a.out/f'activity_test_{seed}.csv').open('w',newline='') as outcomes:
        vals,scores=csv.writer(traces),csv.writer(outcomes)
        vals.writerow(['seed','mode','epoch','n','val_accuracy','val_macro_f1'])
        scores.writerow(['mode','seed','selected_epoch','subject','test_windows','accuracy','macro_f1'])
        if seed==7600:
            linear=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
            linear.fit(summary_features(x.numpy()),y.numpy())
            p=linear.predict(summary_features(folds['test'][0].numpy()))
            write_scores(scores,'window_logistic',seed,0,folds['test'],p);outcomes.flush()
        for mode in ('activity','auxiliary'):
            torch.manual_seed(seed)
            net=ActivityNet();opt=torch.optim.Adam(net.parameters(),lr=.002)
            rng=np.random.default_rng(seed+10000)
            best_score=-1
            for epoch in range(1,36):
                for ids in np.array_split(rng.permutation(len(y)),np.ceil(len(y)/128).astype(int)):
                    xb=x[ids];out,forecast=net(xb)
                    loss=F.cross_entropy(out,y[ids])
                    if mode=='auxiliary':loss=loss+.5*F.mse_loss(forecast,xb[:,1:])
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                p=predict(net,xv);acc,f1=stats(yv,p)
                vals.writerow([seed,mode,epoch,len(yv),acc,f1]);traces.flush()
                if f1>best_score:
                    best,best_score,selected=copy.deepcopy(net.state_dict()),f1,epoch
            net.load_state_dict(best)
            p=predict(net,folds['test'][0])
            write_scores(scores,mode,seed,selected,folds['test'],p);outcomes.flush()

if __name__=='__main__': main()
