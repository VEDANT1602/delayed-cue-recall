"""Summarise acquisition with administrative censoring and paired seed contrasts."""
import csv,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def write(name,head,rows):
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(head);writer.writerows(rows)
def boot_interval(delta):
    a=np.asarray(delta);r=np.random.default_rng(19373)
    return np.quantile(r.choice(a,(20000,len(a)),replace=True).mean(axis=1),[.025,.975])
def records(names):
    for name in names:
        with name.open() as f:yield from csv.DictReader(f)
def event_time(history,threshold):
    last=False
    for t,correct,n in sorted(history):
        passed=correct/n>=threshold
        if passed and last:return t,1
        last=passed
    return 200,0
def summarize():
    groups={}
    all_files=sorted(ROOT.glob('curve_*.csv'))
    for f in all_files:
        for row in records([f]):
            key=(row['task'],row['cell'],int(row['seed']),row['mode'],int(row['delay']))
            groups.setdefault(key,[]).append(tuple(int(row[k]) for k in ('update','correct','n')))
    times=[];results=[];surv=[]
    for (task,cell), g in itertools.groupby(sorted(groups),key=lambda k:k[:2]):
        keys=list(g);seeds=sorted({k[2] for k in keys})
        for d,thresh in itertools.product((8,16,32),(.80,.90,.95)):
            vals={};statuses={}
            for mode in ('cue','auxiliary'):
                e=[event_time(groups[(task,cell,s,mode,d)],thresh) for s in seeds]
                vals[mode]=np.array([p[0] for p in e])
                flags=np.array([p[1] for p in e]);statuses[mode]=flags
                for s,(t,event) in zip(seeds,e):times.append([task,cell,s,mode,d,thresh,t,event])
                for t in range(0,201,10):
                    survival=1-np.mean((vals[mode]<=t)&(flags==1))
                    surv.append([task,cell,mode,d,thresh,t,survival])
            diff=vals['auxiliary']-vals['cue'];lo,hi=boot_interval(diff)
            results.append([task,cell,d,thresh,len(seeds),statuses['cue'].sum(),statuses['auxiliary'].sum(),
                            vals['cue'].mean(),vals['auxiliary'].mean(),diff.mean(),lo,hi,
                            np.sum(diff>0),np.sum(diff==0),np.sum(diff<0)])
    write('events.csv',['task','cell','seed','mode','delay','threshold','restricted_updates','event'],times)
    write('time_summary.csv',['task','cell','delay','threshold','n_pairs','cue_events','aux_events','cue_rmst','aux_rmst','difference','ci_low','ci_high','slower','tied','faster'],results)
    write('survival.csv',['task','cell','mode','delay','threshold','update','survival'],surv)
if __name__=='__main__':summarize()
