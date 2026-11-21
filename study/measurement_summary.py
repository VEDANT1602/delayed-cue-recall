"""Summarize independently labeled activity experiments, including every fit."""
import csv,glob
from pathlib import Path
import numpy as np
from aggregate import ROOT,boot_interval,write,records

def main():
    natural=list(records(sorted(ROOT.glob('activity_test_*.csv'))))
    summary=[]
    for mode in ('window_logistic','activity','auxiliary'):
        r=[x for x in natural if x['mode']==mode and x['subject']=='-1']
        summary.append([mode,len(r),np.mean([float(x['accuracy']) for x in r]),np.mean([float(x['macro_f1']) for x in r]),min(int(x['selected_epoch']) for x in r),max(int(x['selected_epoch']) for x in r)])
    write('activity_summary.csv',['mode','fits','test_accuracy','test_macro_f1','earliest_selection','latest_selection'],summary)
    allrows=list(records(sorted(ROOT.glob('activity_memory_*.csv'))))
    seeds=sorted({int(r['seed']) for r in allrows})
    if len(seeds)!=12:raise ValueError('Expected all 12 measured-label seed pairs')
    rows=[];surv=[]
    from collections import defaultdict
    data=defaultdict(list);areas={};tests={};events={}
    for r in allrows:data[(int(r['seed']),r['mode'])].append(r)
    for mode in ('activity','auxiliary'):
        ls=[];terminal=[];flags=[]
        for seed in seeds:
            r=sorted(data[seed,mode],key=lambda x:int(x['update']))
            assert len(r)==30 and int(r[-1]['update'])==1500 and int(r[-1]['erased_correct'])==128
            good=np.array([int(o['val_correct'])/int(o['val_n'])>=.75 for o in r])
            matches=np.flatnonzero(good[:-1]&good[1:])+1
            event=int(len(matches)>0);time=int(r[matches[0]]['update']) if event else 1500
            test=float(r[-1]['test_correct'])/int(r[-1]['test_n'])
            rows.append([seed,mode,time,event,test]);ls.append(time);flags.append(event);terminal.append(test)
        vals=np.array(ls);flags=np.array(flags)
        areas[mode]=vals;tests[mode]=np.array(terminal);events[mode]=sum(flags)
        for update in range(0,1501,50):
            s=1-np.mean((vals<=update)&(flags==1))
            surv.append([mode,update,s])
    write('measured_label_events.csv',['seed','mode','restricted_updates','event','test_accuracy'],rows)
    write('measured_label_survival.csv',['mode','update','survival'],surv)
    report=[]
    for name,values in [('restricted_updates',areas),('final_accuracy',tests)]:
        delta=values['auxiliary']-values['activity'];lo,hi=boot_interval(delta)
        report.append([name,len(seeds),events['activity'],events['auxiliary'],values['activity'].mean(),values['auxiliary'].mean(),delta.mean(),lo,hi,sum(delta>0),sum(delta==0),sum(delta<0)])
    write('measured_label_summary.csv',['endpoint','pairs','activity_events','auxiliary_events','activity_mean','auxiliary_mean','paired_difference','ci_low','ci_high','positive','tied','negative'],report)
if __name__=='__main__':main()
