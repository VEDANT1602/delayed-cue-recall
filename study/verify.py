"""Check trial counts, counterbalancing and RMST by independent survival integration."""
from collections import defaultdict
import csv,itertools
from pathlib import Path
from math import isclose
ROOT=Path(__file__).resolve().parent
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def test_numeric(c,p):
    if not isclose(c,p,abs_tol=1e-9):raise AssertionError((c,p))
def main():
    files=sorted(ROOT.glob('curve_*csv'))
    rows=[r for f in files for r in read(f)]
    assert len(rows)==224*3*20
    assert len({tuple(r[k] for k in ('task','cell','seed','mode','delay','update')) for r in rows})==len(rows)
    curves=defaultdict(list)
    for r in rows:
        assert 0<=int(r['correct'])<=int(r['n'])==512
        if r['update']=='200':assert r['erased_correct']=='256'
        curves[tuple(r[k] for k in ('task','cell','seed','mode','delay'))].append(r)
    summary=read(ROOT/'time_summary.csv')
    for s in summary:
        values={}
        keys=[k for k in curves if s['task']==k[0] and s['cell']==k[1] and s['delay']==k[4]]
        for mode in ('cue','auxiliary'):
            events=[]
            for key in (k for k in keys if k[3]==mode):
                r=sorted(curves[key],key=lambda x:int(x['update']))
                assert [int(a['update']) for a in r]==list(range(10,201,10))
                pair=next((int(b['update']) for a,b in zip(r,r[1:]) if all(int(q['correct'])>=float(s['threshold'])*int(q['n']) for q in (a,b))),None)
                events.append(pair)
            # Count area under the observed event-free curve up to 200.
            assert len(events)==int(s['n_pairs'])
            area=sum(10*sum(t is None or t>u for t in events)/len(events) for u in range(0,200,10))
            test_numeric(area,float(s['cue_rmst' if mode=='cue' else 'aux_rmst']))
            values[mode]=area
        test_numeric(values['auxiliary']-values['cue'],float(s['difference']))
    assert len([r for r in read(ROOT/'measured_label_events.csv')])==24
    assert sum(len(read(f)) for f in ROOT.glob('activity_memory_*.csv'))==720
    assert sum(len(read(f)) for f in ROOT.glob('activity_curve_*.csv'))==280
    print(f'{len(rows)} trajectory records; {len(summary)} survival contrasts; 24 measured-label and 8 natural activity fits verified.')
if __name__=='__main__':main()
