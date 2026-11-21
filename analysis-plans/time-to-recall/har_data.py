"""Prepare subject-disjoint signals directly from the official UCI archive."""
from pathlib import Path
import hashlib
import io
import zipfile
import urllib.request
import numpy as np

URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
SHA = 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031'

def read_archive(path):
    path=Path(path)
    if not path.exists():
        urllib.request.urlretrieve(URL,path)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA:
        raise ValueError('HAR archive digest mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        payload=outer.read('UCI HAR Dataset.zip')
    arrays={}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for split in ('train','test'):
            pre='UCI HAR Dataset/'+split+'/'
            signals=[]
            for group in ('total_acc','body_acc','body_gyro'):
                for axis in 'xyz':
                    name=pre+f'Inertial Signals/{group}_{axis}_{split}.txt'
                    with z.open(name) as stream:
                        a=np.loadtxt(stream,dtype=np.float32)
                    signals.append(a.reshape(-1,32,4).mean(axis=-1))
            arrays[split]=np.stack(signals,axis=2)
            arrays[split+'_y']=np.loadtxt(z.open(pre+f'y_{split}.txt'),dtype=int)-1
            arrays[split+'_person']=np.loadtxt(z.open(pre+f'subject_{split}.txt'),dtype=int)
    assert set(arrays['train_person']).isdisjoint(arrays['test_person'])
    assert len(set(arrays['train_person']))==21 and len(set(arrays['test_person']))==9
    return arrays

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('archive');p.add_argument('cache')
    a=p.parse_args();np.savez(a.cache,**read_archive(a.archive))
