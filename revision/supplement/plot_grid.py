"""Draw the early accuracy differences and shared-encoder gradient trace."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

here=Path(__file__).resolve().parent
rows=list(csv.DictReader((here/"corridor_summary.csv").open()))
fig,ax=plt.subplots(1,2,figsize=(8,3.55),constrained_layout=True)
for grid,optimizer in zip(ax,("Adam","SGD")):
    contrasts={}
    for row in rows:
        if row['optimizer']==optimizer and row['mode']=='texture' and row['control']=='choice':
            contrasts[(int(row['width']),float(row['lr']))]=float(row['difference_pp'])
    values=np.array([[contrasts[w,r] for r in (0.001,.003,.01)] for w in (6,12,24)])
    grid.imshow(values,cmap='PuOr',norm=TwoSlopeNorm(vcenter=0,vmin=-27,vmax=27),aspect='auto')
    for (i,j),v in np.ndenumerate(values):
        grid.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>15 else '#171717',fontsize=11)
    grid.set_title(optimizer if optimizer=='Adam' else 'SGD, momentum 0.9')
    grid.set_xticks(range(3),['0.001','0.003','0.010'])
    grid.set_yticks(range(3),['6','12','24'])
    grid.set_ylabel('Recurrent width')
    grid.set_xlabel('Learning rate')
fig.savefig(here/'grid.png',dpi=190)
