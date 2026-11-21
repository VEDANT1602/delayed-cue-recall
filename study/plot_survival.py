from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def lines(path, mode, pattern):
    rows=[r for r in csv.DictReader(path.open()) if r['mode']==mode and pattern(r)]
    return [float(r['update']) for r in rows],[100*float(r['survival']) for r in rows]

if __name__=='__main__':
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.2),sharey=True,layout='constrained')
    setups=[('Confirmed GRU · terminal cue','survival.csv',('cue','auxiliary'),lambda r:r['task']=='confirmation' and r['threshold']=='0.9' and r['delay']=='16',200,'90% cue accuracy'),('GRU · recorded activity label','measured_label_survival.csv',('activity','auxiliary'),lambda r:True,1500,'75% activity accuracy')]
    for ax,(title,path,modes,keep,horizon,target) in zip(axes,setups):
        for mode,color,label,style in zip(modes,['#14635F','#B84928'],['Decision only','Decision + next observation'],['-','--']):
            x,y=lines(ROOT/path,mode,keep);ax.step(x,y,where='post',lw=2.2,ls=style,c=color,label=label)
        ax.set(xlim=(0,horizon),ylim=(-3,103),xlabel='Optimizer updates',title=title)
        ax.grid(axis='y',color='#DDDDDD',lw=.65);ax.spines[['top','right']].set_visible(False)
        ax.text(.98,.97,'Target: '+target,transform=ax.transAxes,ha='right',va='top',fontsize=8,color='#444444')
    axes[0].set_ylabel('Runs yet to reach criterion (%)')
    axes[1].legend(loc='center right',frameon=False,fontsize=7)
    fig.savefig(ROOT/'acquisition.png',dpi=200,bbox_inches='tight');plt.close(fig)
