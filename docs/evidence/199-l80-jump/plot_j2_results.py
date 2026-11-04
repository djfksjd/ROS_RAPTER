"""Plain counts from full 5a episode data, not training rewards."""
import collections
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/raptor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parent


def main():
    data=json.loads((E/'eval/j2_full/analysis.json').read_text())
    fig,ax=plt.subplots(figsize=(8,3.5))
    colors={'pass':'#387a5a','apex':'#c7534a','hold':'#d99b40'}
    labels={'pass':'Complete jump','apex':'Apex < 5 cm','hold':'Hold < 1 s'}
    for y,cond in enumerate(('in_place','in_place_dr')):
        counts=collections.Counter(e['primary_failure'] for e in data['episodes'] if e['cond']==cond);left=0
        for key in ('pass','apex','hold'):
            value=counts[key]
            ax.barh(y,value,left=left,color=colors[key],height=.5,label=labels[key] if y==0 else None)
            if value:ax.text(left+value/2,y,str(value),ha='center',va='center',color='white',fontweight='bold')
            left+=value
    ax.set_yticks([0,1],['Nominal (needs 8/8)','Randomized (needs ≥22/24)']);ax.invert_yaxis()
    ax.set_xlim(0,25);ax.set_xticks(range(0,25,4));ax.set_xlabel('Episodes')
    ax.set_title('l80_j2 full 5a: 17/32 complete, 0 falls')
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=3,frameon=False)
    fig.tight_layout();fig.savefig(E/'media/j2_full_results.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
