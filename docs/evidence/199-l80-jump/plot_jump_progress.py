"""j1/j2/j3 comparison: fixed quick seeds and full held-out suites kept separate."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/raptor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_training_j1 import metrics
E=Path(__file__).resolve().parent


def main():
    quick=[]
    for name,block in [('l80_j1',3),('l80_j2',4),('l80_j3',2)]:
        rows=[json.loads(l) for l in (E/f'eval/{name}/quick_{block}M.jsonl').read_text().splitlines()]
        quick.append(metrics(rows)['passed'])
    full=[None]
    for name in ('j2','j3'):
        a=json.loads((E/f'eval/{name}_full/analysis.json').read_text())
        full.append(sum(e['passed'] for e in a['episodes']))
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    colors=['#9a9a9a','#6688ad','#346f9e']
    for ax,values,total,title in zip(axes,[quick,full],[6,32],['Fixed quick set (6 episodes)','Full 5a set (8 nominal + 24 DR)']):
        for x,value in enumerate(values):
            if value is None:
                ax.text(x,4,'Not evaluated',ha='center',va='bottom',fontsize=9,rotation=90)
                continue
            ax.bar(x,value,color=colors[x],width=.6)
            ax.text(x,value+.025*total,f'{value}/{total}',ha='center',fontsize=10)
        ax.set_xticks([0,1,2],['j1 (3M)','j2 (4M)','j3 (2M)']);ax.set_xlim(-.6,2.6)
        ax.set_ylim(0,total*1.15);ax.set_ylabel('Complete jumps');ax.set_title(title,fontsize=10)
        ax.set_yticks(range(0,total+1,2 if total==6 else 8));ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Final checkpoints | all three remain unadopted',fontsize=12)
    fig.tight_layout();fig.savefig(E/'media/j1_j2_j3_jump.png',dpi=160);plt.close(fig)
    (E/'eval/j3_full/chart_data.json').write_text(json.dumps(dict(quick_successes=quick,quick_n=6,
        full_successes=full,full_n=32,j1_full='not evaluated; never zero-filled'),indent=2)+'\n')


if __name__=='__main__':main()
