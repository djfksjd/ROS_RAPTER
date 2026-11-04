"""One plain chart, exclusively from the fixed diagnostic episode summaries."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/raptor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E=Path(__file__).resolve().parent


def main():
    data=json.loads((E/'eval/j1_diagnosis/summary.json').read_text())
    fig,ax=plt.subplots(figsize=(7.5,4.1))
    groups=[('l80_s1p',False),('l80_s1p',True),('l80_j1',False),('l80_j1',True)]
    for x,(name,stochastic) in enumerate(groups):
        rows=[r for r in data['episodes'] if r['policy']==name and r['stochastic']==stochastic]
        for j,r in enumerate(rows):
            ax.scatter(x+(j-1.5)*.11,1000*r['max_flight_s'],color='#2563a6',s=40,zorder=3)
        lo=min(r['max_flight_s'] for r in rows)*1000;hi=max(r['max_flight_s'] for r in rows)*1000
        ax.text(x,hi+5,f'{lo:.0f}–{hi:.0f} ms',ha='center',fontsize=10)
    ax.axhline(100,color='#aa3333',linestyle='--',linewidth=1,label='5a flight threshold (100 ms)')
    ax.set_xticks(range(4),['s1p\nmean action','s1p\nsampled action','j1 (3M)\nmean action','j1 (3M)\nsampled action'])
    ax.set_ylabel('Longest continuous both-feet flight (ms)');ax.set_ylim(-5,120)
    ax.set_title('Training-distribution probes: 4 seeds per group, 12 s each')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();(E/'media').mkdir(exist_ok=True)
    fig.savefig(E/'media/j1_flight_diagnosis.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
