"""Summarize saved evidence and draw one figure; never runs simulation or training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E = Path(__file__).resolve().parent
ROOT = E.parents[2]

def readlines(name): return [json.loads(line) for line in (E/'eval'/name).read_text().splitlines()]

def main():
    search = readlines('search.jsonl'); replay = readlines('replay.jsonl'); terrain = readlines('terrain.jsonl')
    audit = json.loads((E/'eval/search_summary.json').read_text())
    counts = {}
    for mode in ('vertical', 'forward'):
        rows = [r for r in search if r['mode'] == mode]; flights = [r for r in rows if r['result']['event']]
        counts[mode] = dict(candidates=len(rows), flights=len(flights),
            touchdown=sum(r['result']['event']['landed'] for r in flights),
            positive_flight_distance=sum(r['result']['event']['distance_m'] > 0 for r in flights),
            survived=sum(r['result']['t_fall'] is None for r in rows),
            stand_1s=sum(r['result']['stand_after_landing_s'] >= 1. and r['result']['t_fall'] is None for r in rows))
    terrain_summary = []
    for height in (.05, .10, -.05, -.10):
        rows = [r for r in terrain if r['height_m'] == height]
        terrain_summary.append(dict(height_m=height, n=len(rows), reached_6m=sum(r['reached_6m'] for r in rows),
            falls=sum(r['t_fall'] is not None for r in rows),
            impact_bw_range=[min(r['impact_1s_peak_bw'] for r in rows), max(r['impact_1s_peak_bw'] for r in rows)],
            max_current_over_peak=max(r['current_1s_over_peak'] for r in rows)))
    input_check = {}
    for rel,h in audit['input_sha256'].items():
        path = ROOT/rel
        if path.name == 'jump_probe.py': path = E/'eval/search_jump_probe_snapshot.py'
        input_check[rel] = hashlib.sha256(path.read_bytes()).hexdigest() == h
    assert all(input_check.values())
    assert len(search) == 1280 and len(terrain) == 12 and len(replay) == 16
    assert not any(r['result']['stand_after_landing_s'] >= 1. for r in replay)
    summary = dict(search=counts, replay_n=len(replay), replay_falls=sum(r['result']['t_fall'] is not None for r in replay),
                   terrain=terrain_summary, input_hashes_verified=input_check)
    (E/'eval/summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig,axes = plt.subplots(2,2,figsize=(10,6.4))
    for mode, selection, color, label in [('vertical','extreme','#426b94','Vertical search: highest'),
                                         ('forward','score','#b36728','Forward search: highest (backward)')]:
        row = next(r for r in replay if r['mode']==mode and r['selection']==selection and r['seed'] is None)
        a = np.load(E/'eval'/row['trace']); ev = row['result']['event']; origin=min(np.searchsorted(a['t'], .2),len(a['t'])-1)
        axes[0,0].plot(a['t'], (a['xyz'][:,2]-a['xyz'][origin,2])*100, color=color,label=label)
        axes[0,0].axvspan(ev['takeoff_s'],ev['touchdown_s'],alpha=.09,color=color)
        axes[0,1].plot(a['t'], a['xyz'][:,0]-a['xyz'][origin,0],color=color)
        axes[1,0].plot(a['t'],a['feet_force_n'].sum(axis=1)/float(a['weight_n']),color=color)
        for ax in (axes[0,0],axes[0,1],axes[1,0]):
            ax.axvline(row['result']['t_fall'],color=color,ls=':',lw=1)
            ax.set_xlabel('Time (s)');ax.grid(alpha=.18)
    axes[0,0].set(title='Base rise; shaded = selected flight',ylabel='Rise from t=0.20 s (cm)')
    axes[0,0].legend(fontsize=7,loc='lower left')
    axes[0,1].set(title='Both candidates move backward',ylabel='Base x from t=0.20 s (m)')
    axes[1,0].set(title='1 ms foot contact force; dotted = fall',ylabel='Total normal force / body weight')
    ax=axes[1,1]; xs=np.arange(4);ns=[r['reached_6m'] for r in terrain_summary]
    ax.bar(xs,ns,color=['#777777']*3+['#bd6666'],width=.55)
    ax.set_xticks(xs,['+5 cm','+10 cm','-5 cm','-10 cm']); ax.set_ylim(0,3.6);ax.set_yticks([0,1,2,3])
    for x,n in zip(xs,ns):ax.text(x,n+.08,f'{n}/3',ha='center')
    ax.set(title='r1_6M at 4 m/s: reaches x > 6 m',ylabel='Trials without fall');ax.grid(axis='y',alpha=.18)
    fig.suptitle('Frozen l80: no successful jump + 1 s stand found',fontsize=12)
    fig.tight_layout();fig.savefig(E/'media/jump_feasibility.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
