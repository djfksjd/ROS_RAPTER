"""Charts of evidence 167: manual overload table vs thermal model (v1 estimate, v2 fit), and 10-minute hip temperature."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
E = 'docs/evidence/167-robstride-public-data-fit'
F = json.load(open(f'{E}/eval/thermal_fit.json'))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, m in zip(ax[:2], ('RS03', 'RS06')):
    rows = [(c, f) for c, f in zip(F[m]['current']['rows'], F[m]['fitted']['rows']) if c['kind'] == 'rotate' and c['manual_s']]
    x = [c['load'] for c, _ in rows]
    a.semilogy(x, [c['manual_s'] for c, _ in rows], 'ks-', label='manual (vendor test)')
    a.semilogy(x, [c['model_s'] for c, _ in rows], 'o--', color='#7f8c8d', label='v1 estimate')
    a.semilogy(x, [f['model_s'] for _, f in rows], 'o-', color='#c0392b', label='v2 fit')
    a.set_xlabel('constant load (N·m, 100 rpm, 25 C)'); a.set_ylabel('time to 145 C winding (s)')
    a.set_title(f'{m}: overload time'); a.legend(); a.grid(alpha=.3, which='both')
for l in open(f'{E}/eval/long_run.jsonl'):
    r = json.loads(l)
    ax[2].plot([p['t']/60 for p in r['trace']], [p['t_winding'][1] for p in r['trace']], '-' if r['mount'] == 'plate' else '--',
               color='#c0392b' if r['cmd'] == 7 else '#2980b9', label=f"cmd {r['cmd']:g} m/s, {r['mount']} mount")
ax[2].axhline(125, color='#e67e22', lw=.8); ax[2].axhline(145, color='k', lw=.8)
ax[2].text(.1, 126, 'derate start', fontsize=8); ax[2].text(.1, 146, 'cut-off', fontsize=8)
ax[2].set_xlabel('minutes of continuous running'); ax[2].set_ylabel('left hip pitch RS03 winding (C)')
ax[2].set_title('rev2_rz1, realism v2, randomize off'); ax[2].legend(fontsize=8, loc='lower right'); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{E}/media/thermal_fit.png', dpi=110)
