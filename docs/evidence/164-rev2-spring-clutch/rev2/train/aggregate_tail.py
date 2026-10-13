"""Aggregate the tail-heat investigation (tail_probe.json, tail_dynamics.json, td340_tailgear*.json) and draw its chart.
Run from the project root."""
import json, shutil, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
T = Path(__file__).resolve().parent; A = T.parents[2]/'assets'
P = json.loads((T/'tail_probe.json').read_text()); Dn = json.loads((T/'tail_dynamics.json').read_text()); GH = json.loads((T/'td340_tailgear_heat.json').read_text())
G = [json.loads(l) for l in (T/'td340_tailgear.jsonl').read_text().splitlines()]
key = lambda r: f"{r['run']}@{r['cmd']:g}"+('@locked' if r['locked'] else '')
gear = {}
for lab in ('as_trained', 'tail_1p5', 'tail_2p0'):
    for c in (4., 5.5, 7.):
        g = [r for r in G if r['label'] == lab and r['cmd'] == c]
        gear[f'{lab}@{c:g}'] = dict(n=len(g), falls=sum(r['fell'] for r in g), speed=float(np.mean([r['speed_tail'] for r in g])), heat_knee=float(np.mean([r['heat_knee'] for r in g])),
                                    heat_max=float(np.mean([r['heat_max'] for r in g])), hottest=collections.Counter(r['heat_argmax'].replace('_joint', '') for r in g).most_common(1)[0][0])
res = dict(probe={key(r): r for r in P}, dynamics={key(r): r for r in Dn}, gear_zero_shot=gear, gear_heat={f"{r['gear']:g}@{r['cmd']:g}": r for r in GH},
           rollouts=len(P)+len(Dn)+len(G)+len(GH))
(T/'tail_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
FONT = '/Library/Fonts/Arial Unicode.ttf'; font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(), 'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(13, 4.4), dpi=150); col = {'rev2_a8_h2': '#81939F', 'rev2_n10': '#BC6333', 'rev2_td340': '#147F86'}
for run in col:
    ax[0].plot([2, 4, 5.5, 7], [res['probe'][f'{run}@{c:g}']['tail']['heat'] for c in (2, 4, 5.5, 7)], 'o-', color=col[run], label=run)
ax[0].plot([2, 7], [res['probe']['rev2_td340@2']['tail']['heat_dc']/.81]*2, ':', color='k', lw=1, label='중력 지지분')
ax[0].axhline(1., color='#B22', ls='--', lw=1); ax[0].set(xlabel='명령 속도 (m/s)', ylabel='꼬리 pitch 부하제곱 (정격 = 1)', title='꼬리 pitch 발열'); ax[0].legend(fontsize=8)
names = [('own_swing', '꼬리 자체 흔들림'), ('trunk_rotation', '몸통 회전'), ('trunk_translation', '몸통 병진')]; x = np.arange(3)
for i, run in enumerate(col):
    d = res['dynamics'][f'{run}@7']; ax[1].bar(x+(i-1)*.27, [d['std'][k] for k, _ in names], .26, color=col[run])
    ax[1].bar(3+(i-1)*.27, d['tau_std'], .26, color=col[run], edgecolor='k', lw=1)
ax[1].set(xticks=np.arange(4), xticklabels=[n for _, n in names]+['모터 토크(합)'], ylabel='토크 표준편차 (Nm)', title='7 m/s: 꼬리 pitch 토크의 성분')
gs = (1., 1.5, 2.)
for c, ls in ((5.5, 'o--'), (7., 'o-')):
    ax[2].plot(gs, [res['gear_heat'][f'{g:g}@{c:g}']['tail_pitch'] for g in gs], ls, color='#147F86', label=f'꼬리 pitch, {c:g} m/s')
    ax[2].plot(gs, [res['gear_heat'][f'{g:g}@{c:g}']['knee'] for g in gs], ls, color='#132C42', alpha=.5, label=f'무릎, {c:g} m/s')
ax[2].axhline(1., color='#B22', ls='--', lw=1); ax[2].set(xticks=gs, xlabel='꼬리 pitch 감속비', ylabel='부하제곱 (정격 = 1)', title='rev2_td340에 감속만 적용 (재학습 없음)'); ax[2].legend(fontsize=8)
for a_ in ax: a_.grid(alpha=.2)
fig.text(.01, .01, '결정적 rollout, 시드 96001, 30초 중 뒤 25초. MuJoCo 시뮬레이션이며 실물 성능이 아님.', fontsize=8); fig.tight_layout(rect=[0, .04, 1, 1])
fig.savefig(T/'media'/'tail_heat.png'); plt.close(fig); shutil.copyfile(T/'media'/'tail_heat.png', A/'rev2-tail-heat.png')
print(json.dumps(gear, indent=0)[:900]); print(res['rollouts'])
