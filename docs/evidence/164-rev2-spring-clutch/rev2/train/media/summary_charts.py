"""Summary charts of the model_rev 2 runs from stored result files (no simulation is run).
Usage (project root): .venv-sim/bin/python summary_charts.py"""
import json, collections, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
OUT = Path(__file__).resolve().parent; T = OUT.parent
FONT = '/Library/Fonts/Arial Unicode.ttf'; font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(), 'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
RUNS = [('rev2_a8_h2', 'accept.jsonl', '걸쇠\n299회'), ('rev2_n10', 'n10_accept.jsonl', '명목각\n299회'), ('rev2_td340', 'td340_accept.jsonl', '접지각\n299회'),
        ('rev2_r2', 'r2_spring_off_8.jsonl', '래칫 r2\n8회'), ('rev2_r3_ck7M', 'r3_ck7M_spring_off_8.jsonl', '대칭 r3\n8회'), ('rev2_r4b', 'r4b_spring_off_8.jsonl', '롤 r4b\n8회')]
CMDS = (2., 4., 5.5, 7.)


def table(f):
    g = collections.defaultdict(list)
    for l in open(T/f):
        r = json.loads(l)
        if r.get('label', 'as_trained') in ('as_trained', 'baseline', 'accept') and r.get('mode', 'policy') == 'policy': g[r['cmd']].append(r)
    out = {}
    for c in CMDS:
        v = g[c]; ok = [x for x in v if not x['fell']] or v
        out[c] = dict(n=len(v), falls=sum(x['fell'] for x in v), speed=st.mean(x['speed_tail'] for x in ok), heat=st.mean(x['heat_max'] for x in ok), knee=st.mean(x['heat_knee'] for x in ok),
                      J=st.mean(x['spring_peak_J'] for x in ok) if 'spring_peak_J' in ok[0] else None)
    return out


D = {run: table(f) for run, f, _ in RUNS}; json.dump(D, open(OUT/'rev2_summary.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2)); x = range(len(RUNS)); col = {2.: '#9DB7C4', 4.: '#5E8CA3', 5.5: '#2F6480', 7.: '#123B52'}
for i, c in enumerate(CMDS):
    off = [k+(i-1.5)*.2 for k in x]
    ax[0].bar(off, [D[r][c]['speed'] for r, _, _ in RUNS], .19, color=col[c], label=f'명령 {c:g} m/s')
    ax[1].bar(off, [D[r][c]['heat'] for r, _, _ in RUNS], .19, color=col[c])
    ax[2].bar(off, [D[r][c]['J'] or 0 for r, _, _ in RUNS], .19, color=col[c])
ax[1].axhline(1., color='#B5422C', lw=1.2); ax[1].text(len(RUNS)-.6, 1.03, '연속 정격', color='#B5422C', ha='right')
ax[2].axhline(2., color='#B5422C', lw=1.2, ls='--'); ax[2].text(len(RUNS)-.6, 2.15, '기준 2 J', color='#B5422C', ha='right')
for a, t in zip(ax, ('60초 뒤 속도 (m/s)', '가장 뜨거운 모터의 부하제곱 (정격 = 1)', '걸음당 무릎 스프링 저장 (J, 앞 세 정책은 기록 없음)')):
    a.set_title(t); a.set_xticks(list(x)); a.set_xticklabels([l for _, _, l in RUNS], fontsize=9)
ax[0].set_ylim(0, 7.6); ax[0].legend(frameon=False, fontsize=8, ncol=4, loc='upper center', columnspacing=1., handlelength=1.2); fig.suptitle('model_rev 2 정책 비교 · 고정 초기 시드 96001–, 결정적 60초 · 넘어진 시행은 평균에서 제외', fontsize=11)
fig.tight_layout(); fig.savefig(OUT/'rev2_summary.png', dpi=140)

R = [('rev2_r3_ck7M', '대칭 r3'), ('rev2_r4b_ck4M', 'r4b 400만'), ('rev2_r4b', 'r4b 최종')]; sizes = (1.25, 1.5, 1.75, 2.0); fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True); roll = {}
for a, c in zip(ax, (4., 5.5, 7.)):
    for (run, lab), cl in zip(R, ('#9DB7C4', '#5E8CA3', '#123B52')):
        rows = json.load(open(T/f'roll_fine_{run}.json')); y = [sum(r['fell'] for r in rows if r['size'] == s and r['cmd'] == c) for s in sizes]; roll.setdefault(run, {})[c] = y
        a.plot(sizes, y, 'o-', color=cl, label=lab)
    a.set_title(f'{c:g} m/s 주행 중'); a.set_xlabel('롤 충격 (N·m·s)'); a.set_xticks(sizes)
ax[0].set_ylabel('32회 중 넘어진 횟수'); ax[0].legend(frameon=False)
fig.suptitle('몸통 롤 충격 1회 뒤 5초 안에 넘어진 횟수 · 공중 구간 롤 관성 0.187 kg·m² (2.0 N·m·s → 10.7 rad/s)', fontsize=10.5); fig.tight_layout(); fig.savefig(OUT/'roll_impulse.png', dpi=140)
json.dump(roll, open(OUT/'roll_impulse.json', 'w'), indent=1); print(json.dumps({r: {c: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in D[r][c].items()} for c in CMDS} for r in D}, ensure_ascii=False))
