"""Charts for the T1 running evidence (docs/evidence/86-t1-run/eval_*.json -> docs/assets/t1/*.png).
Usage: .venv-sim/bin/python sim/rl/plot_t1.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT/'docs/evidence/86-t1-run'
OUT = ROOT/'docs/assets/t1'
plt.rcParams.update({'font.family': ['AppleGothic', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 9})


def rows(name):
    return json.loads((EV/name).read_text())


def speed_chart():
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    series = [('v2 tail active', 'eval_yaw.json', 'active', 'C0', 'o-'), ('v2 tail locked (same policy)', 'eval_yaw.json', 'locked', 'C0', 'o--'),
              ('tail-locked policy', 'eval_locked_policy.json', 'locked', 'C3', 's--'), ('v4 (turn-capable)', 'eval_v4_yaw_turn.json', 'active', 'C2', '^-'), ('v5b alternating gait', 'eval_v5b_alt.json', 'active', 'C1', 'D-'), ('v8d (current best)', 'eval_v8d.json', 'active', 'C4', 'P-')]
    for label, f, tail, c, st in series:
        r = [x for x in rows(f) if 'impulse_nms' not in x and 'turn_cmd_rad_s' not in x and x['tail'] == tail]
        cmd = [x['cmd_vx'] for x in r]; v = [x['mean_vx'] if x['falls'] == 0 else float('nan') for x in r]
        ax.plot(cmd, v, st, color=c, label=label, ms=4)
        for x in r:
            if x['falls']:
                ax.plot(x['cmd_vx'], 0, 'x', color=c, ms=7)
    ax.plot([0, 11.1], [0, 11.1], ':', color='grey', lw=.8, label='command = achieved')
    ax.axvline(11.1, color='k', lw=.6); ax.text(11.15, 1, '40 km/h', fontsize=8)
    ax.set_xlabel('commanded speed (m/s)'); ax.set_ylabel('achieved body speed (m/s), x = fell')
    ax.set_title('T1 virtual actuator, flat, deterministic 3 x 10 s (MuJoCo, not the real robot)', fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/'speed-vs-command.png'); plt.close(fig)


def impulse_chart():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), dpi=150, sharey=False)
    cases = [('yaw', 'eval_v2_yaw_turn.json', 'eval_v2_pitch.json')]
    for ax, axis in zip(axes, ('z', 'y')):
        f = 'eval_v2_yaw_turn.json' if axis == 'z' else 'eval_v2_pitch.json'
        for tail, c in (('active', 'C0'), ('locked', 'C3')):
            r = [x for x in rows(f) if x.get('impulse_axis') == axis and x['tail'] == tail]
            J = [x['impulse_nms'] for x in r]; ok = [x['episodes']-x['falls'] for x in r]
            ax.bar([j+(-.06 if tail == 'active' else .06) for j in J], ok, width=.11, color=c, label=f'tail {tail}')
        ax.set_xlabel(f'{"yaw" if axis == "z" else "pitch"} impulse at 4 m/s (N·m·s)'); ax.set_ylabel('episodes survived / 3')
        ax.set_ylim(0, 3.4); ax.legend(fontsize=7)
    fig.suptitle('Impulse tolerance, v2 policy, tail free vs held (T1, MuJoCo)', fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'impulse-tolerance.png'); plt.close(fig)


def turn_chart():
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=150)
    for f, label, c, st in (('eval_v2_yaw_turn.json', 'v2 (fast, no turn training)', 'C0', 'o'), ('eval_v4_yaw_turn.json', 'v4 (yaw commands ±2.5 rad/s)', 'C2', '^')):
        r = [x for x in rows(f) if 'turn_cmd_rad_s' in x and x['tail'] == 'active']
        for vx, mk in ((2., .5), (4., .75), (6., 1.)):
            rr = [x for x in r if x['cmd_vx'] == vx]
            ax.plot([x['turn_rad_cmd'] for x in rr], [x['turn_rad'] if x['turn_rad'] is not None else 0 for x in rr], st+'-', color=c, alpha=mk, label=f'{label}, {vx:.0f} m/s')
    ax.plot([0, 5.2], [0, 5.2], ':', color='grey', lw=.8)
    ax.set_xlabel('commanded heading change in 2 s (rad)'); ax.set_ylabel('achieved heading change (rad)')
    ax.set_title('Fast turns while running, tail active, no falls in any case shown (T1, MuJoCo)', fontsize=8)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/'turn-tracking.png'); plt.close(fig)


def gait_chart():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), dpi=150)
    for f, label, c, mk in (('eval_yaw.json', 'v2 (bound)', 'C0', 'o'), ('eval_v5b_alt.json', 'v5b (clock schedule)', 'C1', 'D')):
        r = [x for x in rows(f) if 'impulse_nms' not in x and 'turn_cmd_rad_s' not in x and x['tail'] == 'active' and x['falls'] == 0 and x['cmd_vx'] > 0]
        v = [x['mean_vx'] for x in r]
        if 'lr_phase' in r[0]:
            axes[0].plot(v, [x['lr_phase'] for x in r], mk+'-', color=c, label=label)
        axes[1].plot(v, [x['stride_hz'] for x in r], mk+'-', color=c, label=label)
    axes[0].axhline(.5, color='grey', ls=':', lw=.8); axes[0].text(.3, .52, 'alternating (0.5)', fontsize=7, color='grey')
    axes[0].axhline(.85, color='grey', ls=':', lw=.8); axes[0].text(.3, .87, 'bound (v2 measured 0.85)', fontsize=7, color='grey')
    axes[0].set_ylim(0, 1); axes[0].set_xlabel('body speed (m/s)'); axes[0].set_ylabel('right touchdown phase after left (stride)')
    vv = [0, 8]; axes[1].plot(vv, [1.6+.14*x for x in vv], ':', color='grey', lw=.8, label='stride clock 1.6+0.14v')
    axes[1].set_xlabel('body speed (m/s)'); axes[1].set_ylabel('stride frequency (Hz)'); axes[1].set_ylim(0, 6)
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=6)
    fig.suptitle('Gait symmetry and stride rate: bound (v2) vs clock-scheduled alternating run (v5b), T1 MuJoCo', fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'gait-phase.png'); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    speed_chart(); impulse_chart(); turn_chart(); gait_chart()
    print('wrote', sorted(p.name for p in OUT.glob('*.png')))
