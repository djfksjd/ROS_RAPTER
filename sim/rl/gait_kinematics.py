"""Joint kinematics of a RunEnv (T1) policy per stride, compared with ostrich running data (evidence 86 §9).

Anatomical angles from the MJCF joint coordinates (R-02: hip + = thigh back, knee + = flexion, ankle - = flexion):
  femur angle from vertical, + forward  = -hip
  knee included angle                   = 180 - knee (deg)
  intertarsal (ankle) included angle    = 180 + ankle (deg); 180 = straight
Per foot, stance = debounced loaded phases (>= 60 ms). Reported per phase: mean/min/max of each angle, the ankle
excursion during stance, the ankle's minimum included angle in swing, and the world thigh angular velocity (hip rate + torso
pitch rate) over the last 25 % of swing (> 0 = the leg retracts before touchdown, the ostrich 'constant retraction' rule).
Usage: .venv-sim/bin/python sim/rl/gait_kinematics.py sim/rl/runs/<name>/model.zip --cmd 6 [--out k.json --plot k.png]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load  # noqa: E402
from run_env import RunEnv  # noqa: E402

# ostrich reference (research report ~/Documents/Ostrich_Running_Mechanism_Research_20260929, evidence ids in brackets)
OSTRICH = {'ankle_stance_deg': 168, 'ankle_swing_min_deg': 45, 'ankle_lock_transition_deg': 115,
           'hip_range_3p3mps_deg': 12.7, 'note': 'Schaller 2009 [16]: 60 km/h video, swing flexion to 45 deg, stance 168 deg, '
           'rapid flexion at toe-off, full extension before touchdown; Rubenson 2007/2011 [4,5]: ankle nearly static in stance, '
           'hip flexion range 12.7 deg at 3.3 m/s, femur near horizontal; Daley & Biewener 2006 [20], Blum 2014 [11]: '
           'swing-leg retraction at a constant rate before touchdown'}


def rollout(model_path, cmd, seconds=8., seed=1000, clutch=False, couple=False):
    env = RunEnv(randomize=False, seed=seed, ankle_clutch=clutch, couple_ankle=couple)
    model, venv = load(model_path, env)
    obs = venv.reset()
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    idx = {n: i for i, n in enumerate(env.active)}
    rec = []
    for _ in range(int(seconds/.02)):
        a, _ = model.predict(obs, deterministic=True)
        raw, _, term, _, info = env.step(a[0])
        obs = venv.normalize_obs(raw[None])
        q, qd = env.data.qpos[env.q_adr], env.data.qvel[env.v_adr]
        rec.append({s: dict(hip=q[idx[f'{s}_hip_pitch_joint']], knee=q[idx[f'{s}_knee_pitch_joint']],
                            ankle=q[idx[f'{s}_ankle_pitch_joint']],
                            hipd=qd[idx[f'{s}_hip_pitch_joint']]+env.body_velocity(env.base)[0][1],  # world thigh rate
                            loaded=bool(info['loaded'][s])) for s in ('left', 'right')})
        if term:
            break
    return rec


def phases(flags, start):
    """Debounced stance intervals [i, j) (>= 3 control steps)."""
    out, i, n = [], start, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j-i >= 3:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def analyse(rec, skip=100):
    res = {}
    for s in ('left', 'right'):
        hip = np.array([r[s]['hip'] for r in rec]); knee = np.array([r[s]['knee'] for r in rec])
        ank = np.array([r[s]['ankle'] for r in rec]); hipd = np.array([r[s]['hipd'] for r in rec])
        st = phases([r[s]['loaded'] for r in rec], skip)
        femur = np.degrees(-hip); knee_inc = 180-np.degrees(knee); ank_inc = 180+np.degrees(ank)
        stance_mask = np.zeros(len(rec), bool)
        for i, j in st:
            stance_mask[i:j] = True
        swing = [(st[k][1], st[k+1][0]) for k in range(len(st)-1)]
        exc = [float(ank_inc[i:j].max()-ank_inc[i:j].min()) for i, j in st]
        sw_min = [float(ank_inc[i:j].min()) for i, j in swing if j > i]
        retr = [float(np.mean(hipd[j-max(1, (j-i)//4):j])) for i, j in swing if j-i >= 4]
        m = np.arange(len(rec)) >= skip
        res[s] = {
            'stance': {'femur_deg': round(float(femur[stance_mask].mean()), 1), 'knee_deg': round(float(knee_inc[stance_mask].mean()), 1),
                       'ankle_deg': round(float(ank_inc[stance_mask].mean()), 1), 'ankle_excursion_deg': round(float(np.mean(exc)), 1)},
            'swing': {'femur_deg': round(float(femur[m & ~stance_mask].mean()), 1), 'knee_deg': round(float(knee_inc[m & ~stance_mask].mean()), 1),
                      'ankle_min_deg': round(float(np.mean(sw_min)), 1) if sw_min else None,
                      'late_swing_hip_rate_rad_s': round(float(np.mean(retr)), 2) if retr else None,
                      'retracting_fraction': round(float(np.mean(np.array(retr) > 0)), 2) if retr else None},
            'hip_range_deg': round(float(femur[m].max()-femur[m].min()), 1),
            'n_strides': len(st)}
    return res


def plot(rec, path, title, skip=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['AppleGothic', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 8})
    s = 'left'
    st = phases([r[s]['loaded'] for r in rec], skip)
    if len(st) < 3:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), dpi=150)
    for (i0, _), (i1, _) in zip(st[:-1], st[1:]):
        x = np.linspace(0, 1, i1-i0)
        seg = rec[i0:i1]
        stance_end = next((k for k in range(len(seg)) if not seg[k][s]['loaded']), len(seg))/len(seg)
        for ax, key, f in ((axes[0], 'hip', lambda v: -np.degrees(v)), (axes[1], 'knee', lambda v: 180-np.degrees(v)),
                           (axes[2], 'ankle', lambda v: 180+np.degrees(v))):
            ax.plot(x, f(np.array([r[s][key] for r in seg])), color='C0', alpha=.35, lw=.8)
            ax.axvspan(0, stance_end, color='C1', alpha=.03)
    axes[0].set_title('femur from vertical (+fwd, deg)'); axes[1].set_title('knee included (deg)'); axes[2].set_title('intertarsal included (deg)')
    axes[2].axhline(168, color='C3', ls='--', lw=.8); axes[2].text(.55, 170, 'ostrich stance 168', color='C3', fontsize=6)
    axes[2].axhline(45, color='C3', ls=':', lw=.8); axes[2].text(.55, 47, 'ostrich swing min 45', color='C3', fontsize=6)
    for ax in axes:
        ax.set_xlabel('left stride (0 = touchdown)'); ax.grid(alpha=.3)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--cmd', type=float, nargs='+', default=[4., 6.])
    p.add_argument('--out'); p.add_argument('--plot')
    p.add_argument('--ankle-clutch', action='store_true')
    p.add_argument('--couple-ankle', action='store_true')
    a = p.parse_args()
    out = {'ostrich': OSTRICH}
    for c in a.cmd:
        rec = rollout(a.model, c, clutch=a.ankle_clutch, couple=a.couple_ankle)
        out[str(c)] = analyse(rec)
        print(c, json.dumps(out[str(c)]), flush=True)
        if a.plot and c == a.cmd[-1]:
            plot(rec, a.plot, f'{Path(a.model).parent.name}, {c} m/s command, left leg, stance shaded (T1 MuJoCo)')
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()


def compare_plot(runs, path, cmd=6.):
    """Intertarsal and femur angle over the left stride for several (label, model, clutch) runs, ostrich lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['AppleGothic', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 8})
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7), dpi=150)
    for (label, model, clutch), c in zip(runs, ('C0', 'C2', 'C1')):
        rec = rollout(model, cmd, clutch=clutch)
        for side, ax_i, ls in (('left', 0, '-'), ('right', 1, '--')):
            st = phases([r[side]['loaded'] for r in rec], 100)
            prof = []
            for (i0, _), (i1, _) in zip(st[:-1], st[1:]):
                a = np.array([180+np.degrees(r[side]['ankle']) for r in rec[i0:i1]])
                prof.append(np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(a)), a))
            if prof:
                axes[ax_i].plot(np.linspace(0, 1, 50), np.mean(prof, 0), ls, color=c, label=label)
        res = analyse(rec)
        axes[2].bar([label+' L', label+' R'], [res['left']['swing']['ankle_min_deg'], res['right']['swing']['ankle_min_deg']], color=c)
    for ax, t in zip(axes[:2], ('left leg', 'right leg')):
        ax.axhline(168, color='C3', ls='--', lw=.7); ax.axhline(45, color='C3', ls=':', lw=.7)
        ax.text(.02, 150, 'ostrich stance 168', color='C3', fontsize=6); ax.text(.02, 50, 'ostrich swing 45', color='C3', fontsize=6)
        ax.set_ylim(30, 175); ax.set_title(f'intertarsal included angle, {t}'); ax.set_xlabel('stride (0 = touchdown)'); ax.grid(alpha=.3)
    axes[0].legend(fontsize=6)
    axes[2].axhline(65, color='k', ls=':', lw=.7); axes[2].text(-.4, 67, 'reward target 65', fontsize=6)
    axes[2].axhline(45, color='C3', ls=':', lw=.7); axes[2].set_title('swing minimum (deg)'); axes[2].tick_params(axis='x', labelsize=6, rotation=30)
    fig.suptitle(f'Joint flexion at {cmd:.0f} m/s command, mean stride (T1 MuJoCo, not the real robot)', fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
