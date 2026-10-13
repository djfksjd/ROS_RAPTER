"""Tail-heat probe (no training): why does the tail pitch motor exceed its continuous rating at 7 m/s in rev2_td340?
Deterministic rollouts (init seed 96001, 30 s, first 5 s skipped) of the three model_rev 2 policies. Per joint: mean of
(motor torque / continuous torque)^2. For the tail pitch joint, from 1 ms samples: torque mean (DC) and oscillating part (AC),
angle statistics, time near the joint limit, servo target minus angle, spectrum peak against the stride frequency, the
gravity/bias torque, and the trunk pitch. Variant 'locked': same policy with the tail held at its nominal pose by the servo.
Usage (project root): .venv-sim/bin/python tail_probe.py <out.json>"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
import run_env
from run_env import RunEnv
from train_run import env_kwargs
SECONDS, SKIP = 30., 250


def probe(run, cmd, locked=False):
    a = json.loads((ROOT/'sim/rl/runs'/run/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(a), 'episode_s': SECONDS+5., 'level': 0.}, randomize=False, seed=7); env.init_seed = 96001
    model, venv = load(str(ROOT/'sim/rl/runs'/run/'model.zip'), env); venv.reset(); venv.reset(); m, d = env.model, env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; env.lock_tail = locked; obs = venv.normalize_obs(env._obs()[None])
    it = env.ix['tail_pitch_joint']; qa, va, aa = env.q_adr[it], env.v_adr[it], env.act[it]; sub, ctl = [], []; step0 = mujoco.mj_step

    def mj_step(m_, d_):
        step0(m_, d_); sub.append((d_.qpos[qa], d_.qvel[va], d_.actuator_force[aa], d_.qfrc_bias[va], d_.ctrl[aa], d_.qacc[va]))
    run_env.mujoco.mj_step = mj_step; fell = False
    try:
        for k in range(int(SECONDS/.02)):
            raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
            R = d.xmat[env.base].reshape(3, 3); ctl.append(np.r_[env.heat_inst, info['v_body'][0], np.arcsin(-R[2, 0]), info['loaded']['left'], info['loaded']['right']])
            if term: fell = True; break
    finally:
        run_env.mujoco.mj_step = step0
    sub, ctl = np.array(sub), np.array(ctl); n = len(env.active); s = sub[SKIP*20:]; c = ctl[SKIP:]
    drive = [i for i in range(n) if env.motor_cont[i] < 1e8]; heat = {env.active[i].replace('_joint', ''): float(c[:, i].mean()) for i in drive}
    q, qd, tau, bias, tgt = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]; cont = float(env.motor_cont[it]); lo, hi = m.jnt_range[m.joint('tail_pitch_joint').id]
    td = np.flatnonzero((c[1:, n+2] > 0) & (c[:-1, n+2] == 0)); stride_hz = float(1/(np.mean(np.diff(td))*.02)) if len(td) > 2 else None
    f = np.fft.rfftfreq(len(tau), .001); P = np.abs(np.fft.rfft(tau-tau.mean()))**2; band = lambda a_, b_: float(P[(f >= a_) & (f < b_)].sum()/P.sum())
    res = dict(run=run, cmd=cmd, locked=locked, fell=fell, t=float(len(ctl)*.02), speed=float(c[:, n].mean()), trunk_pitch_mean=float(c[:, n+1].mean()), trunk_pitch_std=float(c[:, n+1].std()), heat=heat,
               tail=dict(cont_Nm=cont, heat=heat['tail_pitch'], heat_dc=float((tau.mean()/cont)**2), heat_ac=float((tau.std()/cont)**2), tau_mean=float(tau.mean()), tau_std=float(tau.std()), tau_absmax=float(np.abs(tau).max()),
                         q_mean=float(q.mean()), q_std=float(q.std()), q_min=float(q.min()), q_max=float(q.max()), range=[float(lo), float(hi)], near_limit_frac=float(np.mean((q < lo+.03) | (q > hi-.03))),
                         target_mean=float(tgt.mean()), target_std=float(tgt.std()), target_beyond_range_frac=float(np.mean((tgt < lo) | (tgt > hi))), err_mean=float((tgt-q).mean()),
                         bias_mean=float(bias.mean()), bias_std=float(bias.std()), qd_rms=float(np.sqrt(np.mean(qd**2))), stride_hz=stride_hz, peak_hz=float(f[1:][P[1:].argmax()]),
                         power_below_stride=band(0, .5*(stride_hz or 1)), power_stride_to_4x=band(.5*(stride_hz or 1), 4.5*(stride_hz or 1)), power_above_20hz=band(20., 1e9),
                         mech_power_pos_W=float(np.mean(np.clip(tau*qd, 0, None)))))
    env.close(); return res


out = []
for run in ('rev2_a8_h2', 'rev2_n10', 'rev2_td340'):
    for cmd in (2., 4., 5.5, 7.):
        r = probe(run, cmd); out.append(r); t = r['tail']
        print(run, cmd, 'spd %.2f' % r['speed'], 'heat', {k: round(v, 2) for k, v in r['heat'].items() if 'right' not in k}, '| tail dc %.2f ac %.2f tau_mean %.1f std %.1f q %.2f±%.2f [%.2f,%.2f] lim %.2f tgt %.2f beyond %.2f bias %.1f trunk %.3f stride %.2f peak %.2f' % (
            t['heat_dc'], t['heat_ac'], t['tau_mean'], t['tau_std'], t['q_mean'], t['q_std'], t['q_min'], t['q_max'], t['near_limit_frac'], t['target_mean'], t['target_beyond_range_frac'], t['bias_mean'], r['trunk_pitch_mean'], t['stride_hz'] or 0, t['peak_hz']), flush=True)
for cmd in (4., 7.):
    r = probe('rev2_td340', cmd, True); out.append(r); t = r['tail']
    print('LOCKED', cmd, 'fell', r['fell'], 't %.1f' % r['t'], 'spd %.2f' % r['speed'], 'tail heat %.2f dc %.2f ac %.2f tau_mean %.1f' % (t['heat'], t['heat_dc'], t['heat_ac'], t['tau_mean']), {k: round(v, 2) for k, v in r['heat'].items() if 'right' not in k}, flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
