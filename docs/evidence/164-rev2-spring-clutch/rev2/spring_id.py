"""Knee spring comparison without training: inverse-dynamics replay on the recorded baseline gait.

Record the frozen policy on the unmodified model (model_rev 1) at each command. Per 1 ms physics step and knee:
tau_req = knee actuator torque + knee spring torque = the torque the knee drive plus its spring delivered for that motion.
For a candidate passive latch spring (k, q0) the same state machine as RunEnv(latch_fallback=False) is replayed on the
recorded knee angle and foot-load flags (engage in swing at q <= q0, hold through the next stance, release at lift-off);
the spring is unilateral (torque -k (q - q0) for q > q0 while latched). The motor then has to deliver
tau_motor = tau_req - tau_spring at the joint. Reported per candidate: heat = mean (motor-side torque / continuous)^2,
fraction of steps above the drive envelope, peak joint torque, and the energy lost at release.
The motion is held fixed: this ranks springs for THAT gait, it does not predict the gait a retrained policy would choose."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl')); sys.path.insert(0, str(ROOT/'modeling'))
from evaluate import load
from run_env import RunEnv, MOTOR_CLASSES, LINK_ETA, motor_available
from train_run import env_kwargs
RUN = ROOT/'sim/rl/runs/scr_a8_h2'; OUT = Path(sys.argv[1])
CONT = .4*MOTOR_CLASSES['B'][0]


def record(cmd, seed, seconds=30.):
    env = RunEnv(**{**env_kwargs(json.loads((RUN/'args.json').read_text())), 'episode_s': seconds+5., 'level': 0.}, randomize=False, seed=7); env.init_seed = seed
    model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset(); m, d = env.model, env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; obs = venv.normalize_obs(env._obs()[None])
    ki = [env.ix[f'{s}_knee_pitch_joint'] for s in ('left', 'right')]; qa, va, aa = env.q_adr[ki], env.v_adr[ki], env.act[ki]
    sub, ctl = [], []; clamp = env._clamp

    def hook():
        r = clamp(); mj = sys.modules['mujoco']; return r
    import mujoco
    step0 = mujoco.mj_step

    def mj_step(m_, d_):
        step0(m_, d_); sub.append(np.r_[d_.qpos[qa], d_.qvel[va], d_.actuator_force[aa], d_.qfrc_passive[va]])
    import run_env; run_env.mujoco.mj_step = mj_step
    try:
        for i in range(int(seconds/.02)):
            ctl.append(np.r_[d.qpos[qa], [env.loaded['left'], env.loaded['right']]])
            raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
            assert not term
    finally:
        run_env.mujoco.mj_step = step0
    kr = env.knee_ratio; env.close()
    return np.array(sub), np.array(ctl), kr


def latch_replay(q_ctl, loaded, q0):
    """RunEnv latch logic without fallback -> boolean per control step."""
    on = np.zeros(len(q_ctl), bool); was = False; seen = False; armed = False
    for i in range(len(q_ctl)):
        if loaded[i]:
            o = was
        else:
            o = False
            if q_ctl[i] > q0:
                armed = True
            if not was and q_ctl[i] <= q0 and armed:
                o = True
            elif was and not seen:
                o = True
        if loaded[i]:
            seen = True
        elif seen:
            o, seen, armed = False, False, False
        on[i] = was = o
    return on


def evaluate(sub, ctl, kr, k, q0, skip=250):
    out = []
    for j in range(2):
        q, qd, req = sub[:, j], sub[:, 2+j], sub[:, 4+j]+sub[:, 6+j]
        on = np.repeat(latch_replay(ctl[:, j], ctl[:, 2+j].astype(bool), q0), 20) if k > 0 else np.zeros(len(q), bool)
        ts = np.where(on, -k*np.clip(q-q0, 0., None), 0.)
        tm = req-ts; N = np.interp(q, *kr)
        ms = tm/(N*LINK_ETA)
        drive = N*LINK_ETA*motor_available('B', N*qd); brake = N*LINK_ETA*MOTOR_CLASSES['B'][0]
        lim = np.where(tm*qd >= 0, drive, brake)
        s = slice(skip*20, None)
        rel = np.flatnonzero(on[:-1] & ~on[1:]); e_loss = float(np.sum(.5*k*np.clip(q[rel]-q0, 0, None)**2))
        out.append(dict(heat=float(np.mean((ms[s]/CONT)**2)), over=float(np.mean(np.abs(tm[s]) > lim[s]+1e-9)), excess_rms=float(np.sqrt(np.mean(np.clip(np.abs(tm[s])-lim[s], 0, None)**2))),
                        peak=float(np.abs(tm[s]).max()), latched=float(on[s].mean()), pos_W=float(np.mean(np.clip(tm[s]*qd[s], 0, None))), neg_W=float(np.mean(np.clip(-tm[s]*qd[s], 0, None))),
                        release_loss_W=e_loss/(len(q)*.001), spring_peak=float(np.abs(ts[s]).max())))
    return {key: float(max(o[key] for o in out)) if key in ('heat', 'over', 'peak', 'excess_rms', 'spring_peak') else float(np.mean([o[key] for o in out])) for key in out[0]}


if __name__ == '__main__':
    res = {}
    K = np.arange(0, 601, 10); Q0 = np.round(np.arange(.80, 1.801, .02), 3)
    for cmd in (2., 4., 7.):
        sub, ctl, kr = record(cmd, 95001)
        req = sub[:, 4:6]+sub[:, 6:8]; st = np.repeat(ctl[:, 2:4].astype(bool), 20, axis=0)
        td = [ctl[i, j] for j in range(2) for i in range(1, len(ctl)) if ctl[i, 2+j] and not ctl[i-1, 2+j]]
        swing_min = [ctl[a:b, j].min() for j in range(2) for a, b in zip(*[np.flatnonzero(np.diff(np.r_[1, ctl[:, 2+j], 1]) == x) for x in (-1, 1)]) if b > a]
        info = dict(knee_touchdown_mean=float(np.mean(td)), knee_touchdown_p05=float(np.quantile(td, .05)), knee_touchdown_p95=float(np.quantile(td, .95)),
                    swing_min_mean=float(np.mean(swing_min)), swing_min_p05=float(np.quantile(swing_min, .05)), swing_min_p95=float(np.quantile(swing_min, .95)),
                    stance_knee_mean=float(sub[:, :2][st].mean()), stance_knee_max=float(sub[:, :2][st].max()), req_peak=float(np.abs(req).max()),
                    req_stance_rms=float(np.sqrt(np.mean(req[st]**2))), actual_motor_heat=float(max(np.mean((sub[5000:, 4+j]/(np.interp(sub[5000:, j], *kr)*LINK_ETA*CONT))**2) for j in range(2))))
        grid = np.array([[evaluate(sub, ctl, kr, k, q0)['heat'] for q0 in Q0] for k in K])
        over = np.array([[evaluate(sub, ctl, kr, k, q0)['over'] for q0 in Q0] for k in K])
        i, j = np.unravel_index(grid.argmin(), grid.shape)
        cands = {'none': (0, 1.), 'k170_q0.956': (168.970423, .956), 'k170_q1.3': (168.970423, 1.3), 'k170_q1.1': (168.970423, 1.1), 'k300_q1.1': (300*11.2912/11.36, 1.1), 'best_heat': (float(K[i])*11.2912/11.36, float(Q0[j]))}
        res[f'{cmd:g}'] = dict(gait=info, candidates={n: dict(k=k, q0=q0, **evaluate(sub, ctl, kr, k, q0)) for n, (k, q0) in cands.items()},
                               best=dict(k_design=float(K[i]), q0=float(Q0[j]), heat=float(grid[i, j])), heat_grid=grid.tolist(), over_grid=over.tolist())
        print(cmd, json.dumps(info)); [print('  ', n, {a: round(b, 3) for a, b in v.items()}) for n, v in res[f'{cmd:g}']['candidates'].items()]
    res['K_design'] = K.tolist(); res['Q0'] = Q0.tolist(); res['note'] = 'grid stiffness is used as given (N m/rad at the joint); best_heat k is rescaled like the env (x 11.2912/11.36)'
    (OUT/'spring_id.json').write_text(json.dumps(res))
