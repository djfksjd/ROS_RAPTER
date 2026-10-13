"""Where does the tail pitch motor torque go? Exact split of the tail-pitch row of the equation of motion at every 1 ms step:
  tau_motor = M[t,t]*qacc_t  (the tail's own swing relative to the trunk)
            + sum_j M[t,j]*qacc_j, j in trunk translation / trunk rotation / other joints  (holding the tail on a moving trunk)
            + bias (gravity, Coriolis) - passive - constraint
Deterministic rollout, init seed 96001, 30 s, first 5 s skipped. Usage (project root): .venv-sim/bin/python tail_dynamics.py <out.json>"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
import run_env
from run_env import RunEnv, LINK_ETA, MOTOR_CLASSES
from train_run import env_kwargs


def probe(run, cmd, locked=False):
    a = json.loads((ROOT/'sim/rl/runs'/run/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(a), 'episode_s': 35., 'level': 0.}, randomize=False, seed=7); env.init_seed = 96001
    model, venv = load(str(ROOT/'sim/rl/runs'/run/'model.zip'), env); venv.reset(); venv.reset(); m, d = env.model, env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; env.lock_tail = locked; obs = venv.normalize_obs(env._obs()[None])
    it = env.ix['tail_pitch_joint']; va, aa = env.v_adr[it], env.act[it]; vec, out_ = np.zeros(m.nv), np.zeros(m.nv); rows = []; step0 = mujoco.mj_step
    others = [j for j in range(6, m.nv) if j != va]

    def mj_step(m_, d_):
        step0(m_, d_); p_ = []
        for sel in ([va], [0, 1, 2], [3, 4, 5], others):
            vec[:] = 0.; vec[sel] = d_.qacc[sel]; mujoco.mj_mulM(m_, d_, out_, vec); p_.append(out_[va])
        vec[:] = 0.; vec[va] = 1.; mujoco.mj_mulM(m_, d_, out_, vec)
        rows.append((d_.actuator_force[aa], *p_, d_.qfrc_bias[va], -d_.qfrc_passive[va], -d_.qfrc_constraint[va], out_[va], d_.qvel[va], d_.qacc[2], d_.qacc[4]))
    run_env.mujoco.mj_step = mj_step
    try:
        for k in range(1500):
            raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
            assert not term
    finally:
        run_env.mujoco.mj_step = step0
    r = np.array(rows)[5000:]; tau = r[:, 0]; parts = dict(own_swing=r[:, 1], trunk_translation=r[:, 2], trunk_rotation=r[:, 3], other_joints=r[:, 4], gravity_coriolis=r[:, 5], passive=r[:, 6], constraint=r[:, 7])
    resid = tau-sum(parts.values()); ac = tau-tau.mean(); var = float(ac.var()); lim = MOTOR_CLASSES['B'][0]*LINK_ETA
    out = dict(run=run, cmd=cmd, locked=locked, tau_mean=float(tau.mean()), tau_std=float(tau.std()), residual_rms=float(np.sqrt(np.mean(resid**2))), inertia_about_joint=float(r[:, 8].mean()),
               at_peak_limit_frac=float(np.mean(np.abs(tau) > .98*lim)), peak_limit_Nm=lim, qd_absmax=float(np.abs(r[:, 9]).max()), qd_rms=float(np.sqrt(np.mean(r[:, 9]**2))),
               trunk_vert_acc_rms=float(r[:, 10].std()), trunk_pitch_acc_rms=float(r[:, 11].std()),
               std={k: float(v.std()) for k, v in parts.items()}, mean={k: float(v.mean()) for k, v in parts.items()},
               share_of_variance={k: float(np.mean((v-v.mean())*ac)/var) for k, v in parts.items()})
    env.close(); return out


res = []
for run, cmd, lk in [('rev2_a8_h2', 7., False), ('rev2_n10', 7., False), ('rev2_td340', 7., False), ('rev2_td340', 5.5, False), ('rev2_td340', 4., False), ('rev2_td340', 7., True)]:
    o = probe(run, cmd, lk); res.append(o)
    print(run, cmd, 'L' if lk else '', 'tau std %.1f I %.3f at_limit %.3f qd_max %.1f resid %.2g | trunk acc vert %.1f pitch %.1f' % (o['tau_std'], o['inertia_about_joint'], o['at_peak_limit_frac'], o['qd_absmax'], o['residual_rms'], o['trunk_vert_acc_rms'], o['trunk_pitch_acc_rms']))
    print('   std', {k: round(v, 1) for k, v in o['std'].items()}); print('   share', {k: round(v, 2) for k, v in o['share_of_variance'].items()}, flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
