"""Read-only policy diagnosis. No learn(), model changes, or training processes.

Default budget: 2 policies x 3 speeds x 4 yaw commands x 2 seeds, 15 s each.
5 s straight + 10 s turn; steady stats over (8,15] s as in turn_eval.
Joint angle/torque/limits: 1 ms during (5,15]; contact forces: 20 ms.
Counterfactuals replace ONLY observation 8 on the same cached state, every 1 s.
Results are episode-level raw JSONL plus compressed 50 Hz traces (not gates).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import mujoco
import numpy as np
import torch
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs

EVIDENCE = Path(__file__).resolve().parent
TERMS = ('track_lin', 'progress', 'track_yaw', 'yaw_err', 'orientation', 'leg_sym',
         'joint_limit', 'action_rate', 'heat', 'slip_lin')


def rollout(run, speed, yaw, seed, trace_dir):
    d = ROOT/'sim/rl/runs'/run
    args = json.loads((d/'args.json').read_text())
    kw = {**env_kwargs(args), 'level': 0., 'episode_s': 20., 'disturb_items': [],
          'roll_impulse': None, 'yaw_impulse': None, 'cmd_max': [0., 0., 0.],
          'cmd_mix': None, 'init_speed': False}
    env = RunEnv(**kw, randomize=False, seed=seed)
    env.init_seed = seed
    policy, vn = load(str(d/'model.zip'), env)
    vn.reset()
    env.resample_steps = 0
    m, dd = env.model, env.data
    idx = np.array([i for i, n in enumerate(env.active) if 'hip_yaw' in n or 'hip_roll' in n])
    names = [env.active[i] for i in idx]
    ai = [list(env.policy_idx).index(i) for i in idx]
    joint_samples, traces, cf, contact_samples = [], [], [], []
    contact_pairs = {}
    original = env.rz.after_substep

    def sample(e, dt):
        original(e, dt)
        if e.steps >= 250:
            torque = dd.actuator_force[env.act[idx]].copy()
            limits = m.actuator_forcerange[env.act[idx]]
            joint_samples.append(np.concatenate((dd.qpos[env.q_adr[idx]], torque,
                np.where(torque >= 0, limits[:, 1], -limits[:, 0]), dd.qvel[env.v_adr[idx]])))

    env.rz.after_substep = sample
    raw, fell = env._obs(), None
    norm_info = dict(std8=float(np.sqrt(vn.obs_rms.var[8]+vn.epsilon)), mean8=float(vn.obs_rms.mean[8]),
                     clip_obs=float(vn.clip_obs), values={})
    for c in (-1.5, -1., -.5, 0., .5, 1., 1.5):
        o = raw.copy(); o[8] = .5*c
        norm_info['values'][str(c)] = float(vn.normalize_obs(o[None])[0, 8])
    try:
        for k in range(750):
            env.command = np.array([speed, 0., yaw if k >= 250 else 0.])
            raw[6:9] = env.command*np.array([.1, 2., .5])
            if k >= 250 and k % 50 == 0:
                obs = np.repeat(raw[None], 2, axis=0); obs[:, 8] = [0., .5]
                actions = policy.predict(vn.normalize_obs(obs), deterministic=True)[0]
                cf.append(dict(t=k*.02, actions_yaw0=actions[0].tolist(), actions_yaw1=actions[1].tolist(),
                               l2=float(np.linalg.norm(actions[1]-actions[0])),
                               hip_delta_target_rad=((actions[1, ai]-actions[0, ai])*env.scale[idx]).tolist()))
            action = policy.predict(vn.normalize_obs(raw[None]), deterministic=True)[0][0]
            raw, reward, term, trunc, info = env.step(action)
            w, v = env.body_velocity(env.base)
            R = dd.xmat[env.base].reshape(3, 3)
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            weighted = [env.weights.get(n, 0)*info['terms'].get(n, 0) for n in TERMS]
            traces.append([dd.time, *v, *w, roll, info['tilt'], reward,
                           *dd.qpos[env.q_adr[idx]], *env.ctrl_prev[idx], *action[ai], *weighted])
            if k >= 250:
                for j in range(dd.ncon):
                    contact = dd.contact[j]
                    gs = (int(contact.geom1), int(contact.geom2))
                    if not any(g in env.ground for g in gs):
                        continue
                    f = np.zeros(6); mujoco.mj_contactForce(m, dd, j, f)
                    pair = '|'.join(m.geom(g).name or str(g) for g in gs)
                    contact_pairs[pair] = dict(mu=contact.friction.tolist(), dim=int(contact.dim),
                                              geom_mu=[m.geom_friction[g].tolist() for g in gs],
                                              priority=[int(m.geom_priority[g]) for g in gs])
                    if f[0] > 1.:
                        # Sliding tangent magnitude / isotropic Coulomb capacity; not a full 6D cone test.
                        contact_samples.append([contact.friction[0], np.linalg.norm(f[1:3])/(contact.friction[0]*f[0])])
            if term or trunc:
                fell = dd.time if term else None
                break
    finally:
        env.rz.after_substep = original
        vn.close()
    T, J = np.asarray(traces), np.asarray(joint_samples)
    S = T[T[:, 0] > 8.+1e-6]
    joints = {}
    n = len(idx)
    for j, i in enumerate(idx):
        q, tau, cap, vel = J[:, j], J[:, n+j], J[:, 2*n+j], J[:, 3*n+j]
        joints[names[j]] = dict(range_rad=[float(env.lo[i]), float(env.hi[i])],
            scale_rad=float(env.scale[i]), angle_min=float(q.min()), angle_max=float(q.max()),
            angle_p01_p99=np.quantile(q, [.01, .99]).tolist(),
            near_stop_share=float(np.mean((q <= env.lo[i]+.01) | (q >= env.hi[i]-.01))),
            torque_rms_nm=float(np.sqrt(np.mean(tau*tau))), torque_peak_nm=float(np.max(np.abs(tau))),
            torque_limit_min_max_nm=[float(cap.min()), float(cap.max())],
            torque_at_limit_share=float(np.mean((cap > 1e-6) & (np.abs(tau) >= .98*cap))),
            peak_velocity_rad_s=float(np.max(np.abs(vel))),
            action_at_clip_share=float(np.mean(np.abs(T[250:, 10+2*n+j]) >= .98)))
    columns = ['time', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'roll', 'tilt', 'reward']
    columns += [f'{what}:{name}' for what in ('q', 'target', 'action') for name in names]
    columns += [f'weighted:{term}' for term in TERMS]
    trace_name = f'{run}_v{speed:g}_w{yaw:g}_s{seed}.npz'
    np.savez_compressed(trace_dir/trace_name, data=T, columns=np.array(columns))
    C = np.asarray(contact_samples)
    return dict(run=run, speed=speed, yaw=yaw, seed=seed, t_fall=fell, complete=len(T) == 750 and fell is None,
        control_steps=len(T), joint_samples=len(J), trace=trace_name,
        steady=dict(vx=float(S[:, 1].mean()), vy=float(S[:, 2].mean()), wz=float(S[:, 6].mean()),
                    roll_mean=float(S[:, 7].mean()), roll_abs_p95=float(np.quantile(np.abs(S[:, 7]), .95)),
                    tilt_max=float(S[:, 8].max()), reward_mean=float(S[:, 9].mean())),
        weighted_reward_per_second={term: float(S[:, 10+3*n+j].mean()) for j, term in enumerate(TERMS)},
        joints=joints, normalization=norm_info, counterfactual=cf, contact_pairs=contact_pairs,
        contact=dict(samples=len(C), mu_min=float(C[:, 0].min()), mu_max=float(C[:, 0].max()),
                     tangent_utilization_p95=float(np.quantile(C[:, 1], .95)),
                     tangent_utilization_ge98_share=float(np.mean(C[:, 1] >= .98))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', default=['l80_r1_6M', 'l80_t1'])
    p.add_argument('--speeds', nargs='+', type=float, default=[2., 4., 6.])
    p.add_argument('--yaws', nargs='+', type=float, default=[.5, -.5, 1., 1.5])
    p.add_argument('--seeds', nargs='+', type=int, default=[7200, 7201])
    p.add_argument('--out', type=Path, default=EVIDENCE/'eval/diagnosis.jsonl')
    a = p.parse_args()
    torch.set_num_threads(1)
    traces = a.out.with_suffix(''); traces.mkdir(parents=True, exist_ok=True)
    files = [ROOT/'sim/rl/runs'/r/n for r in a.runs for n in ('model.zip', 'vecnorm.pkl', 'args.json')]
    files += [ROOT/'sim/rl/rev3_l80.json', ROOT/'sim/rl/realism_robstride_l80.json']
    digest = lambda: {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    before = digest()
    with a.out.open('x') as f:
        for run in a.runs:
            for speed in a.speeds:
                for yaw in a.yaws:
                    for seed in a.seeds:
                        row = rollout(run, speed, yaw, seed, traces)
                        f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
                        print(f'{run} v={speed} w={yaw} seed={seed}: {row["steady"]}', flush=True)
    after = digest()
    a.out.with_suffix('.manifest.json').write_text(json.dumps(dict(before=before, after=after, unchanged=before == after,
        randomize=False, seeds=a.seeds, speeds=a.speeds, yaws=a.yaws, runs=a.runs), indent=2)+'\n')
    assert before == after


if __name__ == '__main__':
    main()
