"""Read-only 1 ms contact-moment and angular-momentum audit for a saved policy."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from contact_wrench import contact_wrench
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs


def momentum(m, d, root):
    # mj_step leaves position-derived fields at the pre-integration pose.
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    mujoco.mj_subtreeVel(m, d)
    return d.subtree_angmom[root].copy()


def probe(run, yaw, seed, seconds=12., instrument=True, yaw_scale=None, turn_seconds=6.):
    args = json.loads((Path(run)/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(args), 'level': 0., 'episode_s': seconds+5.}, randomize=False, seed=7)
    env.init_seed = seed
    model, venv = load(str(Path(run)/'model.zip'), env)
    physics = []
    original = mujoco.mj_step
    scratch = None

    def state_momentum(m, d):
        # Recompute pose-derived values only on a copy. The saved policy's
        # observation timing must remain identical to uninstrumented RunEnv.
        mujoco.mj_copyData(scratch, m, d)
        return momentum(m, scratch, env.root)

    def measured_step(m, d):
        before = state_momentum(m, d)
        time = float(d.time)
        original(m, d)
        force, moment, contacts = contact_wrench(m, d, env.root)
        after = state_momentum(m, d)
        physics.append(dict(t=time, command_yaw=float(env.command[2]),
                            contact_moment_z=float(moment[2]),
                            contact_force=force.tolist(), contacts=contacts,
                            delta_Lz=float(after[2]-before[2]), Lz=float(after[2])))
    try:
        venv.reset()
        scratch = mujoco.MjData(env.model)
        if yaw_scale is not None:
            from symmetric_rollout import RawPolicyAdapter
            model=RawPolicyAdapter(model,venv,env,yaw_scale)
        env.command = np.array([4., 0., 0.]); env.resample_steps = 0
        if instrument:
            mujoco.mj_step = measured_step
        control = []
        for _ in range(int(seconds/.02)):
            env.command[2] = yaw if 4. <= round(float(env.data.time), 12) < 4.+turn_seconds else 0.
            obs = venv.normalize_obs(env._obs()[None])
            action, _ = model.predict(obs, deterministic=True)
            _, _, term, _, info = env.step(action[0])
            control.append(dict(t=float(env.data.time), yaw_rate=float(env.body_velocity(env.base)[0][2]),
                                vx=float(info['v_body'][0]), tilt=float(info['tilt']),phase=float(env.phase),
                                action=action[0].tolist(),loaded={k:bool(v) for k,v in info['loaded'].items()},
                                grf={k:float(v) for k,v in info['grf'].items()}))
            if term:
                break
        turn = [r for r in physics if 4. <= r['t'] < 4.+turn_seconds]
        dt = env.model.opt.timestep
        impulse = sum(r['contact_moment_z']*dt for r in turn)
        delta = sum(r['delta_Lz'] for r in turn)
        return dict(run=run, yaw=yaw, seed=seed, instrumented=instrument, fell=bool(term),yaw_scale=yaw_scale,turn_seconds=turn_seconds,
                    contact_yaw_impulse_Nms=impulse, delta_Lz_Nms=delta,
                    momentum_balance_residual_Nms=delta-impulse,
                    control=control, physics=physics, final_qpos=env.data.qpos.tolist(),
                    final_qvel=env.data.qvel.tolist())
    finally:
        mujoco.mj_step = original
        venv.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--out', required=True)
    p.add_argument('--yaw', type=float, nargs='+', default=[-1., 0., 1.])
    p.add_argument('--seeds', type=int, nargs='+', default=[9001, 9002])
    p.add_argument('--uninstrumented', action='store_true')
    p.add_argument('--yaw-scale',type=float)
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--turn-seconds',type=float,default=6.)
    a = p.parse_args()
    if a.turn_seconds<=0 or a.seconds<4.+a.turn_seconds:
        p.error("positive turn interval must be contained in trial")
    if a.yaw_scale is not None and (not np.isfinite(a.yaw_scale) or a.yaw_scale<=0):
        p.error("yaw scale must be finite and positive")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        p.error(f'refusing to overwrite {out}')
    with out.open('x') as f:
        for yaw in a.yaw:
            for seed in a.seeds:
                row = probe(a.run, yaw, seed,seconds=a.seconds,instrument=not a.uninstrumented,
                            yaw_scale=a.yaw_scale,turn_seconds=a.turn_seconds)
                f.write(json.dumps(row)+'\n'); f.flush()
                print(json.dumps({k:v for k,v in row.items() if k not in ('control','physics','final_qpos','final_qvel')}), flush=True)


if __name__ == '__main__':
    main()
