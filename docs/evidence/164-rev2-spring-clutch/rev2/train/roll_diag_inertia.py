import sys, json
import numpy as np, mujoco
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
RUN = ROOT/'sim/rl/runs'/sys.argv[1]
def go(cmd, seed, size=2.0):
    env = RunEnv(**{**env_kwargs(json.loads((RUN/'args.json').read_text())), 'episode_s': 15., 'level': 0.}, randomize=False, seed=7); env.init_seed = seed
    model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset(); d = env.data; m = env.model
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; obs = venv.normalize_obs(env._obs()[None])
    rng = np.random.default_rng(seed); at = int((5.+rng.uniform(0., .5))/.02); sign = rng.choice([-1., 1.])
    J = [m.joint(f'{s}_hip_roll_joint') for s in ('left', 'right')]; A = [m.actuator(f'{s}_hip_roll_joint').id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{s}_hip_roll_joint') >= 0 else None for s in ('left', 'right')]
    print('hip roll range', J[0].range, 'actuators', [m.actuator(i).name for i in range(m.nu)])
    for k in range(at+9):
        if k == at:
            # whole-body roll inertia about the COM, body x axis
            mujoco.mj_subtreeVel(m, d); com = d.subtree_com[env.base]; R = d.xmat[env.base].reshape(3, 3); x = R[:, 0]; I = 0.
            for b in range(1, m.nbody):
                r = d.xipos[b]-com; Rb = d.ximat[b].reshape(3, 3); Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
                I += x @ Ib @ x + m.body_mass[b]*(r @ r-(r @ x)**2)
            print('whole-body roll inertia about COM', round(I, 3), 'kg m2 -> rigid roll rate', round(size/I, 2), 'rad/s; trunk body inertia', m.body_inertia[env.base])
            env.apply_impulse('r', sign*size)
        raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
        if k >= at-1:
            q = [d.qpos[j.qposadr[0]] for j in J]; qd = [d.qvel[j.dofadr[0]] for j in J]; tq = [d.qfrc_actuator[j.dofadr[0]] for j in J]
            print(f"  +{(k-at)*.02:+.2f}s tilt {info['tilt']:.2f} hipR q {q[0]:+.2f} {q[1]:+.2f} qd {qd[0]:+.1f} {qd[1]:+.1f} torque {tq[0]:+.1f} {tq[1]:+.1f} loaded {[bool(v) for v in env.loaded.values()]}")
        if term: break
go(4., 96009)
