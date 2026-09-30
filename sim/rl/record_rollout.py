"""Record a deterministic RunEnv (T1) policy rollout as per-link world poses for rendering with the visual GLB meshes.

The MuJoCo body frames of the R-02 model are the URDF link frames, and the GLBs in
src/raptor_description/meshes/r02/ are authored in those link frames (modeling/morphloom/split_links.py), so the
recorded xpos/xquat of each body can be applied to its GLB directly. Nothing is re-animated by hand.
Output (.npz): names, t, pos (frames x links x 3), quat (w x y z), plus the model/policy sha256 and the run settings
so a video can be traced to the exact policy. Render with modeling/render_rollout.py.
Usage: .venv-sim/bin/python sim/rl/record_rollout.py sim/rl/runs/<name>/model.zip out.npz --cmd 6 [--ankle-clutch]
       [--turn RATE START DUR] [--impulse AXIS NMS AT] [--fps 25]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load  # noqa: E402
from run_env import RunEnv  # noqa: E402

MESHES = HERE.parents[1]/'src/raptor_description/meshes/r02'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('out')
    p.add_argument('--cmd', type=float, default=6.)
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--fps', type=float, default=25.)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--ankle-clutch', action='store_true')
    p.add_argument('--couple-ankle', action='store_true')
    p.add_argument('--tail', choices=['active', 'locked'], default='active')
    p.add_argument('--turn', type=float, nargs=3, metavar=('RATE', 'START', 'DUR'))
    p.add_argument('--impulse', nargs=3, metavar=('AXIS', 'NMS', 'AT'))
    a = p.parse_args()
    env = RunEnv(randomize=False, seed=a.seed, ankle_clutch=a.ankle_clutch, couple_ankle=a.couple_ankle)
    model, venv = load(a.model, env)
    obs = venv.reset()
    env.command = np.array([a.cmd, 0., 0.]); env.resample_steps = 0
    env.lock_tail = a.tail == 'locked'
    obs = venv.normalize_obs(env._obs()[None])
    m, d = env.model, env.data
    names = [g.stem for g in sorted(MESHES.glob('*.glb')) if m.body(g.stem) is not None]
    ids = [m.body(n).id for n in names]
    every = max(1, int(round(1/(a.fps*.02))))
    t, pos, quat, fell, applied = [], [], [], False, False
    for k in range(int(a.seconds/.02)):
        if a.turn:
            env.command[2] = a.turn[0] if a.turn[1] <= d.time < a.turn[1]+a.turn[2] else 0.
        if a.impulse and not applied and d.time >= float(a.impulse[2]):
            env.apply_impulse(a.impulse[0], float(a.impulse[1])); applied = True
        act, _ = model.predict(obs, deterministic=True)
        raw, _, term, _, info = env.step(act[0])
        obs = venv.normalize_obs(raw[None])
        if k % every == 0:
            t.append(d.time); pos.append(d.xpos[ids].copy()); quat.append(d.xquat[ids].copy())
        if term:
            fell = True
            break
    meta = dict(policy=str(a.model), policy_sha=sha(a.model), vecnorm_sha=sha(Path(a.model).with_name('vecnorm.pkl')),
                model_xml=env.model_path, model_sha=sha(env.model_path), cmd=a.cmd, seed=a.seed, clutch=a.ankle_clutch,
                couple=a.couple_ankle, tail=a.tail, turn=a.turn, impulse=a.impulse, fell=fell, fps=a.fps,
                tier='T1 virtual actuator, MuJoCo, not the real robot')
    np.savez_compressed(a.out, names=np.array(names), t=np.array(t), pos=np.array(pos), quat=np.array(quat),
                        meta=json.dumps(meta))
    print(json.dumps({**meta, 'frames': len(t), 'links': len(names)}))


if __name__ == '__main__':
    main()
