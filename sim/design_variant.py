"""Design variants of the MuJoCo Raptor, one parameter at a time, with the same stability metrics.

Edits the compiled-from-URDF model through MjSpec (no URDF change): move mass from the body to a
forward head, scale leg segment lengths, change stance width, scale total mass, scale joint velocity
limits. A variant is a simulation hypothesis; only adopted changes go to the Xacro later.

Metrics
- com_z, rear/front margin: CoM height and distance from the CoM to the rear/front end of the
  contact patch after 2 s of holding the default crouch (servo hold, no controller).
- alpha: rocking angle atan(outer pad edge y / com_z) (evidence 74 model; larger = harder to tip sideways).
- noise_survival_s: mean time before a fall with random position-target noise (std 0.1, 50 Hz)
  through the Gazebo-like servo, 8 episodes capped at 10 s. A proxy for how forgiving the body is
  while a policy explores, not a walking result.

Usage: .venv-sim/bin/python sim/design_variant.py --out docs/evidence/76-design/sweep.json
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'rl'))
BASE_XML = HERE/'raptor_digitigrade.xml'
LEG_BODIES = ('thigh_link', 'shin_link', 'foot_link')


def build(head_mass=0., head_x=.45, leg_scale=1., half_width=None, mass_scale=1., vel_scale=1.):
    spec = mujoco.MjSpec.from_file(str(BASE_XML))
    base = spec.body('base_link')
    if head_mass > 0:  # relocate mass (battery/computer) into a forward neck/head, total unchanged
        base.mass -= head_mass
        head = base.add_body(name='head_link', pos=[head_x, 0, .05], mass=head_mass,
                             inertia=[head_mass*.004]*3, explicitinertial=True)
        head.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.09, .05, .05], conaffinity=0, mass=0,
                      rgba=[.9, .9, .9, 1])
    for side in ('left', 'right'):
        hip = spec.body(f'{side}_hip_roll_link')
        if half_width is not None:
            hip.pos = [hip.pos[0], half_width if side == 'left' else -half_width, hip.pos[2]]
        if leg_scale != 1.:
            for name in LEG_BODIES:
                body = spec.body(f'{side}_{name}')
                if name != 'thigh_link':
                    body.pos = [body.pos[0], body.pos[1], body.pos[2]*leg_scale]
                if name == 'foot_link':
                    continue  # keep the foot pad; its offset from the ankle is the shin/foot geometry
                for g in body.geoms:
                    g.pos = [g.pos[0], g.pos[1], g.pos[2]*leg_scale]
                    g.size = [g.size[0], g.size[1], g.size[2]*leg_scale]
    if mass_scale != 1.:
        for b in spec.bodies:
            if b.mass > 0:
                b.mass *= mass_scale
                b.inertia = [i*mass_scale for i in b.inertia]
    model = spec.compile()
    model.actuator_ctrlrange[:] *= vel_scale
    return spec, model


def metrics(params, episodes=8):
    import raptor_env as re
    spec, model = build(**params)
    with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False) as f:
        f.write(spec.to_xml())
        path = f.name
    env = re.RaptorEnv('flat', randomize=False, seed=0, model_path=path, vel_scale=params.get('vel_scale', 1.))
    env.reset()
    for _ in range(100):  # 2 s hold of the default crouch
        env.step(np.zeros(env.action_space.shape[0]))
    d, m = env.data, env.model
    com = d.subtree_com[env.base].copy()
    xs = [d.contact[i].pos[0] for i in range(d.ncon)]
    ys = [abs(d.contact[i].pos[1]) for i in range(d.ncon)]
    row = {**params, 'mass_kg': round(float(m.body_subtreemass[env.base]), 2), 'com_z': round(float(com[2]), 3),
           'rear_margin': round(float(com[0]-min(xs)), 3), 'front_margin': round(float(max(xs)-com[0]), 3),
           'alpha': round(float(np.arctan(max(ys)/com[2])), 3), 'standing_tilt': round(float(env.step(np.zeros(env.action_space.shape[0]))[4]['tilt']), 3)}
    rng = np.random.default_rng(0)
    times = []
    for ep in range(episodes):
        e = re.RaptorEnv('flat', randomize=False, seed=ep, model_path=path, vel_scale=params.get('vel_scale', 1.))
        e.reset()
        t = 0
        while t < 500:
            _, _, term, _, _ = e.step(np.clip(rng.normal(0, .1, e.action_space.shape[0]), -1, 1))
            t += 1
            if term:
                break
        times.append(t*.02)
    row['noise_survival_s'] = round(float(np.mean(times)), 2)
    Path(path).unlink()
    return row


SWEEP = [
    {},
    {'head_mass': 1.}, {'head_mass': 2.}, {'head_mass': 3.},
    {'leg_scale': .85}, {'leg_scale': .7},
    {'half_width': .14}, {'half_width': .10},
    {'mass_scale': 12/17.3},
    {'vel_scale': 4.},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out')
    p.add_argument('--variant', help='JSON dict of parameters; default runs the one-at-a-time sweep')
    a = p.parse_args()
    variants = [json.loads(a.variant)] if a.variant else SWEEP
    rows = []
    for v in variants:
        rows.append(metrics(v))
        print(json.dumps(rows[-1]), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1)+'\n')


if __name__ == '__main__':
    main()
