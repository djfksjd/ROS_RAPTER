"""Build the R-02 MuJoCo model from the design calculator (modeling/design_r02.py).

Same joint/actuator names as the 12-DOF model, so sim/rl works unchanged (pass the nominal pose with
--crouch HIP KNEE ANKLE). Masses and lengths come from design_r02.Params; nothing is tuned here.

- Pelvis-mounted actuators: hip roll actuator in the torso; hip pitch, knee and ankle pitch actuators on
  the hip-roll link at the hip (they roll with the leg); ankle roll actuator also at the hip (cable).
- Legs: femur, tibia, metatarsus as light tubes (+ transmission mass along the femur).
- Foot at the MTP: toe frame mounted at the nominal metatarsus inclination so the toes lie flat in the
  nominal pose; digit III (main) and IV (24 deg out) with passive MTP/interphalangeal springs, digit II a
  raised sickle claw, a small MTP bumper pad.
- Head/neck mass forward, tail with a concentrated tip mass.
- Velocity servos like the existing model (ctrl = joint speed target, ctrlrange = speed limit).
- Optional ankle 'Achilles' parallel spring (--achilles K PRELOAD).

Usage: .venv-sim/bin/python modeling/build_r02.py [--speed 3.0] [--out sim/raptor_r02.xml]
"""
import argparse
from math import cos, radians, sin
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_r02 as D  # noqa: E402

GREY, DARK, LIGHT = [.42, .44, .47, 1], [.12, .13, .15, 1], [.6, .62, .65, 1]


def rod_inertia(m, length, axis='z', r=.02):
    """Solid rod along an axis: (Ixx, Iyy, Izz)."""
    t, a = m*length**2/12 + m*r*r/4, m*r*r/2
    return {'x': [a, t, t], 'y': [t, a, t], 'z': [t, t, a]}[axis]


def body(parent, name, pos, mass, ipos, inertia, quat=None):
    b = parent.add_body(name=name, pos=list(pos))
    if quat is not None:
        b.quat = list(quat)
    b.mass, b.ipos, b.inertia, b.explicitinertial = mass, list(ipos), list(inertia), True
    return b


def box(b, size, pos, rgba, contact=True, **kw):
    g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(pos), rgba=rgba, **kw)
    g.contype, g.conaffinity, g.mass = 1, 0, 0  # like the existing model: bodies collide with the floor only
    if not contact:
        g.contype = 0
    return g


def hinge(b, name, axis, rng, torque, **kw):
    j = b.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), range=list(rng), **kw)
    if torque:
        j.actfrcrange, j.actfrclimited = [-torque, torque], True
    return j


def yq(deg):
    a = radians(deg)/2
    return [cos(a), 0, 0, sin(a)]


def pq(rad):
    return [cos(rad/2), 0, sin(rad/2), 0]


def toe(foot, side, n, pos, quat, length, k_prox, k_dist, rng=(-.6, .9), tip_friction=None):
    m = .01
    for part, k in (('proximal', k_prox), ('distal', k_dist)):
        L = length/2
        b = body(foot, f'{side}_toe_{n}_{part}_link', pos, m, [L/2, 0, 0], rod_inertia(m, L, 'x', .01), quat)
        hinge(b, f'{side}_toe_{n}_{part}_joint', [0, 1, 0], rng, 0, stiffness=k, damping=.05, armature=1e-4,
              frictionloss=.005)
        g = box(b, [L/2, .012, .012], [L/2, 0, 0], DARK, priority=1, friction=[.8, .005, .0001])
        if part == 'distal' and tip_friction:
            g.friction = [tip_friction, .005, .0001]
        foot, pos, quat = b, [L, 0, 0], None


def build(p, speed, achilles_k=0., achilles_preload=0.):
    st = D.stand(p)
    hip0, knee0 = radians(-st['hip_pitch_deg']), radians(st['knee_flex_deg'])  # MuJoCo: -pitch = leg forward
    meta_abs = hip0                              # pantograph: metatarsus parallel to the femur
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep, spec.option.integrator, spec.option.cone = .001, mujoco.mjtIntegrator.mjINT_IMPLICITFAST, mujoco.mjtCone.mjCONE_ELLIPTIC
    w = spec.worldbody
    w.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .05])
    root = w.add_body(name='base_root')
    root.add_freejoint(name='floating_base')
    torso_m = p.torso_frame + p.battery + p.compute_sensors + 2*p.act_big + p.act_small  # + hip roll and tail yaw actuators
    base = body(root, 'base_link', [0, 0, p.hip_stand + .03], torso_m, [0, 0, 0],
                [torso_m*(.16**2 + .12**2)/12, torso_m*(p.torso_len**2 + .12**2)/12, torso_m*(p.torso_len**2 + .16**2)/12])
    box(base, [p.torso_len/2, .08, .06], [0, 0, 0], GREY)
    base.add_site(name='imu')
    neck = body(base, 'head_link', [p.head_x, 0, .08], p.head_neck, [0, 0, 0], [p.head_neck*.004]*3)
    box(neck, [.09, .04, .04], [0, 0, 0], LIGHT)
    lm = p.tube_density
    for side in ('left', 'right'):
        s = 1. if side == 'left' else -1.
        hr = body(base, f'{side}_hip_roll_link', [0, s*.10, -.03], 3*p.act_big + p.act_small, [0, 0, 0],
                  [.004, .004, .004])
        hinge(hr, f'{side}_hip_roll_joint', [1, 0, 0], [-.5, .5], 36)
        box(hr, [.05, .04, .04], [0, 0, 0], DARK, contact=False)
        mt = lm*p.femur + p.transmission
        th = body(hr, f'{side}_thigh_link', [0, 0, 0], mt, [0, 0, -p.femur/2], rod_inertia(mt, p.femur))
        hinge(th, f'{side}_hip_pitch_joint', [0, 1, 0], [-1.6, .9], 36)
        box(th, [.02, .02, p.femur/2], [0, 0, -p.femur/2], LIGHT)
        ms = lm*p.tibia
        sh = body(th, f'{side}_shin_link', [0, 0, -p.femur], ms, [0, 0, -p.tibia/2], rod_inertia(ms, p.tibia))
        hinge(sh, f'{side}_knee_pitch_joint', [0, 1, 0], [.2, 2.6], 36)
        box(sh, [.018, .018, p.tibia/2], [0, 0, -p.tibia/2], LIGHT)
        mm = lm*p.meta
        me = body(sh, f'{side}_metatarsus_link', [0, 0, -p.tibia], mm, [0, 0, -p.meta/2], rod_inertia(mm, p.meta))
        ank = hinge(me, f'{side}_ankle_pitch_joint', [0, 1, 0], [-2.5, .2], 36)
        if achilles_k:  # parallel 'Achilles' spring, preloaded to carry achilles_preload N·m in the nominal pose
            ank.stiffness, ank.springref = [achilles_k, 0, 0], -knee0 + achilles_preload/achilles_k
        box(me, [.015, .015, p.meta/2], [0, 0, -p.meta/2], LIGHT)
        # foot frame at the MTP, pitched so that its x axis is horizontal forward in the nominal pose
        ft = body(me, f'{side}_foot_link', [0, 0, -p.meta], .04, [0, 0, 0], [2e-5]*3, pq(-meta_abs))
        hinge(ft, f'{side}_ankle_roll_joint', [1, 0, 0], [-.5, .5], 17, armature=.01)
        pad = ft.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.018, 0, 0], pos=[-.005, 0, -.002], rgba=DARK)
        pad.contype, pad.conaffinity, pad.mass = 1, 0, 0
        lat, med = ('1', '3') if side == 'left' else ('3', '1')
        toe(ft, side, '2', [0, -s*.006, -.008], yq(0), p.toe3, 30., 20., tip_friction=1.2)   # digit III
        toe(ft, side, lat, [0, s*.012, -.008], yq(s*24), p.toe4, 20., 14.)                   # digit IV
        toe(ft, side, med, [-.01, -s*.02, .01], pq(-.7), .06, 15., 10., rng=(-.3, .3))       # digit II claw
    # tail: yaw link at the torso rear, pitch link carrying the tail tube and the tip mass
    tb = -p.torso_len/2
    ty = body(base, 'tail_yaw_link', [tb, 0, 0], p.act_small, [0, 0, 0], [5e-4]*3)
    hinge(ty, 'tail_yaw_joint', [0, 0, 1], [-.8, .8], 36)
    box(ty, [.04, .03, .03], [-.02, 0, 0], DARK, contact=False)
    mtail = p.tail_struct + p.tail_tip
    cx = -(p.tail_struct*p.tail_len/2 + p.tail_tip*p.tail_len)/mtail
    Iyy = p.tail_struct*p.tail_len**2/12 + p.tail_struct*(p.tail_len/2 + cx)**2 + p.tail_tip*(p.tail_len + cx)**2
    tl = body(ty, 'tail_link', [-.04, 0, 0], mtail, [cx, 0, 0], [1e-3, Iyy, Iyy])
    hinge(tl, 'tail_pitch_joint', [0, 1, 0], [-.6, .6], 36)
    box(tl, [p.tail_len/2, .02, .02], [-p.tail_len/2, 0, 0], GREY)
    box(tl, [.04, .03, .03], [-p.tail_len, 0, 0], DARK)
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE and 'toe' not in j.name:
            a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.ctrlrange, a.ctrllimited = [-speed, speed], True
            a.forcerange, a.forcelimited = list(j.actfrcrange), True
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.gainprm = [100.] + [0.]*9   # kv; sim/rl overrides it (servo kv)
            a.biasprm = [0., 0., -100.] + [0.]*7
    spec.add_sensor(name='imu_quat', type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
    spec.add_sensor(name='imu_gyro', type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
    return spec, (hip0, knee0, -knee0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=3., help='joint speed limit rad/s (current spec 2-3; raised in step 3)')
    ap.add_argument('--achilles', type=float, nargs=2, metavar=('K', 'PRELOAD'), default=(0., 0.),
                    help='ankle parallel spring N·m/rad and nominal-pose preload N·m (DESIGN_R02 4.3: 77, 6)')
    ap.add_argument('--out', default='sim/raptor_r02.xml')
    a = ap.parse_args()
    spec, pose = build(D.Params(), a.speed, *a.achilles)
    m = spec.compile()
    Path(a.out).write_text(f'<!-- Generated by modeling/build_r02.py from modeling/design_r02.py. Do not edit. -->\n' + spec.to_xml())
    print(f'wrote {a.out}: mass {sum(m.body_mass):.2f} kg, nominal pose (hip, knee, ankle) = {np.round(pose, 3).tolist()}')


if __name__ == '__main__':
    main()
