"""Foot variants of the MuJoCo Raptor (toes and foot pad; masses and joint count unchanged).

didactyl: ostrich/dromaeosaur-like functional two-toe foot (research 2026-09-29, Schaller 2011 JEB,
Xing 2024 iScience): digit III forward, longer and stiffer with a high-friction claw tip (main load);
digit IV turned 24 deg outward, shorter and softer (lateral outrigger); digit II (medial) raised as a
sickle claw that only touches when the foot rolls inward. The flat pad stays (the dromaeosaur
metatarsophalangeal pad touches the ground).
ostrich: didactyl toes plus a smaller pad raised 15 mm (the ostrich MTP joint is permanently elevated),
so the toes carry the load; the pad only touches as a bumper. With the flat pad, the pad carried
85-89% of the foot load while walking (measured 2026-09-29).
A variant is a simulation hypothesis, not an adopted design.

Usage: .venv-sim/bin/python sim/foot_variant.py sim/raptor_digitigrade_ankleroll_ankle10.xml OUT.xml --variant ostrich
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np


def yaw_quat(deg):
    a = np.radians(deg)/2
    return [np.cos(a), 0, 0, np.sin(a)]


def pitch_quat(rad):
    return [np.cos(rad/2), 0, np.sin(rad/2), 0]


def scale_segment(body, length_scale):
    """Scale a toe segment's length along x (geom, inertial offset, child offset)."""
    for g in body.geoms:
        g.size = [g.size[0]*length_scale, g.size[1], g.size[2]]
        g.pos = [g.pos[0]*length_scale, g.pos[1], g.pos[2]]
    body.ipos = [body.ipos[0]*length_scale, body.ipos[1], body.ipos[2]]
    for child in body.bodies:
        child.pos = [child.pos[0]*length_scale, child.pos[1], child.pos[2]]


def set_toe(spec, side, toe, pos, quat, length_scale=1., stiffness_scale=1., tip_friction=None, joint_range=None):
    for part in ('proximal', 'distal'):
        body = spec.body(f'{side}_toe_{toe}_{part}_link')
        if part == 'proximal':
            body.pos, body.quat = pos, quat
        scale_segment(body, length_scale)
        joint = spec.joint(f'{side}_toe_{toe}_{part}_joint')
        joint.stiffness = joint.stiffness*stiffness_scale
        if joint_range is not None:
            joint.range = joint_range
        if part == 'distal' and tip_friction is not None:
            for g in body.geoms:
                g.friction = [tip_friction, g.friction[1], g.friction[2]]


def didactyl(spec):
    for side in ('left', 'right'):
        s = 1. if side == 'left' else -1.  # +y is lateral for the left foot, medial for the right
        lateral, medial = ('1', '3') if side == 'left' else ('3', '1')
        # digit III: main, forward, slightly medial of centre, longer and stiffer, claw tip friction
        set_toe(spec, side, '2', [.03, -s*.008, -.008], yaw_quat(0), length_scale=1.1, stiffness_scale=1.3, tip_friction=1.2)
        # digit IV: lateral outrigger, 24 deg outward, shorter and softer
        set_toe(spec, side, lateral, [.025, s*.03, -.008], yaw_quat(s*24), length_scale=.85, stiffness_scale=.7)
        # digit II: medial sickle claw, raised 0.6 rad; ground contact only when the foot rolls inward
        set_toe(spec, side, medial, [.015, -s*.035, .01], pitch_quat(-.6), length_scale=.8, stiffness_scale=1.,
                joint_range=[-.3, .3])
    return spec


def ostrich(spec):
    didactyl(spec)
    for side in ('left', 'right'):
        pad = spec.body(f'{side}_foot_link').geoms[0]
        pad.size = [pad.size[0]*.6, pad.size[1]*.7, pad.size[2]]
        pad.pos = [pad.pos[0]-.01, pad.pos[1], pad.pos[2]+.015]
    return spec


VARIANTS = {'didactyl': didactyl, 'ostrich': ostrich}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base')
    p.add_argument('out')
    p.add_argument('--variant', choices=list(VARIANTS), default='didactyl')
    a = p.parse_args()
    spec = VARIANTS[a.variant](mujoco.MjSpec.from_file(a.base))
    spec.compile()
    Path(a.out).write_text(spec.to_xml())
    print('wrote', a.out)


if __name__ == '__main__':
    main()
