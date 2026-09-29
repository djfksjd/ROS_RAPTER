#!/usr/bin/env python3
"""Offline FK bound: can the COM sit laterally over a flat stance foot within joint limits?

Read-only kinematic check, not a stability certificate. The stance (left) foot frame is
treated as level ground. Support is bounded generously by the full y-extent of all foot and
passive-toe collision boxes at their default pose; collisions, swing clearance, toe
deflection and dynamics are ignored, so a negative result is conservative (optimistic support).
"""
import argparse
import itertools
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from support_model import origin, rotation

ACTIVE = [f'{s}_{j}_joint' for s in ('left', 'right')
          for j in ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')] + ['tail_yaw_joint', 'tail_pitch_joint']
CORNERS = [np.array(c) for c in itertools.product([-1, 1], repeat=3)]


class Model:
    def __init__(self, urdf):
        root = ET.fromstring(urdf)
        joints = root.findall('joint')
        self.limits = {j.get('name'): (float(j.find('limit').get('lower')), float(j.find('limit').get('upper')))
                       for j in joints if j.get('type') == 'revolute'}
        self.chain = [(j.get('name'), j.find('parent').get('link'), j.find('child').get('link'),
                       origin(j.find('origin')), list(map(float, j.find('axis').get('xyz').split()))
                       if j.get('type') == 'revolute' else None) for j in joints]
        children = {c for _, _, c, _, _ in self.chain}
        self.root = next(p for _, p, _, _, _ in self.chain if p not in children)
        self.masses = [(l.get('name'), float(l.find('inertial/mass').get('value')), origin(l.find('inertial/origin'))[:, 3])
                       for l in root.findall('link') if l.find('inertial') is not None]
        self.total = sum(m for _, m, _ in self.masses)
        self.boxes = [(l.get('name'), origin(c.find('origin')),
                       np.array(list(map(float, c.find('geometry/box').get('size').split())))/2)
                      for l in root.findall('link') if l.get('name') == 'left_foot_link' or l.get('name').startswith('left_toe_')
                      for c in l.findall('collision') if c.find('geometry/box') is not None]  # R-02 pad is a sphere

    def fk(self, q):
        transforms, pending = {self.root: np.eye(4)}, list(self.chain)
        while pending:
            before = len(pending)
            for item in pending[:]:
                name, parent, child, joint_origin, axis = item
                if parent in transforms:
                    value = transforms[parent] @ joint_origin
                    transforms[child] = value @ rotation(axis, q.get(name, 0.)) if axis else value
                    pending.remove(item)
            if len(pending) == before:
                raise ValueError('Disconnected URDF tree')
        return transforms

    def evaluate(self, q):
        """Lateral margin (m, positive = COM inside) and base tilt relative to the flat stance foot."""
        t = self.fk(q)
        inverse = np.linalg.inv(t['left_foot_link'])
        com = sum(m*(t[n] @ o)[:3] for n, m, o in self.masses)/self.total
        y = (inverse @ np.r_[com, 1])[1]
        points = np.array([(inverse @ t[n] @ o @ np.r_[h*c, 1])[:3] for n, o, h in self.boxes for c in CORNERS])
        return (min(y-points[:, 1].min(), points[:, 1].max()-y), float(y),
                [float(points[:, 1].min()), float(points[:, 1].max())], math.acos(max(-1., min(1., inverse[2, 2]))))


def search(model, samples, seed):
    rng = np.random.default_rng(seed)
    draw = lambda: {k: rng.uniform(*model.limits[k]) for k in ACTIVE}
    best = max((draw() for _ in range(samples)), key=lambda q: model.evaluate(q)[0])
    results = []
    for start in [best] + [draw() for _ in range(2)]:
        q, current, step = dict(start), model.evaluate(start)[0], .2
        while step > 1e-3:
            improved = False
            for k in ACTIVE:
                for delta in (step, -step):
                    trial = q | {k: min(max(q[k]+delta, model.limits[k][0]), model.limits[k][1])}
                    value = model.evaluate(trial)[0]
                    if value > current+1e-9:
                        q, current, improved = trial, value, True
            if not improved:
                step /= 2
        margin, com_y, support_y, tilt = model.evaluate(q)
        results.append({'margin_m': margin, 'com_y_m': com_y, 'support_y_m': support_y, 'base_tilt_rad': tilt,
                        'joints': {k: round(v, 4) for k, v in q.items()}})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('urdf')
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    model = Model(open(args.urdf).read())
    upright = model.evaluate({'left_hip_pitch_joint': -.1, 'left_knee_pitch_joint': .4, 'left_ankle_pitch_joint': -.3})
    print(json.dumps({'limitation': 'Kinematic bound only: optimistic support, no collision/clearance/dynamics.',
                      'total_mass_kg': model.total, 'limits': {k: model.limits[k] for k in ACTIVE},
                      'upright_crouch_margin_m': upright[0], 'samples': args.samples, 'seed': args.seed,
                      'hill_climb': search(model, args.samples, args.seed)}, indent=2))
