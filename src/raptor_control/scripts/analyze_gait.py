#!/usr/bin/env python3
"""Read-only FK analysis of recorded gait snapshots; not a stability certificate."""
import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from support_model import origin, rotation



def support_margin(points, xy):
    """Signed distance to a convex hull; negative means outside, metres."""
    points = sorted(set((float(p[0]),float(p[1])) for p in points))
    if len(points)<3: return None
    def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def chain(seq):
        hull=[]
        for p in seq:
            while len(hull)>=2 and cross(hull[-2],hull[-1],p)<=0: hull.pop()
            hull.append(p)
        return hull
    hull=chain(points)[:-1]+chain(list(reversed(points)))[:-1]
    if len(hull)<3: return None
    return min(cross(a,b,xy)/np.linalg.norm(np.array(b)-np.array(a))
               for a,b in zip(hull,hull[1:]+hull[:1]))


def analyze(urdf, rows):
    root = ET.fromstring(urdf)
    result = []
    for row in rows:
        if 'all_joint_positions' not in row:
            continue
        state = row['all_joint_positions']
        q = row['pose']['orientation']
        x, y, z, w = [q.get(k, 0.) for k in ['x', 'y', 'z', 'w']]
        base = np.eye(4)
        base[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        base[:3, 3] = [row['pose']['position'].get(k, 0.) for k in ['x', 'y', 'z']]
        transforms = {'base_root': base}
        pending = list(root.findall('joint'))
        while pending:
            previous = len(pending)
            for joint in pending[:]:
                parent = joint.find('parent').get('link')
                if parent not in transforms:
                    continue
                value = transforms[parent] @ origin(joint.find('origin'))
                if joint.get('type') != 'fixed':
                    name = joint.get('name')
                    if joint.get('type') != 'revolute' or name not in state:
                        raise ValueError('Missing or unsupported joint: '+name)
                    if state[name]['age_wall_s'] > .5:
                        raise ValueError('Stale joint: '+name)
                    value = value @ rotation(list(map(float, joint.find('axis').get('xyz').split())),
                                             state[name]['position'])
                transforms[joint.find('child').get('link')] = value
                pending.remove(joint)
            if len(pending) == previous:
                raise ValueError('Disconnected URDF tree')
        weighted = np.zeros(3)
        mass = 0.
        feet = {'left': [], 'right': []}
        for link in root.findall('link'):
            name = link.get('name')
            inertial = link.find('inertial')
            if inertial is not None:
                m = float(inertial.find('mass').get('value'))
                weighted += m*(transforms[name] @ origin(inertial.find('origin')))[:3, 3]
                mass += m
            for side in feet:
                if name == side+'_foot_link' or name.startswith(side+'_toe_'):
                    for collision in link.findall('collision'):
                        box = collision.find('geometry/box')
                        if box is None:
                            raise ValueError('Unsupported foot collision')
                        half = np.array(list(map(float, box.get('size').split())))/2
                        t = transforms[name] @ origin(collision.find('origin'))
                        feet[side] += [(t @ np.r_[half*np.array(sign), 1.])[:3]
                                       for sign in itertools.product([-1, 1], repeat=3)]
        item = {'phase': row['phase'], 'com_world': (weighted/mass).tolist(),
                'foot_collision_min_z': {s: float(min(p[2] for p in points)) for s, points in feet.items()},
                'contact_y_bounds': {}, 'contact_hull_margin_xy': {}}
        for side, contact in row['contacts'].items():
            points = [p for key, v in contact['samples'].items()
                      if key in contact['active_sources'] for p in v['points']]
            item['contact_hull_margin_xy'][side] = support_margin(points,item['com_world'][:2])
            item['contact_y_bounds'][side] = [min(p[1] for p in points), max(p[1] for p in points)] if points else None
        result.append(item)
    return {'limitation': 'Pose and ROS snapshots are not synchronized; flat z=0 ground only. '
            'Contact y bounds are not a support polygon or dynamic stability proof.', 'snapshots': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('urdf', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.urdf.read_text(), json.loads(args.report.read_text())), indent=2))
