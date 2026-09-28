#!/usr/bin/env python3
"""Emit a Morphloom AssemblyIR (mm, Three.js Y-up) for the digitigrade Raptor visuals.

Every component id is `<urdf_link>__<part>` and is authored in that link's local ROS frame,
then placed with the URDF zero-pose transform, so the compiled GLB can be split back into
link-local visual meshes. Visual only: collision, inertia and joints stay in the Xacro.
Dimensions come from the generated URDF; styling follows the AI concept (inferred).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/raptor_control/scripts'))
from lateral_feasibility import Model  # noqa: E402

C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # ROS (x fwd, y left, z up) -> Three (x, y up, z)

MATERIALS = {
    'ivory': ('painted armour', {'color': '#d8d3c6', 'surface': 'coated-metal', 'roughness': .42, 'metalness': .15,
                                 'clearcoat': .35, 'clearcoatRoughness': .3}),
    'graphite': ('anodized frame', {'color': '#1d2126', 'surface': 'anodized-metal', 'roughness': .36, 'metalness': .85}),
    'titanium': ('brushed actuator', {'color': '#8d949b', 'surface': 'brushed-metal', 'roughness': .28, 'metalness': 1.,
                                      'anisotropy': .5}),
    'steel': ('polished pin', {'color': '#b8bdc2', 'surface': 'polished-metal', 'roughness': .14, 'metalness': 1.}),
    'copper': ('cable jacket', {'color': '#b4642a', 'surface': 'soft-touch-polymer', 'roughness': .55, 'metalness': 0.}),
    'accent': ('orange ring', {'color': '#c8662a', 'surface': 'anodized-metal', 'roughness': .35, 'metalness': .7}),
    'rubber': ('sole rubber', {'color': '#0f1012', 'surface': 'rubber', 'roughness': .85, 'metalness': 0.}),
    'claw': ('claw polymer', {'color': '#2a2d31', 'surface': 'molded-polymer', 'roughness': .3, 'metalness': .1,
                              'clearcoat': .6}),
    'lens': ('camera glass', {'color': '#0b1a24', 'surface': 'optical-glass', 'roughness': .04, 'metalness': 0.,
                              'transmission': .25, 'ior': 1.5}),
}


def rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)+s*k+(1-c)*k@k


def rpy(r, p, y):
    return rot((0, 0, 1), y)@rot((0, 1, 0), p)@rot((1, 0, 0), r)


def euler_xyz(m):
    """Three.js Euler 'XYZ' from a rotation matrix."""
    ey = math.asin(max(-1., min(1., m[0, 2])))
    if abs(m[0, 2]) < .9999999:
        return [math.atan2(-m[1, 2], m[2, 2]), ey, math.atan2(-m[0, 1], m[0, 0])]
    return [math.atan2(m[2, 1], m[1, 1]), ey, 0.]


# Primitive axes: a Three cylinder runs along Three y (= ROS z); a Three torus's axis is Three z (= ROS -y).
CYL_AXIS = {'z': np.eye(3), 'x': rot((0, 1, 0), math.pi/2), 'y': rot((1, 0, 0), -math.pi/2)}
RING_AXIS = {'y': np.eye(3), 'x': rot((0, 0, 1), math.pi/2), 'z': rot((1, 0, 0), math.pi/2)}


class Builder:
    def __init__(self, urdf):
        self.model = Model(urdf)
        self.frames = self.model.fk({})
        self.components = []

    def add(self, link, name, geometry, xyz=(0, 0, 0), r_ros=np.eye(3), mat='graphite', detail=''):
        frame = self.frames[link]
        p_ros = frame[:3, :3]@np.array(xyz, float)+frame[:3, 3]
        r = C@(frame[:3, :3]@r_ros)@C.T
        label, material = MATERIALS[mat]
        self.components.append({
            'id': f'{link}__{name}', 'name': f'{link} {name}'.replace('_', ' ')[:120],
            'category': 'mechanical', 'materialName': label, 'detail': detail or f'visual part of {link}',
            'position': [round(float(v)*1000, 3) for v in C@p_ros],
            'rotation': [round(v, 6) for v in euler_xyz(r)],
            'geometry': geometry, 'material': dict(material),
            'evidence': {'status': 'inferred', 'source': 'styling from AI concept; envelope from Xacro'}})

    # ---- primitive helpers (sizes in metres, ROS link frame) ----
    def box(self, link, name, size, xyz, mat, radius=.006, r=np.eye(3)):
        sx, sy, sz = size  # ROS extents -> Three (x, z_ros, y_ros)
        self.add(link, name, {'op': 'roundedBox', 'size': [sx*1000, sz*1000, sy*1000],
                              'radius': min(radius, min(size)/2.2)*1000, 'segments': 3}, xyz, r, mat)

    def cyl(self, link, name, radius, length, xyz, axis, mat, radius_top=None, segments=40):
        """axis: 'x'/'y'/'z' (ROS) or a rotation matrix taking ROS z to the cylinder axis."""
        r = CYL_AXIS[axis] if isinstance(axis, str) else axis
        self.add(link, name, {'op': 'cylinder', 'radiusTop': (radius if radius_top is None else radius_top)*1000,
                              'radiusBottom': radius*1000, 'depth': length*1000, 'radialSegments': segments},
                 xyz, r, mat)

    def ring(self, link, name, radius, tube, xyz, axis, mat):
        self.add(link, name, {'op': 'torus', 'radius': radius*1000, 'tube': tube*1000,
                              'radialSegments': 12, 'tubularSegments': 48}, xyz, RING_AXIS[axis], mat)

    def plate(self, link, name, profile_xz, thickness, y, mat, bevel=.002):
        """Extruded panel: profile in the link's x-z plane, thickness along y (centred at y)."""
        self.add(link, name, {'op': 'extrude', 'points': [[x*1000, z*1000] for x, z in profile_xz],
                              'depth': thickness*1000, 'bevelSize': bevel*1000, 'bevelThickness': bevel*1000,
                              'bevelSegments': 2}, (0, y, 0), np.eye(3), mat)

    def tube(self, link, name, points, radius, mat):
        frame = self.frames[link]
        world = [C@(frame[:3, :3]@np.array(p, float)+frame[:3, 3]) for p in points]
        label, material = MATERIALS[mat]
        self.components.append({
            'id': f'{link}__{name}', 'name': f'{link} {name}'.replace('_', ' ')[:120], 'category': 'interconnect',
            'materialName': label, 'detail': f'cable/rod on {link}',
            'geometry': {'op': 'tube', 'points': [[round(float(v)*1000, 3) for v in w] for w in world],
                         'radius': radius*1000, 'radialSegments': 10},
            'material': dict(material),
            'evidence': {'status': 'inferred', 'source': 'styling from AI concept'}})


def build_base(b):
    L = 'base_link'
    b.box(L, 'core', (.50, .20, .15), (-.02, 0, 0), 'graphite', .02)
    for side, y in (('l', 1), ('r', -1)):
        b.plate(L, f'armor_{side}', [(-.22, -.06), (.14, -.085), (.26, -.035), (.27, .05), (.10, .085), (-.18, .075),
                                     (-.25, .02)], .014, y*.118, 'ivory', .003)
        b.plate(L, f'armor_trim_{side}', [(-.20, -.045), (.12, -.07), (.22, -.03), (.18, -.022), (.10, -.058),
                                          (-.19, -.035)], .004, y*.127, 'accent', .001)
        b.cyl(L, f'hip_bearing_{side}', .058, .045, (0, y*.145, -.06), 'y', 'titanium')
        b.ring(L, f'hip_bearing_ring_{side}', .05, .006, (0, y*.168, -.06), 'y', 'accent')
        b.tube(L, f'cable_{side}', [(.24, y*.10, -.02), (.12, y*.135, -.05), (.02, y*.14, -.10), (-.05, y*.12, -.12)],
               .007, 'copper')
        for i, x in enumerate((-.16, -.06, .04)):
            b.cyl(L, f'panel_bolt_{side}{i}', .006, .004, (x, y*.127, .045), 'y', 'steel', segments=16)
    # head module and twin-camera visor
    b.box(L, 'head', (.14, .19, .12), (.24, 0, .005), 'graphite', .018)
    b.plate(L, 'head_cowl', [(.17, .045), (.32, .03), (.33, .055), (.19, .075)], .20, 0, 'ivory', .004)
    b.box(L, 'snout', (.08, .16, .085), (.31, 0, -.012), 'graphite', .016)
    b.plate(L, 'snout_cowl', [(.27, .03), (.35, .012), (.352, .034), (.29, .05)], .165, 0, 'ivory', .004)
    b.box(L, 'visor', (.012, .14, .045), (.346, 0, -.012), 'graphite', .006)
    for side, y in (('l', .042), ('r', -.042)):
        b.cyl(L, f'lens_{side}', .019, .014, (.352, y, -.012), 'x', 'lens')
        b.ring(L, f'lens_bezel_{side}', .022, .004, (.356, y, -.012), 'x', 'titanium')
    b.box(L, 'chin', (.10, .12, .03), (.25, 0, -.07), 'ivory', .01)
    # spine, mast and camera pod
    b.box(L, 'spine_rail', (.44, .05, .022), (-.03, 0, .09), 'titanium', .008)
    for side, y in (('l', .06), ('r', -.06)):
        b.cyl(L, f'rail_bar_{side}', .006, .40, (-.02, y, .105), 'x', 'steel', segments=16)
    b.cyl(L, 'mast', .028, .09, (.04, 0, .145), 'z', 'graphite')
    b.cyl(L, 'mast_cap', .042, .025, (.04, 0, .2), 'z', 'titanium')
    b.ring(L, 'lidar_ring', .038, .006, (.04, 0, .215), 'z', 'accent')
    b.cyl(L, 'lidar_window', .034, .02, (.04, 0, .225), 'z', 'lens')
    b.box(L, 'camera_pod', (.10, .09, .08), (-.09, 0, .145), 'graphite', .012)
    b.cyl(L, 'camera_pod_lens', .022, .02, (-.035, 0, .15), 'x', 'lens')
    b.box(L, 'heat_sink', (.08, .07, .02), (-.18, 0, .12), 'titanium', .003)
    for i in range(5):
        b.box(L, f'fin_{i}', (.006, .07, .025), (-.21+i*.014, 0, .14), 'titanium', .001)
    b.box(L, 'belly_pod', (.22, .13, .05), (-.03, 0, -.11), 'graphite', .012)
    b.cyl(L, 'tail_flange', .065, .03, (-.285, 0, 0), 'x', 'titanium')
    b.ring(L, 'tail_flange_ring', .058, .006, (-.27, 0, 0), 'x', 'accent')


def build_leg(b, side):
    s = 1 if side == 'left' else -1
    H, T, S, F = (f'{side}_{n}_link' for n in ('hip_roll', 'thigh', 'shin', 'foot'))
    b.box(H, 'actuator', (.11, .09, .10), (0, 0, -.055), 'graphite', .015)
    b.plate(H, 'cover', [(-.05, -.02), (.05, -.02), (.055, -.09), (-.045, -.10)], .01, s*.05, 'ivory', .002)
    # thigh: large hip-pitch drive, armoured femur, piston
    b.cyl(T, 'hip_drive', .082, .085, (0, s*.01, 0), 'y', 'ivory')
    b.cyl(T, 'hip_drive_cap', .05, .075, (0, s*.012, 0), 'y', 'titanium')
    b.ring(T, 'hip_drive_ring', .068, .007, (0, s*.056, 0), 'y', 'accent')
    b.cyl(T, 'hip_drive_bolt', .014, .085, (0, s*.012, 0), 'y', 'steel', segments=20)
    b.plate(T, 'femur_armor', [(-.07, -.02), (.075, -.035), (.058, -.20), (.028, -.25), (-.035, -.245), (-.065, -.14)],
            .09, 0, 'ivory', .005)
    b.box(T, 'femur_strut', (.035, .085, .22), (-.02, 0, -.13), 'graphite', .01)
    b.cyl(T, 'femur_piston', .011, .17, (.05, s*.035, -.13), 'z', 'steel', segments=20)
    b.cyl(T, 'femur_piston_body', .017, .08, (.05, s*.035, -.08), 'z', 'graphite', segments=24)
    b.cyl(T, 'knee_drive', .052, .095, (0, 0, -.26), 'y', 'titanium')
    b.ring(T, 'knee_ring', .045, .005, (0, s*.049, -.26), 'y', 'accent')
    # shin: twin struts, calf cover, ankle drive
    for i, y in enumerate((.026, -.026)):
        b.box(S, f'strut_{i}', (.028, .022, .28), (-.012, y, -.15), 'graphite', .008)
    b.plate(S, 'calf_cover', [(-.05, -.03), (-.02, -.02), (-.015, -.23), (-.035, -.27), (-.06, -.20)], .07, 0,
            'ivory', .003)
    b.plate(S, 'shin_armor', [(.02, -.04), (.055, -.05), (.05, -.20), (.03, -.255), (.012, -.24)], .075, 0, 'ivory', .004)
    b.cyl(S, 'tibia_piston', .01, .22, (.03, 0, -.14), 'z', 'steel', segments=20)
    b.cyl(S, 'tibia_piston_body', .016, .09, (.03, 0, -.07), 'z', 'graphite', segments=24)
    b.tube(S, 'cable', [(-.055, s*.03, -.02), (-.06, s*.035, -.15), (-.045, s*.03, -.28)], .005, 'copper')
    b.cyl(S, 'ankle_drive', .044, .085, (0, 0, -.30), 'y', 'titanium')
    b.ring(S, 'ankle_ring', .038, .005, (0, s*.044, -.30), 'y', 'accent')
    # foot: inclined metatarsus + pad (matches digitigrade_foot.xacro collision)
    beta, length = .35, .24
    end = np.array([length*math.sin(beta), 0, -length*math.cos(beta)])
    tilt = rot((0, 1, 0), -beta)
    b.box(F, 'metatarsus', (.045, .05, length), end/2, 'graphite', .012, tilt)
    b.box(F, 'metatarsus_guard', (.018, .058, length*.8), end/2+np.array([.028, 0, .006]), 'ivory', .006, tilt)
    b.cyl(F, 'metatarsus_spring', .009, length*.7, end/2+np.array([-.03, 0, .01]), tilt, 'accent', segments=16)
    pad_x, pad_z = end[0]-.04, end[2]+.01-.0175
    b.box(F, 'pad', (.15, .10, .035), (pad_x, 0, pad_z), 'rubber', .01)
    b.box(F, 'pad_frame', (.13, .085, .02), (pad_x, 0, pad_z+.022), 'graphite', .006)
    b.cyl(F, 'ball_joint', .03, .07, (end[0], 0, end[2]+.02), 'y', 'titanium')
    b.cyl(F, 'sickle_claw', .012, .06, (pad_x-.06, s*.035, pad_z+.04), rot((0, 1, 0), -.9), 'claw', radius_top=.002)
    for d, y in ((1, .040), (2, 0), (3, -.040)):
        for part, length_ in (('proximal', .070), ('distal', .075)):
            link = f'{side}_toe_{d}_{part}_link'
            b.box(link, 'segment', (length_, .026, .026), (length_/2, 0, 0), 'graphite', .006)
            b.box(link, 'cap', (length_*.7, .022, .008), (length_*.5, 0, .016), 'ivory', .003)
            b.cyl(link, 'hinge', .014, .03, (0, 0, 0), 'y', 'titanium', segments=24)
            if part == 'distal':
                b.cyl(link, 'claw', .011, .05, (length_+.018, 0, -.008), rot((0, 1, 0), 2.0), 'claw', radius_top=.0015)


def build_tail(b):
    b.cyl('tail_yaw_link', 'housing', .058, .15, (-.075, 0, 0), 'x', 'ivory', radius_top=.05)
    b.ring('tail_yaw_link', 'ring', .055, .006, (-.012, 0, 0), 'x', 'accent')
    b.cyl('tail_yaw_link', 'pitch_drive', .036, .11, (-.155, 0, 0), 'y', 'titanium')
    L = 'tail_link'
    segments, span = 9, .66
    step = span/segments
    b.tube(L, 'spine', [(-.01, 0, 0), (-.25, 0, .015), (-.50, 0, .07), (-.66, 0, .13)], .012, 'graphite')
    for i in range(segments):
        u = i/(segments-1)
        x = -.03-step*i
        z = .004+.13*u**2
        w = .11*(1-.6*u)
        b.box(L, f'band_{i:02d}', (step*.84, w, w*.86), (x, 0, z), 'ivory', min(.018, w/3), rot((0, 1, 0), -.28*u))
        b.box(L, f'band_joint_{i:02d}', (step*.2, w*.86, w*.74), (x-step*.5, 0, z-.004*u), 'graphite', w/5, rot((0, 1, 0), -.28*u))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('urdf')
    parser.add_argument('output')
    args = parser.parse_args()
    b = Builder(Path(args.urdf).read_text())
    build_base(b)
    for side in ('left', 'right'):
        build_leg(b, side)
    build_tail(b)
    links = sorted({c['id'].split('__')[0] for c in b.components})
    assembly = {'schema': 'morphloom.assembly/0.1', 'name': 'raptor-digitigrade-visual', 'units': 'mm',
                'components': b.components}
    Path(args.output).write_text(json.dumps(assembly, indent=1))
    print(f'{len(b.components)} components over {len(links)} links -> {args.output}')


if __name__ == '__main__':
    main()
