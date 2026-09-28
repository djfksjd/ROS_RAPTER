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
    'bone': ('claw polymer', {'color': '#cfc6b2', 'surface': 'molded-polymer', 'roughness': .28, 'metalness': 0.,
                              'clearcoat': .5}),
    'label': ('warning label', {'color': '#b8452a', 'surface': 'molded-polymer', 'roughness': .5, 'metalness': 0.}),
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

    def plan_plate(self, link, name, profile_xy, thickness, z, mat, bevel=.002):
        """Extruded panel seen from above: profile in the link's x-y plane, thickness along z (centred at z)."""
        self.add(link, name, {'op': 'extrude', 'points': [[x*1000, y*1000] for x, y in profile_xy],
                              'depth': thickness*1000, 'bevelSize': bevel*1000, 'bevelThickness': bevel*1000,
                              'bevelSegments': 2}, (0, 0, z), rot((1, 0, 0), -math.pi/2), mat)

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


def bolt_circle(b, link, name, center, radius, count, axis, mat='steel', size=.005):
    """Bolt heads on a disc face whose normal is ROS `axis` ('y' or 'x')."""
    for i in range(count):
        a = 2*math.pi*i/count
        u, v = radius*math.cos(a), radius*math.sin(a)
        offset = (u, 0, v) if axis == 'y' else (0, u, v)
        b.cyl(link, f'{name}_{i}', size, .006, tuple(np.add(center, offset)), axis, mat, segments=10)


def claw(b, link, name, root, length, radius, s_rot=0., mat='bone'):
    """Curved claw from three tapered segments bending forward then down (ROS x forward)."""
    pos = np.array(root, float)
    for k, (bend, taper) in enumerate(((1.35, .78), (1.95, .6), (2.5, .4))):
        r = rot((0, 0, 1), s_rot)@rot((0, 1, 0), bend)
        seg = length/3
        r_bot = radius*(1 if k == 0 else [1, .78, .47][k])
        b.cyl(link, f'{name}_{k}', r_bot, seg, pos+r@np.array([0, 0, seg/2]), r, mat,
              radius_top=r_bot*taper, segments=18)
        pos = pos+r@np.array([0, 0, seg*.92])


def build_base(b):
    L = 'base_link'
    b.box(L, 'core', (.50, .20, .15), (-.02, 0, 0), 'graphite', .02)
    b.box(L, 'frame_top', (.40, .17, .02), (-.06, 0, .075), 'graphite', .006)
    for side, y in (('l', 1), ('r', -1)):
        # ivory armour covers the front half; the rear half shows the drive train (reference views)
        b.plate(L, f'armor_{side}', [(-.02, -.07), (.16, -.085), (.27, -.035), (.28, .05), (.12, .085), (.0, .08),
                                     (-.04, .02)], .014, y*.118, 'ivory', .003)
        b.plate(L, f'armor_trim_{side}', [(.0, -.05), (.14, -.07), (.24, -.03), (.2, -.022), (.12, -.058),
                                          (.01, -.04)], .004, y*.127, 'accent', .001)
        for i, (x, z) in enumerate(((.05, .05), (.15, .055), (.22, .02), (.05, -.05))):
            b.cyl(L, f'panel_bolt_{side}{i}', .005, .004, (x, y*.127, z), 'y', 'steel', segments=12)
        for i, x in enumerate((-.23, -.17, -.11, -.05)):
            r = .036 if i % 2 else .03
            b.cyl(L, f'gear_{side}{i}', r, .022, (x, y*.108, .005), 'y', 'graphite')
            b.cyl(L, f'gear_hub_{side}{i}', r*.45, .026, (x, y*.11, .005), 'y', 'titanium', segments=24)
            b.ring(L, f'gear_rim_{side}{i}', r*.82, .0035, (x, y*.12, .005), 'y', 'titanium')
        b.plate(L, f'octagon_frame_{side}', [(-.09, -.045), (-.06, -.075), (.03, -.075), (.06, -.045), (.06, .005),
                                              (.03, .035), (-.06, .035), (-.09, .005)], .01, y*.1, 'graphite', .002)
        b.cyl(L, f'hip_bearing_{side}', .058, .045, (0, y*.145, -.06), 'y', 'titanium')
        b.ring(L, f'hip_bearing_ring_{side}', .05, .006, (0, y*.168, -.06), 'y', 'accent')
        b.tube(L, f'cable_{side}', [(.24, y*.10, -.02), (.12, y*.135, -.05), (.02, y*.14, -.10), (-.05, y*.12, -.12)],
               .007, 'copper')
        b.tube(L, f'pipe_loop_{side}', [(.13, y*.03, .125), (.16, y*.09, .1), (.17, y*.125, .04), (.14, y*.13, -.02)],
               .006, 'copper')
        b.tube(L, f'drive_cable_{side}', [(-.24, y*.1, .05), (-.16, y*.125, .065), (-.08, y*.13, .06),
                                          (-.02, y*.12, .04)], .005, 'copper')
    # wedge head (reference): arrow-shaped helmet in plan, chamfered cheek plates, blunt nose with slot
    b.box(L, 'head_core', (.17, .17, .10), (.23, 0, -.005), 'graphite', .02)
    b.plan_plate(L, 'helmet', [(.11, .1), (.25, .097), (.33, .05), (.358, .012), (.358, -.012), (.33, -.05),
                               (.25, -.097), (.11, -.1)], .022, .058, 'ivory', .004)
    b.plan_plate(L, 'helmet_ridge', [(.14, .03), (.30, .022), (.33, 0), (.30, -.022), (.14, -.03)], .01, .073,
                 'ivory', .002)
    b.cyl(L, 'helmet_sensor', .008, .006, (.29, 0, .08), 'z', 'lens', segments=16)
    for side, y in (('l', 1), ('r', -1)):
        b.plate(L, f'cheek_{side}', [(.10, -.07), (.28, -.058), (.345, -.03), (.352, .005), (.33, .04), (.24, .052),
                                     (.11, .055)], .014, y*.097, 'ivory', .003)
        b.plate(L, f'cheek_trim_{side}', [(.14, -.045), (.29, -.04), (.32, -.028), (.15, -.032)], .004, y*.105,
                'accent', .001)
        b.cyl(L, f'cheek_sensor_{side}', .009, .006, (.27, y*.105, .012), 'y', 'lens', segments=16)
        b.ring(L, f'cheek_sensor_ring_{side}', .011, .002, (.27, y*.108, .012), 'y', 'titanium')
    b.box(L, 'nose', (.05, .09, .07), (.337, 0, -.012), 'graphite', .016)
    b.box(L, 'mouth_slot', (.02, .07, .006), (.355, 0, -.03), 'rubber', .002)
    b.box(L, 'visor', (.01, .085, .03), (.359, 0, .004), 'graphite', .006)
    for side, y, r in (('l', .026, .011), ('c', 0, .006), ('r', -.026, .011)):
        b.cyl(L, f'lens_{side}', r, .01, (.362, y, .004), 'x', 'lens')
        b.ring(L, f'lens_bezel_{side}', r+.0025, .0025, (.365, y, .004), 'x', 'titanium')
    b.cyl(L, 'antenna', .006, .13, (.12, -.09, .15), rot((1, 0, 0), .55)@rot((0, 1, 0), -.6), 'titanium',
          radius_top=.001, segments=12)
    # upper sensor sled with twin headlights and a tall lens mast
    b.box(L, 'sled', (.15, .12, .045), (.13, 0, .135), 'graphite', .012)
    b.box(L, 'sled_plate', (.13, .11, .01), (.13, 0, .162), 'ivory', .004)
    for side, y in (('l', .035), ('r', -.035)):
        b.cyl(L, f'headlight_{side}', .014, .012, (.207, y, .135), 'x', 'lens')
        b.ring(L, f'headlight_ring_{side}', .016, .003, (.212, y, .135), 'x', 'titanium')
        b.tube(L, f'handle_{side}', [(.07, y*1.7, .165), (.08, y*1.7, .185), (.18, y*1.7, .185), (.19, y*1.7, .165)],
               .004, 'steel')
    b.cyl(L, 'mast', .026, .07, (.12, 0, .2), 'z', 'graphite')
    b.ring(L, 'mast_ring', .029, .004, (.12, 0, .215), 'z', 'accent')
    b.cyl(L, 'mast_lens', .032, .035, (.12, 0, .252), 'z', 'graphite')
    b.cyl(L, 'mast_window', .028, .012, (.12, 0, .275), 'z', 'lens')
    # camera box with label and cable loop, spine rail and heat sink
    b.box(L, 'camera_pod', (.10, .09, .085), (-.03, 0, .135), 'graphite', .012)
    b.box(L, 'camera_face', (.012, .08, .07), (.022, 0, .135), 'ivory', .006)
    b.box(L, 'camera_label', (.004, .03, .018), (.029, .015, .145), 'label', .002)
    b.tube(L, 'camera_cable', [(-.07, .045, .15), (-.1, .06, .12), (-.09, .05, .08), (-.06, .03, .09)], .004, 'graphite')
    b.box(L, 'spine_rail', (.44, .05, .022), (-.03, 0, .09), 'titanium', .008)
    for side, y in (('l', .06), ('r', -.06)):
        b.cyl(L, f'rail_bar_{side}', .006, .40, (-.02, y, .105), 'x', 'steel', segments=16)
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
    # thigh: ivory hip drive with bolt circle, armoured femur, twin linkage rods, piston
    b.cyl(T, 'hip_drive', .082, .085, (0, s*.01, 0), 'y', 'ivory')
    b.cyl(T, 'hip_drive_cap', .05, .09, (0, s*.012, 0), 'y', 'titanium')
    b.ring(T, 'hip_drive_ring', .068, .007, (0, s*.056, 0), 'y', 'accent')
    b.cyl(T, 'hip_drive_hub', .02, .1, (0, s*.012, 0), 'y', 'graphite', segments=24)
    bolt_circle(b, T, 'hip_bolt', (0, s*.058, 0), .038, 8, 'y')
    b.plate(T, 'femur_armor', [(-.07, -.02), (.075, -.035), (.058, -.20), (.028, -.25), (-.035, -.245), (-.065, -.14)],
            .09, 0, 'ivory', .005)
    b.box(T, 'femur_strut', (.035, .085, .22), (-.02, 0, -.13), 'graphite', .01)
    for i, x in enumerate((-.045, -.03)):
        b.tube(T, f'linkage_{i}', [(x, s*.052, -.02), (x+.01, s*.054, -.13), (x+.02, s*.052, -.24)], .0055, 'steel')
    b.cyl(T, 'femur_piston', .011, .17, (.05, s*.035, -.13), 'z', 'steel', segments=20)
    b.cyl(T, 'femur_piston_body', .017, .08, (.05, s*.035, -.08), 'z', 'graphite', segments=24)
    b.cyl(T, 'knee_drive', .052, .095, (0, 0, -.26), 'y', 'titanium')
    b.ring(T, 'knee_ring', .045, .005, (0, s*.049, -.26), 'y', 'accent')
    bolt_circle(b, T, 'knee_bolt', (0, s*.05, -.26), .03, 6, 'y')
    # shin: twin struts, armour, label strip, cross link, ankle rotor
    for i, y in enumerate((.026, -.026)):
        b.box(S, f'strut_{i}', (.028, .022, .28), (-.012, y, -.15), 'graphite', .008)
    b.plate(S, 'calf_cover', [(-.05, -.03), (-.02, -.02), (-.015, -.23), (-.035, -.27), (-.06, -.20)], .07, 0,
            'ivory', .003)
    b.plate(S, 'shin_armor', [(.02, -.04), (.055, -.05), (.05, -.20), (.03, -.255), (.012, -.24)], .075, 0, 'ivory', .004)
    b.box(S, 'label_strip', (.004, .03, .05), (.056, s*.0, -.11), 'label', .002)
    b.box(S, 'cross_link', (.07, .016, .02), (.0, s*.042, -.17), 'titanium', .005)
    b.cyl(S, 'tibia_piston', .01, .22, (.03, 0, -.14), 'z', 'steel', segments=20)
    b.cyl(S, 'tibia_piston_body', .016, .09, (.03, 0, -.07), 'z', 'graphite', segments=24)
    b.tube(S, 'cable', [(-.055, s*.03, -.02), (-.06, s*.035, -.15), (-.045, s*.03, -.28)], .005, 'copper')
    b.cyl(S, 'ankle_drive', .044, .085, (0, 0, -.30), 'y', 'titanium')
    b.ring(S, 'ankle_ring', .038, .006, (0, s*.044, -.30), 'y', 'accent')
    b.cyl(S, 'ankle_rotor', .022, .09, (0, 0, -.30), 'y', 'accent', segments=24)
    # foot: inclined metatarsus + pad (matches digitigrade_foot.xacro collision)
    beta, length = .35, .24
    end = np.array([length*math.sin(beta), 0, -length*math.cos(beta)])
    tilt = rot((0, 1, 0), -beta)
    b.box(F, 'metatarsus', (.045, .05, length), end/2, 'graphite', .012, tilt)
    b.box(F, 'metatarsus_guard', (.018, .058, length*.82), end/2+np.array([.028, 0, .006]), 'steel', .006, tilt)
    b.cyl(F, 'metatarsus_spring', .009, length*.7, end/2+np.array([-.03, 0, .01]), tilt, 'accent', segments=16)
    b.box(F, 'bracket', (.03, .064, .03), end*.35+np.array([-.02, 0, 0]), 'ivory', .006, tilt)
    pad_x, pad_z = end[0]-.04, end[2]+.01-.0175
    b.box(F, 'pad', (.15, .10, .035), (pad_x, 0, pad_z), 'rubber', .01)
    b.box(F, 'pad_frame', (.13, .085, .02), (pad_x, 0, pad_z+.022), 'graphite', .006)
    b.cyl(F, 'ball_joint', .03, .07, (end[0], 0, end[2]+.02), 'y', 'titanium')
    claw(b, F, 'sickle_claw', (pad_x-.055, s*.035, pad_z+.03), .07, .011, s_rot=math.pi)
    for d, y in ((1, .040), (2, 0), (3, -.040)):
        for part, length_ in (('proximal', .070), ('distal', .075)):
            link = f'{side}_toe_{d}_{part}_link'
            b.box(link, 'segment', (length_, .026, .026), (length_/2, 0, 0), 'graphite', .006)
            b.box(link, 'cap', (length_*.7, .022, .008), (length_*.5, 0, .016), 'ivory', .003)
            b.cyl(link, 'knuckle', .016, .032, (0, 0, 0), 'y', 'titanium', segments=24)
            b.cyl(link, 'knuckle_pin', .006, .036, (0, 0, 0), 'y', 'steel', segments=12)
            if part == 'distal':
                claw(b, link, 'claw', (length_-.004, 0, .004), .075, .012)


def build_tail(b):
    b.cyl('tail_yaw_link', 'housing', .058, .15, (-.075, 0, 0), 'x', 'ivory', radius_top=.05)
    b.ring('tail_yaw_link', 'ring', .055, .006, (-.012, 0, 0), 'x', 'accent')
    b.cyl('tail_yaw_link', 'pitch_drive', .036, .11, (-.155, 0, 0), 'y', 'titanium')
    bolt_circle(b, 'tail_yaw_link', 'pitch_bolt', (-.155, .056, 0), .022, 6, 'y')
    L = 'tail_link'
    segments, span = 16, .93
    step = span/segments
    curve = lambda u: .07*u**2  # noqa: E731  gentle upward sweep (reference tail is nearly level)
    b.tube(L, 'spine', [(-.01, 0, 0), (-.3, 0, curve(.32)), (-.6, 0, curve(.65)), (-.94, 0, curve(1.))], .01, 'graphite')
    for i in range(segments):
        u = i/(segments-1)
        x = -.03-step*i
        z = curve(u)
        w = .088*(1-.74*u)
        tilt = rot((0, 1, 0), -.14*u)
        b.box(L, f'band_{i:02d}', (step*.84, w, w*.84), (x, 0, z), 'ivory', min(.016, w/3), tilt)
        b.box(L, f'band_joint_{i:02d}', (step*.2, w*.84, w*.72), (x-step*.5, 0, z), 'graphite', w/5, tilt)
        b.box(L, f'band_seam_{i:02d}', (step*.66, w*.9, .004), (x, 0, z+w*.3), 'graphite', .0015, tilt)
        if i < 11:
            for side, y in (('l', 1), ('r', -1)):
                b.cyl(L, f'band_bolt_{i:02d}{side}', .003, .004, (x, y*w/2, z-w*.1), 'y', 'graphite', segments=10)
    b.cyl(L, 'tip', .012, .05, (-.03-step*(segments-1)-step*.7, 0, curve(1.)+.003), rot((0, 1, 0), -math.pi/2), 'ivory', radius_top=.002,
          segments=16)
    for side, y in (('l', 1), ('r', -1)):
        b.tube(L, f'side_cable_{side}', [(-.02, y*.045, -.008), (-.3, y*.036, .0), (-.6, y*.026, .02),
                                         (-.86, y*.014, .05)], .0032, 'copper')


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
