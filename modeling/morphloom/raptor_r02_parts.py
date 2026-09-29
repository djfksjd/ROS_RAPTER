#!/usr/bin/env python3
"""Emit a Morphloom AssemblyIR for the R-02 visuals (target form: user's raptor-views reference).

Same conventions as raptor_parts.py (component id `<urdf_link>__<part>`, authored in the link's ROS frame at
the URDF zero pose). Dimensions come from config/r02_design.yaml and the R-02 URDF, so the visual envelope
follows the physics model: torso with the front sensor pod built in (no neck), sensor mast and camera box,
pelvis drum actuators, femur/tibia/metatarsus with belt covers and a visible Achilles spring, two-toe feet with
claws plus a raised digit-II sickle claw, 16-band rising tail. Visual only; collision/inertia stay in the Xacro.

Usage: python3 modeling/morphloom/raptor_r02_parts.py R02.urdf OUT/assembly.json [--style reference]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import raptor_parts as P
from raptor_parts import Builder, bolt_circle, claw, rot

ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT/'src/raptor_description/config/r02_design.yaml').read_text())
G = D['geometry']


def build_base(b):
    L, tl, pod = 'base_link', G['torso_len'], G['pod_len']
    front = tl/2 + pod
    b.box(L, 'core', (tl, .16, .12), (0, 0, 0), 'graphite', .02)
    b.box(L, 'frame_top', (tl*.85, .14, .016), (-.02, 0, .062), 'graphite', .005)
    b.box(L, 'belly_pod', (tl*.55, .11, .04), (-.02, 0, -.075), 'graphite', .01)
    for side, y in (('l', 1), ('r', -1)):
        b.plate(L, f'armor_{side}', [(-.03, -.052), (.12, -.062), (.2, -.03), (.2, .045), (.09, .064), (-.01, .06),
                                     (-.05, .012)], .012, y*.086, 'ivory', .003)
        b.plate(L, f'armor_trim_{side}', [(-.01, -.04), (.11, -.052), (.18, -.024), (.15, -.018), (.1, -.042),
                                          (0, -.03)], .004, y*.093, 'accent', .001)
        for i, (x, z) in enumerate(((.02, .04), (.11, .045), (.17, .015), (.02, -.04))):
            b.cyl(L, f'panel_bolt_{side}{i}', .004, .004, (x, y*.093, z), 'y', 'steel', segments=12)
        for i, x in enumerate((-.17, -.125, -.08, -.035)):
            r = .03 if i % 2 else .025
            b.cyl(L, f'gear_{side}{i}', r, .018, (x, y*.084, .004), 'y', 'graphite')
            b.cyl(L, f'gear_hub_{side}{i}', r*.45, .022, (x, y*.086, .004), 'y', 'titanium', segments=24)
            b.ring(L, f'gear_rim_{side}{i}', r*.82, .003, (x, y*.094, .004), 'y', 'titanium')
        b.box(L, f'hip_mount_{side}', (.11, .03, .09), (0, y*.07, -.03), 'titanium', .01)
        b.tube(L, f'cable_{side}', [(.19, y*.07, -.02), (.1, y*.095, -.045), (.02, y*.1, -.07)], .006, 'copper')
        b.tube(L, f'drive_cable_{side}', [(-.19, y*.07, .04), (-.12, y*.09, .052), (-.05, y*.092, .045),
                                          (-.01, y*.085, .03)], .0045, 'copper')
    # front sensor pod: wedge helmet, cheek plates, blunt nose with a three-lens cluster (target form)
    x0 = tl/2
    b.box(L, 'pod_core', (pod, .12, .09), (x0 + pod/2, 0, -.004), 'graphite', .018)
    b.plan_plate(L, 'helmet', [(x0-.06, .08), (x0+.05, .078), (front-.02, .04), (front, .01), (front, -.01),
                               (front-.02, -.04), (x0+.05, -.078), (x0-.06, -.08)], .02, .05, 'face', .004)
    b.plan_plate(L, 'helmet_ridge', [(x0-.03, .024), (front-.04, .018), (front-.02, 0), (front-.04, -.018),
                                     (x0-.03, -.024)], .009, .063, 'face', .002)
    for side, y in (('l', 1), ('r', -1)):
        b.plate(L, f'cheek_{side}', [(x0-.02, -.055), (front-.05, -.048), (front-.005, -.024), (front, .005),
                                     (front-.02, .034), (x0+.04, .044), (x0-.02, .046)], .012, y*.07, 'face', .003)
        b.cyl(L, f'cheek_sensor_{side}', .008, .006, (front-.06, y*.077, .01), 'y', 'lens', segments=16)
        b.ring(L, f'cheek_sensor_ring_{side}', .0095, .0018, (front-.06, y*.08, .01), 'y', 'titanium')
    b.box(L, 'visor', (.01, .075, .028), (front, 0, .004), 'graphite', .006)
    for side, y, r in (('l', .024, .01), ('c', 0, .0055), ('r', -.024, .01)):
        b.cyl(L, f'lens_{side}', r, .01, (front + .004, y, .004), 'x', 'lens')
        b.ring(L, f'lens_bezel_{side}', r+.0022, .0022, (front + .007, y, .004), 'x', 'titanium')
    b.box(L, 'mouth_slot', (.016, .06, .005), (front, 0, -.026), 'rubber', .002)
    # sensor mast and camera box on top of the torso
    mz = G['mast_z']
    b.box(L, 'sled', (.12, .10, .035), (.08, 0, .082), 'graphite', .01)
    b.box(L, 'sled_plate', (.11, .09, .008), (.08, 0, .102), 'ivory', .003)
    for side, y in (('l', .03), ('r', -.03)):
        b.cyl(L, f'headlight_{side}', .012, .01, (.142, y, .082), 'x', 'lens')
        b.ring(L, f'headlight_ring_{side}', .0135, .0025, (.146, y, .082), 'x', 'titanium')
    b.cyl(L, 'mast', .022, mz - .02, (.06, 0, .1 + (mz - .02)/2), 'z', 'graphite')
    b.ring(L, 'mast_ring', .025, .0035, (.06, 0, .1 + (mz - .02)*.4), 'z', 'accent')
    b.cyl(L, 'mast_lens', .028, .032, (.06, 0, mz + .096), 'z', 'graphite')
    b.cyl(L, 'mast_window', .024, .01, (.06, 0, mz + .116), 'z', 'lens')
    b.box(L, 'camera_pod', (.085, .075, .07), (-.05, 0, .1), 'graphite', .01)
    b.box(L, 'camera_face', (.01, .066, .058), (-.006, 0, .1), 'face', .005)
    b.box(L, 'camera_label', (.003, .026, .015), (-.001, .012, .11), 'label', .0015)
    b.box(L, 'spine_rail', (tl*.9, .04, .018), (-.02, 0, .072), 'titanium', .006)
    b.box(L, 'heat_sink', (.07, .06, .016), (-.15, 0, .078), 'titanium', .003)
    for i in range(4):
        b.box(L, f'fin_{i}', (.005, .06, .02), (-.175 + i*.014, 0, .094), 'titanium', .001)
    b.cyl(L, 'tail_flange', .052, .02, (-tl/2 - .005, 0, 0), 'x', 'titanium')
    b.ring(L, 'tail_flange_ring', .046, .005, (-tl/2 + .006, 0, 0), 'x', 'accent')


def build_leg(b, side):
    s = 1 if side == 'left' else -1
    H, T, S, M, F = (f'{side}_{n}_link' for n in ('hip_roll', 'thigh', 'shin', 'metatarsus', 'foot'))
    fe, ti, me = G['femur'], G['tibia'], G['meta']
    # pelvis drum actuator block (hip pitch, knee and ankle drives live here; belts run down the leg)
    b.cyl(H, 'drum', .068, .085, (0, s*.03, 0), 'y', 'ivory')
    b.cyl(H, 'drum_cap', .044, .09, (0, s*.032, 0), 'y', 'titanium')
    b.ring(H, 'drum_ring', .058, .006, (0, s*.075, 0), 'y', 'accent')
    b.cyl(H, 'drum_hub', .018, .095, (0, s*.032, 0), 'y', 'graphite', segments=24)
    bolt_circle(b, H, 'drum_bolt', (0, s*.077, 0), .032, 8, 'y')
    b.box(H, 'drive_pack', (.07, .06, .08), (-.07, s*.02, -.01), 'graphite', .012)
    # femur: armour, strut, belt cover to the knee pulley
    b.plate(T, 'femur_armor', [(-.05, -.015), (.055, -.025), (.045, -fe*.8), (.022, -fe), (-.028, -fe*.98),
                               (-.05, -fe*.55)], .07, 0, 'ivory', .004)
    b.box(T, 'femur_strut', (.03, .07, fe*.9), (-.015, 0, -fe/2), 'graphite', .008)
    b.box(T, 'belt_cover', (.022, .018, fe*.95), (-.02, s*.045, -fe/2), 'titanium', .006)
    b.cyl(T, 'knee_pulley', .042, .08, (0, 0, -fe), 'y', 'titanium')
    b.ring(T, 'knee_ring', .036, .004, (0, s*.041, -fe), 'y', 'accent')
    bolt_circle(b, T, 'knee_bolt', (0, s*.043, -fe), .024, 6, 'y')
    # tibia: twin struts, armour, cable to the ankle pulley
    for i, y in enumerate((.02, -.02)):
        b.box(S, f'strut_{i}', (.024, .018, ti*.95), (-.008, y, -ti/2), 'graphite', .007)
    b.plate(S, 'shin_armor', [(.012, -.03), (.04, -.04), (.036, -ti*.72), (.022, -ti*.92), (.008, -ti*.88)], .06, 0,
            'ivory', .003)
    b.plate(S, 'calf_cover', [(-.04, -.03), (-.016, -.02), (-.012, -ti*.8), (-.028, -ti*.95), (-.045, -ti*.7)], .055, 0,
            'ivory', .003)
    b.box(S, 'label_strip', (.003, .024, .04), (.041, 0, -ti*.35), 'label', .0015)
    b.tube(S, 'cable', [(-.045, s*.026, -.02), (-.05, s*.03, -ti*.5), (-.035, s*.026, -ti*.95)], .0045, 'copper')
    b.cyl(S, 'ankle_pulley', .036, .075, (0, 0, -ti), 'y', 'titanium')
    b.ring(S, 'ankle_ring', .03, .005, (0, s*.038, -ti), 'y', 'accent')
    # metatarsus: slender strut, guard, Achilles spring along the back, MTP joint drum
    b.box(M, 'strut', (.03, .034, me*.92), (0, 0, -me/2), 'graphite', .01)
    b.box(M, 'guard', (.014, .04, me*.78), (.022, 0, -me*.5), 'steel', .005)
    b.cyl(M, 'achilles_spring', .008, me*.72, (-.026, 0, -me*.45), 'z', 'accent', segments=16)
    b.ring(M, 'spring_coil_0', .011, .0022, (-.026, 0, -me*.25), 'z', 'steel')
    b.ring(M, 'spring_coil_1', .011, .0022, (-.026, 0, -me*.45), 'z', 'steel')
    b.ring(M, 'spring_coil_2', .011, .0022, (-.026, 0, -me*.65), 'z', 'steel')
    b.cyl(M, 'mtp_drum', .026, .06, (0, 0, -me), 'y', 'titanium')
    # foot (MTP frame, toes flat in the nominal pose): roll bearing, bumper pad
    pad = D['pad']
    b.cyl(F, 'roll_bearing', .02, .05, (.0, 0, .0), 'x', 'graphite')
    b.add(F, 'pad', {'op': 'sphere', 'radius': pad['radius']*1000, 'widthSegments': 24, 'heightSegments': 16},
          (pad['pos'][0], 0, pad['pos'][2]), np.eye(3), 'rubber')
    # toes: digit III (toe_2), IV (lateral), II (medial raised sickle claw)
    digits = {'III': '2', 'IV': '1' if s > 0 else '3', 'II': '3' if s > 0 else '1'}
    for digit, n in digits.items():
        half = D['toes'][digit]['length']/2
        for part in ('proximal', 'distal'):
            link = f'{side}_toe_{n}_{part}_link'
            b.box(link, 'segment', (half, .024, .022), (half/2, 0, 0), 'graphite', .005)
            b.box(link, 'cap', (half*.7, .02, .007), (half*.5, 0, .014), 'ivory', .0025)
            b.cyl(link, 'knuckle', .014, .03, (0, 0, 0), 'y', 'titanium', segments=24)
            if part == 'distal':
                length = P.SICKLE if digit == 'II' else P.CLAW*(1. if digit == 'III' else .8)
                claw(b, link, 'claw', (half - .004, 0, .004), length, .012 if digit == 'II' else .01)


def build_tail(b):
    Y, L = 'tail_yaw_link', 'tail_link'
    b.cyl(Y, 'housing', .05, .05, (-.02, 0, 0), 'x', 'ivory', radius_top=.045)
    b.ring(Y, 'ring', .047, .005, (-.004, 0, 0), 'x', 'accent')
    b.cyl(Y, 'pitch_drive', .032, .1, (-.04, 0, 0), 'y', 'titanium')
    bolt_circle(b, Y, 'pitch_bolt', (-.04, .051, 0), .02, 6, 'y')
    span, n = G['tail_len'], 16
    step = span/n
    curve = lambda u: P.TAIL_RISE*u**2  # noqa: E731  visual upward sweep; collision stays straight
    b.tube(L, 'spine', [(-.01, 0, 0), (-span*.33, 0, curve(.33)), (-span*.66, 0, curve(.66)), (-span, 0, curve(1.))], .009,
           'graphite')
    for i in range(n):
        u = i/(n - 1)
        x, z, w = -.03 - step*i, curve(u), .08*(1 - .72*u)
        tilt = rot((0, 1, 0), -P.TAIL_TILT*u)
        b.box(L, f'band_{i:02d}', (step*.84, w, w*.84), (x, 0, z), P.BAND, min(.015, w/3), tilt)
        b.box(L, f'band_joint_{i:02d}', (step*.2, w*.84, w*.72), (x - step*.5, 0, z), 'graphite', w/5, tilt)
        b.box(L, f'band_seam_{i:02d}', (step*.66, w*.9, .0035), (x, 0, z + w*.3), 'graphite', .0015, tilt)
    b.cyl(L, 'tip', .011, .05, (-.03 - step*(n - 1) - step*.7, 0, curve(1.) + .003), rot((0, 1, 0), -math.pi/2), 'ivory',
          radius_top=.002, segments=16)
    for side, y in (('l', 1), ('r', -1)):
        b.tube(L, f'side_cable_{side}', [(-.02, y*.04, -.008), (-span*.33, y*.032, max(0., curve(.33) - .008)),
                                         (-span*.66, y*.022, max(.02, curve(.66) - .01))], .003, 'copper')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('output')
    ap.add_argument('--style', choices=sorted(P.STYLES), default='reference')
    a = ap.parse_args()
    P.STYLE = P.STYLES[a.style]
    if a.style == 'reference':  # silver armour, chrome bands, longer talons, rising tail (target views)
        P.CLAW, P.SICKLE, P.TAIL_RISE, P.TAIL_TILT, P.BAND = .08, .09, .18, .4, 'chrome'
    b = Builder(Path(a.urdf).read_text())
    build_base(b)
    for side in ('left', 'right'):
        build_leg(b, side)
    build_tail(b)
    links = sorted({c['id'].split('__')[0] for c in b.components})
    Path(a.output).write_text(json.dumps({'schema': 'morphloom.assembly/0.1', 'name': 'raptor-r02-visual', 'units': 'mm',
                                          'components': b.components}, indent=1))
    print(f'{len(b.components)} components over {len(links)} links -> {a.output}')


if __name__ == '__main__':
    main()
