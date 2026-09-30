"""R-03 skeleton study: where must the hip sit relative to the whole-body CoM for an ostrich-like stance, and what
does that cost in tail balance? (evidence 86 §13)

Uses the R-02 mass table (design_r02.Params, 11.36 kg). Two knobs:
  hip_back  (m): the hip joint moved rearward along the torso = every torso-mounted mass (frame, battery, compute,
                 pelvis actuators, pod, mast) moves forward by hip_back relative to the hip. The tail root stays at the
                 torso rear, so it also moves forward by hip_back.
  tail_tip  (kg): the tail-end mass (R-02 0.40 kg), total mass kept by moving the removed mass to the battery.
For each pair: CoM x ahead of the hip at the stance height, tail/body pitch inertia ratio, and whether an
ostrich-like leg (femur 60-70 deg forward of vertical, intertarsal ~150 deg, metatarsus near vertical) can put the
toe COP under the CoM with the R-02 segment lengths (femur 0.22, tibia 0.38, metatarsus 0.32).
"""
import json
import sys
from math import cos, radians, sin
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_r02 as D  # noqa: E402


def ostrich_leg_reach(p, femur_deg=(55., 70.), inter_deg=(145., 160.), meta_fwd_deg=(-5., 15.)):
    """Horizontal MTP position ahead of the hip over a family of ostrich-like leg poses (angles from vertical)."""
    xs = []
    for f in np.linspace(*femur_deg, 4):
        for it in np.linspace(*inter_deg, 4):
            for mf in np.linspace(*meta_fwd_deg, 5):
                tib_back = (180-it) - mf          # intertarsal included = 180 - (tibia back + meta forward)
                x = p.femur*sin(radians(f)) - p.tibia*sin(radians(tib_back)) + p.meta*sin(radians(mf))
                z = p.femur*cos(radians(f)) + p.tibia*cos(radians(tib_back)) + p.meta*cos(radians(mf))
                xs.append((x, z, f, it, mf))
    return xs


def study(hip_backs=(0., .1, .2, .3, .4), tail_tips=(.40, .20, .10, 0.)):
    base = D.Params()
    legs = ostrich_leg_reach(base)
    lo, hi = min(x for x, *_ in legs), max(x for x, *_ in legs)
    rows = []
    for tt in tail_tips:
        p = D.Params(tail_tip=tt, battery=base.battery+(base.tail_tip-tt))
        parts, M = D.masses(p)
        for hb in hip_backs:
            tail_base = -p.torso_len/2 + hb
            body = [(parts['leg actuators x2 (pelvis)'] + p.torso_frame + p.battery + p.compute, hb),
                    (p.sensor_pod, p.torso_len/2 + p.pod_len/2 + hb), (p.mast_sensors, .06 + hb), (2*p.act_small, tail_base),
                    (p.tail_struct, tail_base - p.tail_len/2), (p.tail_tip, tail_base - p.tail_len)]
            leg_m = M - sum(m for m, _ in body)
            cx = (sum(m*x for m, x in body) + leg_m*.05)/M   # legs roughly under the hip, slightly forward
            I_tail = p.tail_struct*p.tail_len**2/3 + p.tail_tip*p.tail_len**2
            I_body = sum(m*(x-cx)**2 for m, x in body[:3]) + .02*(M - p.tail_tip - p.tail_struct) + leg_m*.05
            need = cx - p.cop_frac*p.toe3                  # MTP x so that the toe COP is under the CoM
            ok = lo <= need <= hi
            rows.append(dict(tail_tip_kg=tt, hip_back_m=hb, com_ahead_of_hip_m=round(cx, 3), tail_to_body_pitch=round(I_tail/I_body, 2),
                             mtp_needed_m=round(need, 3), ostrich_pose_reach_m=[round(lo, 3), round(hi, 3)], ostrich_stance_possible=ok))
    return rows


if __name__ == '__main__':
    rows = study()
    print(json.dumps(rows, indent=1))
