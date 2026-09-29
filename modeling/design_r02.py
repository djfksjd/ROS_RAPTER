"""R-02 design calculator: lighter, lower, ostrich/dromaeosaur-proportioned raptor (user decision 2026-09-29).

Every number the R-02 model generator uses comes from here; derived quantities follow explicit equations.
Nothing here is a walking result: it sizes a design and states what a run at each speed would demand.
v2 follows the UNI_AI (gpt-6-sol) review of v1: trajectory-based joint speeds and torques, reach check,
GRF from duty factor, CoM from part positions, tail ratio against the tail-less body.

Model
- Sagittal leg, pantograph approximation of the bird leg (metatarsus parallel to the femur, as in
  BirdBot/Cassie): hip->MTP is a two-link chain of lengths (femur+meta) and tibia; knee and ankle
  flexions are equal and opposite. Toes III (main) and IV (24 deg out) lie flat; the centre of
  pressure (COP) sits cop_frac of toe III in front of the MTP.
- Running at speed v, duty factor d, stride frequency f: stance t_c = d/f, the hip moves over a fixed
  COP at height h_run; swing brings the foot forward with a (1-cos) profile and a sine clearance.
  Joint angles by IK every 1 ms, velocities by finite difference.
- Vertical GRF half-sine per leg, peak F = pi*M*g/(4d) (mean over the stride = Mg/2 per leg); the force
  acts along COP->hip (spring-mass leg). Stance joint torques = moment of the force about each joint.
  Swing adds leg inertia about the hip x hip angular acceleration.
- Duty factor and relative stride frequency from ostrich data (Rubenson 2011, Smith 2010, Daley 2016),
  extrapolated beyond ostrich speeds (u > 3.2) - marked in the output.

Usage: python3 modeling/design_r02.py [--sweep] [--out docs/design_r02.json]
"""
import argparse
from dataclasses import dataclass, asdict, replace
import json
from math import atan2, cos, pi, sin, sqrt

import numpy as np

G = 9.81
DT = 1e-3
REACH = .90  # max hip-MTP distance as a fraction of full reach: keeps the knee flexed (>~25 deg), away from the straight-leg singularity


@dataclass
class Params:
    # --- masses (kg); actuator masses are catalogue classes (UNVERIFIED parts)
    act_big: float = .62          # ~36 N·m-class QDD: hip pitch, knee
    act_small: float = .40        # ~17 N·m-class QDD: hip roll, ankle pitch, ankle roll, tail
    transmission: float = .15     # per leg: belts/cables/pulleys/bearings to knee, ankle, ankle roll
    torso_frame: float = 1.0
    battery: float = .6           # ~108 Wh at 180 Wh/kg (UNVERIFIED cell choice)
    compute_sensors: float = .5
    head_neck: float = .45
    tube_density: float = .6      # kg/m of leg structure (carbon tube + joint housings)
    foot_toes: float = .10
    tail_struct: float = .25
    tail_tip: float = .40
    # --- geometry (m)
    femur: float = .22           # sweep optimum 1 : 1.73 : 1.45 (ostrich 1 : 1.89 : 1.82)
    tibia: float = .38
    meta: float = .32
    toe3: float = .10
    toe4: float = .085
    cop_frac: float = .3
    torso_len: float = .40
    head_x: float = .42
    tail_len: float = .90
    # --- postures
    hip_stand: float = .55        # crouched standing / walking hip height
    run_extension: float = .92    # running hip height as a fraction of max reach (femur+meta+tibia)
    clearance_frac: float = .12   # swing foot lift as a fraction of hip height
    # --- speed stages (m/s)
    stages: tuple = (1.6, 4.0, 11.1)
    tail_swing_s: float = .30     # time for a 90 deg tail swing (rest to rest)


def leg_len(p):
    return p.femur + p.tibia + p.meta


def leg_ik(p, dx, dz):
    """Hip->MTP vector (dx, dz) -> (hip pitch from vertical, knee flexion), pantograph legs. None if unreachable."""
    L1, L2 = p.femur + p.meta, p.tibia
    d = sqrt(dx*dx + dz*dz)
    if not abs(L1 - L2) + 1e-6 < d < REACH*(L1 + L2):
        return None
    c = (L1*L1 + L2*L2 - d*d)/(2*L1*L2)
    knee = pi - np.arccos(np.clip(c, -1, 1))                       # flexion (0 = straight)
    a = atan2(dx, -dz)                                              # hip->MTP from vertical, + forward
    b = np.arcsin(np.clip(L2*sin(pi - knee)/d, -1, 1))
    return a + b, knee                                              # femur leans forward of the hip-MTP line


def points(p, hip, knee):
    """Joint positions relative to the hip for the pantograph leg (x forward, z up)."""
    uf = np.array([sin(hip), -cos(hip)])
    ut = np.array([sin(hip - knee), -cos(hip - knee)])
    K = p.femur*uf
    A = K + p.tibia*ut
    M = A + p.meta*uf
    return K, A, M


def masses(p):
    act_leg = p.act_small + 2*p.act_big + 2*p.act_small
    leg_struct = p.tube_density*leg_len(p) + p.foot_toes + p.transmission
    parts = {'leg actuators x2 (pelvis)': 2*act_leg, 'tail actuators': 2*p.act_small, 'torso frame': p.torso_frame,
             'battery': p.battery, 'compute+sensors': p.compute_sensors, 'head+neck': p.head_neck,
             'leg structure x2 (tubes, feet, transmission)': 2*leg_struct, 'tail structure': p.tail_struct,
             'tail tip': p.tail_tip}
    return parts, sum(parts.values())


def stand(p):
    """Crouched standing pose: MTP placed so that the whole-body CoM lies over the toe COP."""
    parts, M = masses(p)
    tail_base = -p.torso_len/2
    zt = p.hip_stand + .03
    body = [(parts['leg actuators x2 (pelvis)'] + p.torso_frame + p.battery + p.compute_sensors, 0., zt),
            (p.head_neck, p.head_x, zt + .08), (2*p.act_small, tail_base, zt),
            (p.tail_struct, tail_base - p.tail_len/2, zt), (p.tail_tip, tail_base - p.tail_len, zt)]
    mtp_x = 0.
    for _ in range(50):
        sol = leg_ik(p, mtp_x, -p.hip_stand)
        if sol is None:
            return None
        K, A, Mp = points(p, *sol)
        lm = p.tube_density
        leg = [(2*lm*p.femur, K[0]/2, p.hip_stand + K[1]/2), (2*lm*p.tibia, (K[0]+A[0])/2, p.hip_stand + (K[1]+A[1])/2),
               (2*(lm*p.meta + p.transmission), (A[0]+Mp[0])/2, p.hip_stand + (A[1]+Mp[1])/2),
               (2*p.foot_toes, Mp[0] + .03, .01)]
        allp = body + leg
        cx = sum(m*x for m, x, _ in allp)/M
        cz = sum(m*z for m, _, z in allp)/M
        new = cx - p.cop_frac*p.toe3
        if abs(new - mtp_x) < 1e-5:
            break
        mtp_x = new
    hip, knee = sol
    W = M*G
    cop = np.array([mtp_x + p.cop_frac*p.toe3, -p.hip_stand])
    # static single-leg support, vertical force through the COP
    tq = {n: abs((cop - j)[0])*W for n, j in (('knee', K), ('ankle', A), ('mtp', Mp))}
    tq['hip'] = abs(cop[0])*W
    # tail vs tail-less body pitch inertia about the whole CoM; roll about the stance foot
    I_tail = p.tail_struct*p.tail_len**2/3 + p.tail_tip*p.tail_len**2
    rest = [(m, x, z) for m, x, z in allp if not (x <= tail_base - .01 and abs(z - zt) < 1e-9)]
    I_body = sum(m*((x - cx)**2 + (z - cz)**2) for m, x, z in rest) + .02*(M - p.tail_tip - p.tail_struct)
    return dict(total_mass=round(M, 2), mass_parts={k: round(v, 3) for k, v in parts.items()},
                leg_fraction=round(sum(m for m, *_ in leg)/M, 3),
                com_x_from_hip=round(cx, 3), com_height=round(cz, 3), mtp_x_from_hip=round(mtp_x, 3),
                hip_pitch_deg=round(np.degrees(hip), 1), knee_flex_deg=round(np.degrees(knee), 1),
                static_single_leg_torque_nm={k: round(v, 1) for k, v in tq.items()},
                tail_inertia_base=round(I_tail, 3), body_pitch_inertia_no_tail=round(I_body, 3),
                tail_to_body_pitch=round(I_tail/I_body, 2), tail_to_roll_about_foot=round(I_tail/(M*cz*cz), 3),
                toe_support={'length_m': round(p.toe3 + .01, 3), 'lateral_half_width_m': round(p.toe4*sin(np.radians(24)) + .013, 3)})


def gait_params(u):
    """Duty factor and relative stride frequency f*sqrt(L/g) vs speed u = v/sqrt(gL) (ostrich data)."""
    d = float(np.interp(u, [.29, .6, .96, 1.24, 3.2], [.66, .55, .42, .38, .30]))
    fr = float(np.interp(u, [.29, .96, 3.2], [.40, .49, .62]))
    return (d if u <= 3.2 else .30), (fr if u <= 3.2 else .62)


def foot_path(p, v, h, d, f):
    """Foot (COP) relative to the hip over one stride: stance over a fixed COP, then a cubic Hermite
    swing that starts and ends at the stance relative velocity -v (no jump at lift-off/touchdown, zero
    foot speed over ground at contact; its end overshoot is swing-leg retraction) with a (1-cos) lift."""
    T = 1/f
    tc, ts = d*T, (1 - d)*T
    step = v*tc
    t = np.arange(0, T, DT)
    xs, zs = np.empty_like(t), np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < tc:
            xs[i], zs[i] = step/2 - v*ti, -h
        else:
            s_ = (ti - tc)/ts
            h00, h10, h01, h11 = 2*s_**3 - 3*s_**2 + 1, s_**3 - 2*s_**2 + s_, -2*s_**3 + 3*s_**2, s_**3 - s_**2
            xs[i] = h00*(-step/2) + h10*(-v*ts) + h01*(step/2) + h11*(-v*ts)
            zs[i] = -h + p.clearance_frac*h*(1 - cos(2*pi*s_))/2   # zero vertical speed at lift-off/touchdown
    q = [leg_ik(p, x - p.cop_frac*p.toe3, z) for x, z in zip(xs, zs)]
    return t, tc, xs, zs, (None if any(qi is None for qi in q) else np.array(q, float))


def stride_plan(p, v):
    """Highest running hip height (<= run_extension * max reach) whose whole foot path stays reachable;
    if none down to 0.6 * reach works, the stride frequency is raised (shorter steps)."""
    Lmax = leg_len(p)
    for boost in np.arange(1., 3.01, .05):
        for h in np.arange(p.run_extension*Lmax, .6*Lmax - 1e-9, -.01):
            u = v/sqrt(G*h)
            d, fr = gait_params(u)
            f = boost*fr/sqrt(h/G)
            path = foot_path(p, v, h, d, f)
            if path[-1] is not None:
                return h, u, d, f, float(boost), path
    return None


def run_stage(p, v):
    _, M = masses(p)
    L = leg_len(p)
    plan = stride_plan(p, v)
    if plan is None:
        return dict(speed_mps=v, speed_kmh=round(v*3.6, 1), reachable=False)
    h, u, d, f, boost, (t, tc, xs, zs, q) = plan
    T = 1/f
    qd = np.gradient(q, DT, axis=0)
    qdd = np.gradient(qd, DT, axis=0)
    F = np.pi*M*G/(4*d)
    tau = {'hip': 0., 'knee': 0., 'ankle': 0., 'mtp': 0.}
    power = {'hip': 0., 'knee': 0.}
    lm = p.tube_density
    # swing inertia about the hip (pantograph, mid-crouch): transmission runs along the femur (proximal)
    I_leg = (lm*p.femur + p.transmission)*p.femur**2/3 + lm*p.tibia*p.femur**2 + (lm*p.meta + p.foot_toes)*(.7*L)**2
    for i, ti in enumerate(t):
        K, A, Mp = points(p, *q[i])
        tk = 0.
        if ti < tc:
            Fz = F*sin(pi*ti/tc)
            cop = np.array([xs[i], zs[i]])
            n = -cop/np.linalg.norm(cop)                  # COP -> hip (spring-mass leg force direction)
            fvec = n*Fz/n[1]
            for name, j in (('knee', K), ('ankle', A), ('mtp', Mp)):
                r = cop - j
                m_ = abs(r[0]*fvec[1] - r[1]*fvec[0])
                tau[name] = max(tau[name], m_)
                if name == 'knee':
                    tk = m_
        else:
            tau['hip'] = max(tau['hip'], abs(I_leg*qdd[i, 0]))
        power['hip'] = max(power['hip'], abs(I_leg*qdd[i, 0]*qd[i, 0]))
        power['knee'] = max(power['knee'], abs(tk*qd[i, 1]))
    return dict(speed_mps=v, speed_kmh=round(v*3.6, 1), froude_u=round(u, 2), beyond_ostrich=bool(u > 3.2),
                reachable=True, stride_freq_boost=round(boost, 2), run_hip_height=round(h, 3), duty=round(d, 2),
                stride_hz=round(f, 2), stance_s=round(tc, 3), stride_m=round(v*T, 2), peak_grf_bw=round(F/(M*G), 2),
                peak_joint_speed_rad_s={'hip': round(float(np.abs(qd[:, 0]).max()), 1),
                                        'knee_and_ankle': round(float(np.abs(qd[:, 1]).max()), 1)},
                peak_torque_nm={k: round(v2, 1) for k, v2 in tau.items()},
                peak_mech_power_w={k: round(v2) for k, v2 in power.items()},
                electrical_power_w_cot1=round(M*G*v))       # cost of transport ~1 (UNVERIFIED assumption)


def tail_actuator(p):
    th, T = pi/2, p.tail_swing_s
    alpha = 4*th/T**2                                 # bang-bang, rest to rest
    I = p.tail_struct*p.tail_len**2/3 + p.tail_tip*p.tail_len**2
    grav = G*(p.tail_struct*p.tail_len/2 + p.tail_tip*p.tail_len)
    return dict(swing_90deg_s=T, peak_speed_rad_s=round(2*th/T, 1), peak_torque_nm=round(I*alpha + grav, 1),
                angular_momentum_n_m_s=round(I*2*th/T, 2))


def design(p):
    s = stand(p)
    return dict(params=asdict(p), stand=s, stages=[run_stage(p, v) for v in p.stages], tail=tail_actuator(p),
                battery_wh=round(p.battery*180), runtime_min_at_stage1=round(p.battery*180/(masses(p)[1]*G*p.stages[0])*60, 1))


def sweep(base):
    rows = []
    for femur in (.22, .26, .30):
        for tibia in (.30, .38, .43):
            for meta in (.20, .26, .32):
                p = replace(base, femur=femur, tibia=tibia, meta=meta)
                s = stand(p)
                if s is None:
                    continue
                st = [run_stage(p, v) for v in p.stages]
                rows.append(dict(femur=femur, tibia=tibia, meta=meta, leg=round(leg_len(p), 2), mass=s['total_mass'],
                                 com_h=s['com_height'], knee_flex=s['knee_flex_deg'],
                                 stage=[(x['speed_kmh'], x.get('froude_u'), x.get('stride_freq_boost'),
                                         x['reachable'] and x['peak_joint_speed_rad_s']['knee_and_ankle'],
                                         x['reachable'] and x['peak_torque_nm']['knee']) for x in st]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args()
    r = design(Params())
    if a.sweep:
        r['sweep'] = sweep(Params())
    print(json.dumps(r, indent=1, ensure_ascii=False, default=float))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(r, f, indent=1, ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
