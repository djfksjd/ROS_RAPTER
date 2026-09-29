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
    compute: float = .30          # computer in the torso
    mast_sensors: float = .20     # camera/lidar mast on top of the torso (target form, raptor-views)
    sensor_pod: float = .45       # front sensor pod built into the torso front (no neck; forward counterweight)
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
    pod_len: float = .14          # sensor pod length ahead of the torso front
    mast_z: float = .14           # mast sensor height above the torso centre
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
    # per leg: hip roll, hip pitch, knee, ankle pitch = big class (static single-leg ankle 19.6 N·m and
    # the 14 km/h ankle 43 N·m exceed the small class); ankle roll = small class
    act_leg = 4*p.act_big + p.act_small
    leg_struct = p.tube_density*leg_len(p) + p.foot_toes + p.transmission
    parts = {'leg actuators x2 (pelvis)': 2*act_leg, 'tail actuators': 2*p.act_small, 'torso frame': p.torso_frame,
             'battery': p.battery, 'compute': p.compute, 'mast sensors': p.mast_sensors, 'front sensor pod': p.sensor_pod,
             'leg structure x2 (tubes, feet, transmission)': 2*leg_struct, 'tail structure': p.tail_struct,
             'tail tip': p.tail_tip}
    return parts, sum(parts.values())


def stand(p):
    """Crouched standing pose: MTP placed so that the whole-body CoM lies over the toe COP."""
    parts, M = masses(p)
    tail_base = -p.torso_len/2
    zt = p.hip_stand + .03
    body = [(parts['leg actuators x2 (pelvis)'] + p.torso_frame + p.battery + p.compute, 0., zt),
            (p.sensor_pod, p.torso_len/2 + p.pod_len/2, zt), (p.mast_sensors, .06, zt + p.mast_z), (2*p.act_small, tail_base, zt),
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


def springs(stages, mtp_defl=.35, ankle_share=.5, ankle_defl=.3):
    """Passive spring sizing per speed stage (sizing rule, not an optimum):
    - MTP (toe-base) spring: deflects mtp_defl rad at the stage's peak MTP torque (ostrich MTP excursion
      70-80 deg in running; 98 % of its elastic work is at the MTP, Rubenson 2011).
    - Ankle 'Achilles' spring in parallel with the motor: carries ankle_share of the peak ankle torque at
      ankle_defl rad; the motor supplies the rest. Energy stored = 0.5 k theta^2."""
    out = []
    for st in stages:
        if not st.get('reachable'):
            continue
        tm, ta = st['peak_torque_nm']['mtp'], st['peak_torque_nm']['ankle']
        km, ka = tm/mtp_defl, ankle_share*ta/ankle_defl
        out.append(dict(speed_kmh=st['speed_kmh'], mtp_k_nm_per_rad=round(km, 1), mtp_energy_j=round(.5*km*mtp_defl**2, 2),
                        ankle_k_nm_per_rad=round(ka, 1), ankle_energy_j=round(.5*ka*ankle_defl**2, 2),
                        ankle_motor_peak_nm=round(ta*(1 - ankle_share), 1)))
    return out


def design(p):
    s = stand(p)
    stages = [run_stage(p, v) for v in p.stages]
    return dict(params=asdict(p), stand=s, stages=stages, springs=springs(stages), tail=tail_actuator(p),
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


# ---------------------------------------------------------------------------------------------
# Link table for the Xacro (single source): link-frame mass, CoM and diagonal inertia, joint limits
# and foot/toe layout. Link frames are the URDF zero pose: legs straight down, joint axes as in the
# 12-DOF model (hip roll x, pitch joints y, ankle roll x at the MTP, tail yaw z / pitch y).
LIMITS = {'hip_roll': (-.5, .5), 'hip_pitch': (-1.6, .9), 'knee_pitch': (.2, 2.6), 'ankle_pitch': (-2.5, .2),
          'ankle_roll': (-.5, .5), 'tail_yaw': (-.8, .8), 'tail_pitch': (-.8, .8)}
TOES = {  # digit: (x, y_out, z, yaw_out_deg, pitch_up_rad, length, k_proximal, k_distal, range, tip_friction)
    'III': (0., -.006, -.008, 0., 0., None, 30., 20., (-.6, .9), 1.2),
    'IV': (0., .012, -.008, 24., 0., None, 20., 14., (-.6, .9), None),
    'II': (-.01, -.02, .01, 0., .7, .06, 15., 10., (-.3, .3), None),
}
TOE_SEGMENT_MASS = .01
PAD_RADIUS, PAD_POS = .018, (-.005, 0., -.002)


def composite(parts):
    """[(mass, (x, y, z), (ixx, iyy, izz) own)] -> mass, com, diagonal inertia about the com."""
    m = sum(q[0] for q in parts)
    c = np.sum([q[0]*np.array(q[1]) for q in parts], axis=0)/m
    I = np.zeros(3)
    for mi, r, own in parts:
        d = np.array(r) - c
        I += np.array(own) + mi*np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
    return m, c, I


def box_i(m, x, y, z):
    return (m*(y*y + z*z)/12, m*(x*x + z*z)/12, m*(x*x + y*y)/12)


def rod_z(m, L, r=.02):
    return (m*L*L/12 + m*r*r/4, m*L*L/12 + m*r*r/4, m*r*r/2)


def link_table(p, joint_speed=3.):
    s = stand(p)
    hip0, knee0 = -np.radians(s['hip_pitch_deg']), np.radians(s['knee_flex_deg'])
    tl = p.torso_len
    base = composite([
        (p.torso_frame, (0, 0, 0), box_i(p.torso_frame, tl, .16, .12)),
        (p.battery, (-.02, 0, -.02), box_i(p.battery, .15, .07, .05)),
        (p.compute, (.05, 0, .02), box_i(p.compute, .10, .08, .03)),
        (p.mast_sensors, (.06, 0, p.mast_z), box_i(p.mast_sensors, .08, .08, .10)),
        (p.sensor_pod, (tl/2 + p.pod_len/2, 0, 0), box_i(p.sensor_pod, p.pod_len, .12, .10)),
        (p.act_big, (0, .06, -.02), (5e-4,)*3), (p.act_big, (0, -.06, -.02), (5e-4,)*3),   # hip roll actuators
        (p.act_small, (-tl/2 + .03, 0, 0), (3e-4,)*3),                                    # tail yaw actuator
    ])
    hip_act = 3*p.act_big + p.act_small
    lm = p.tube_density
    tail_m = p.tail_struct + p.tail_tip
    tail = composite([(p.tail_struct, (-p.tail_len/2, 0, 0), (1e-4, p.tail_struct*p.tail_len**2/12, p.tail_struct*p.tail_len**2/12)),
                      (p.tail_tip, (-p.tail_len, 0, 0), (2e-4,)*3)])
    L = lambda m, c, I: {'mass': round(float(m), 5), 'com': [round(float(v), 5) for v in c], 'inertia': [round(float(v), 7) for v in I]}
    links = {
        'base_link': L(*base),
        'hip_roll_link': L(hip_act, (0, 0, 0), (2/5*hip_act*.06**2,)*3),
        'thigh_link': L(lm*p.femur + p.transmission, (0, 0, -p.femur/2), rod_z(lm*p.femur + p.transmission, p.femur)),
        'shin_link': L(lm*p.tibia, (0, 0, -p.tibia/2), rod_z(lm*p.tibia, p.tibia)),
        'metatarsus_link': L(lm*p.meta, (0, 0, -p.meta/2), rod_z(lm*p.meta, p.meta)),
        'foot_link': L(p.foot_toes - 6*TOE_SEGMENT_MASS, (0, 0, 0), (2e-5,)*3),
        'tail_yaw_link': L(p.act_small, (-.02, 0, 0), (5e-4,)*3),
        'tail_link': L(*tail),
    }
    toes = {}
    for d, (x, y, z, yaw, up, length, kp, kd, rng, fr) in TOES.items():
        length = length or (p.toe3 if d == 'III' else p.toe4)
        toes[d] = dict(x=x, y_out=y, z=z, yaw_out_deg=yaw, pitch_up=up, length=length, k_proximal=kp, k_distal=kd,
                       range=list(rng), tip_friction=fr)
    big, small = 36., 17.
    joints = {n: dict(lower=lo, upper=hi, velocity=joint_speed,
                      effort=small if n == 'ankle_roll' else big) for n, (lo, hi) in LIMITS.items()}
    return dict(generated_by='modeling/design_r02.py', total_mass=s['total_mass'],
                nominal_pose={'hip_pitch': round(float(hip0), 4), 'knee_pitch': round(float(knee0), 4), 'ankle_pitch': round(float(-knee0), 4)},
                foot_mount_pitch=round(float(-hip0), 4),   # toe frame pitched so the toes lie flat in the nominal pose
                geometry=dict(femur=p.femur, tibia=p.tibia, meta=p.meta, hip_half_width=.10, hip_drop=.03, torso_len=tl,
                              pod_len=p.pod_len, mast_z=p.mast_z, tail_len=p.tail_len, hip_stand=p.hip_stand),
                links=links, joints=joints, toes=toes, toe_segment_mass=TOE_SEGMENT_MASS,
                pad=dict(radius=PAD_RADIUS, pos=list(PAD_POS)),
                achilles=dict(stiffness=77., preload=6.))   # DESIGN_R02 4.3 (14 km/h sizing); enabled by a Xacro arg


def export(p, path):
    """Write the design table as JSON (valid YAML) for xacro.load_yaml."""
    with open(path, 'w') as f:
        json.dump(link_table(p), f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--export', help='write the Xacro design table (e.g. src/raptor_description/config/r02_design.yaml)')
    a = ap.parse_args()
    r = design(Params())
    if a.export:
        export(Params(), a.export)
    if a.sweep:
        r['sweep'] = sweep(Params())
    print(json.dumps(r, indent=1, ensure_ascii=False, default=float))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(r, f, indent=1, ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
