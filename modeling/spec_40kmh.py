"""40 km/h requirement calculator for R-02 (docs/design/40kmh-spec.md).

Design basis: v = 11.1 m/s (40 km/h). 14 km/h is only a milestone. Nothing here is a running result;
it states what a steady run at 11.1 m/s on flat ground would demand of the R-02 geometry.

Model (all explicit, all approximations named)
1. SLIP stance: point mass M on a massless linear spring leg (rest length L0 = hip -> centre of
   pressure at touchdown, angle th_td from vertical). For a chosen flight time the spring stiffness
   is solved so the stance is mirror-symmetric (periodic gait, Geyer 2005). Outputs: contact time,
   step/stride frequency, peak GRF (body weights), leg compression, leg stiffness, apex height.
2. Joint kinematics: the hip path from the SLIP stance is inverse-mapped through the pantograph leg
   of design_r02.py every 0.1 ms (knee and ankle flexions equal and opposite, toes III/IV flat, COP
   0.03 m in front of the MTP). Swing: cubic Hermite between take-off and touchdown foot positions
   relative to the hip, over one stride minus one stance, with a (1-cos) lift.
3. Joint torques: stance = moment of the SLIP leg force (along COP -> hip, magnitude k*dL) about each
   joint (the hip sees none: the force passes through it); swing = leg inertia about the hip x hip
   angular acceleration (hip), shank+foot inertia x tibia angular acceleration (knee). Gravity on the
   leg and Coriolis terms are ignored (< 10 % of the stance moments at this speed).
4. Springs in parallel with the motors: tau_motor = tau_required - tau_spring, spring torque
   -k (q - q0), bilateral or unilateral (tendon: one direction only). (k, q0) per joint chosen on a
   grid to minimise the peak motor power over the whole stride, subject to peak motor torque not
   above the no-spring peak. Cases: (a) none, (b) ankle (Achilles), (c) ankle + knee + hip.
5. Mass scenarios: every mass scaled by M/11.36, geometry fixed -> the SLIP solution is unchanged
   (dimensionless stiffness fixed), GRF, torques, powers and swing inertia all scale with M.
6. Tail: pitch correction from angular momentum conservation, actuator from a rest-to-rest swing
   within one flight phase; yaw the same about the vertical axis.
7. Structure: toe III/IV bending at the MTP and shin (tibia tube) bending from the peak leg force.

Usage: .venv-sim/bin/python modeling/spec_40kmh.py [--json out.json] [--md]
"""
import argparse
from dataclasses import dataclass, replace
import json
from math import atan2, cos, pi, sin, sqrt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_r02 as D  # noqa: E402

G = D.G
DT = 1e-4
T_BLEND = .02   # s, lift-off / touchdown transition window (path smoothing)
V40 = 11.1


@dataclass
class Gait:
    v: float = V40
    th_td_deg: float = 35.     # touchdown leg angle from vertical (ostrich fast running ~30-40 deg, see spec)
    duty: float = .25          # stance / STRIDE period (per-stride duty factor as in design_r02 and the ostrich data)
    reach_frac: float = .999   # hip->MTP distance at take-off as a fraction of the IK reach limit
    clearance_frac: float = .12
    kappa: float = .25         # swing radial end-rate fraction of the SLIP lift-off/touchdown rate


# ------------------------------------------------------------------ 1. SLIP
def slip_stance(M, k, L0, th, vx, vz):
    """Integrate one SLIP stance (RK4). Returns arrays t, x, z (CoM relative to the COP), F (leg force)."""
    def f(s):
        x, z, vx_, vz_ = s
        L = sqrt(x*x + z*z)
        F = k*(L0 - L)
        return np.array([vx_, vz_, F*x/(L*M), F*z/(L*M) - G])
    s = np.array([-L0*sin(th), L0*cos(th), vx, vz])
    out, t = [s.copy()], 0.
    while True:
        k1 = f(s); k2 = f(s + DT/2*k1); k3 = f(s + DT/2*k2); k4 = f(s + DT*k3)
        s = s + DT/6*(k1 + 2*k2 + 2*k3 + k4)
        t += DT
        out.append(s.copy())
        if t > DT and sqrt(s[0]**2 + s[1]**2) >= L0:
            break
        if s[1] <= 0 or t > 1.:
            return None
    a = np.array(out)
    L = np.hypot(a[:, 0], a[:, 1])
    return np.arange(len(a))*DT, a, k*(L0 - L)


def slip_fixed_point(M, L0, th, v, t_flight):
    """Spring stiffness k for which the stance is mirror-symmetric (take-off angle = -th) given the
    touchdown vertical speed from a ballistic flight of t_flight. Bisection on k."""
    vz = -G*t_flight/2
    def takeoff_angle(k):
        r = slip_stance(M, k, L0, th, v, vz)
        if r is None:
            return -pi
        x, z = r[1][-1, 0], r[1][-1, 1]
        return atan2(x, z)              # + forward of the COP at take-off
    lo, hi = 2e3, 4e5
    for _ in range(60):
        mid = sqrt(lo*hi)
        if takeoff_angle(mid) > th:     # too soft: leg sweeps past the mirror angle
            lo = mid
        else:
            hi = mid
    k = sqrt(lo*hi)
    t, a, F = slip_stance(M, k, L0, th, v, vz)
    return k, t, a, F, vz


def slip(p, g: Gait):
    M = D.masses(p)[1]
    th = np.radians(g.th_td_deg)
    # L0 = hip -> COP; the hip -> MTP distance at take-off (MTP cop_frac*toe3 behind the COP) must stay inside the IK reach
    R, c = g.reach_frac*D.REACH*D.leg_len(p), p.cop_frac*p.toe3
    L0 = -c*sin(th) + sqrt(R*R - c*c*cos(th)**2)
    # contact length ~ 2 L0 sin th at speed v gives the stance time; per-stride duty d gives the flight per step
    tc0 = 2*L0*sin(th)/g.v
    t_flight = tc0*(1/(2*g.duty) - 1)
    for _ in range(4):                 # refine the flight time with the actual stance time
        k, t, a, F, vz = slip_fixed_point(M, L0, th, g.v, t_flight)
        tc = t[-1]
        t_flight = tc*(1/(2*g.duty) - 1)
    step = tc + t_flight
    T = 2*step
    L = np.hypot(a[:, 0], a[:, 1])
    dL = float(L0 - L.min())
    Fpk = float(F.max())
    vz_to = float(a[-1, 3])
    return dict(mass_kg=round(M, 2), v_mps=g.v, L0_m=round(L0, 3), touchdown_angle_deg=g.th_td_deg,
                hip_height_td_m=round(L0*cos(th), 3), contact_s=round(tc, 4), flight_s=round(t_flight, 4),
                duty_per_stride=round(tc/T, 3), step_hz=round(1/step, 2), stride_hz=round(1/T, 2), stride_m=round(g.v*T, 2),
                contact_length_m=round(float(a[-1, 0] - a[0, 0]), 3), apex_rise_m=round(vz**2/(2*G), 3),
                peak_grf_bw=round(Fpk/(M*G), 2), peak_grf_n=round(Fpk), leg_compression_m=round(dL, 3),
                leg_stiffness_kn_m=round(k/1e3, 1), dimensionless_stiffness=round(k*L0/(M*G), 1),
                vertical_impulse_ok=round(abs(vz_to + vz)/abs(vz), 3),   # 0 = perfectly periodic
                froude_u=round(g.v/sqrt(G*L0*cos(th)), 2), froude_number=round(g.v**2/(G*L0*cos(th)), 1),
                _raw=(k, t, a, F, tc, t_flight))


# ------------------------------------------------------------------ 2-3. joints over one stride
def joint_traj(p, g: Gait, S):
    """Hip-relative foot (MTP) path over a stride from the SLIP stance, IK to (hip, knee) every DT."""
    k, t, a, F, tc, tf = S['_raw']
    T = 2*(tc + tf)
    ts = T - tc
    cop_off = p.cop_frac*p.toe3
    # stance: MTP relative to the hip = -(CoM relative to COP) - cop offset
    xs_st, zs_st = -a[:, 0] - cop_off, -a[:, 1]
    # swing in polar coordinates about the hip (angle from the downward vertical, + behind): cubic Hermite in
    # angle with the stance angular velocity matched at lift-off and touchdown (leg retraction overshoot, no
    # jump in hip rate), radius shortened by a (1-cos) profile for clearance -> the path never leaves the reach
    phi = np.arctan2(-xs_st, -zs_st)
    r0 = np.hypot(xs_st, zs_st)
    p0, p1, w0, w1 = phi[-1], phi[0], (phi[-1] - phi[-2])/DT, (phi[1] - phi[0])/DT
    rd0, rd1 = (r0[-1] - r0[-2])/DT, (r0[1] - r0[0])/DT          # radial rates: extending at lift-off, compressing at touchdown
    n = int(round(ts/DT))
    s_ = np.arange(1, n + 1)/n
    h00, h10, h01, h11 = 2*s_**3 - 3*s_**2 + 1, s_**3 - 2*s_**2 + s_, -2*s_**3 + 3*s_**2, s_**3 - s_**2
    ph = h00*p0 + h10*w0*ts + h01*p1 + h11*w1*ts
    h = -0.5*(zs_st[0] + zs_st[-1])
    # the SLIP leg extends at ~7 m/s at lift-off; the joints must stop that within the transition window
    # (T_BLEND): the swing radius uses reduced end rates (kappa) so it fits the reach, and the whole path
    # is smoothed with a T_BLEND moving average so the lift-off/touchdown transitions have finite
    # accelerations (= actuator/transition bandwidth of the sizing, an explicit assumption)
    rr = h00*r0[-1] + h10*g.kappa*rd0*ts + h01*r0[0] + h11*g.kappa*rd1*ts - g.clearance_frac*h*(1 - np.cos(2*pi*s_))/2
    xs_sw, zs_sw = -rr*np.sin(ph), -rr*np.cos(ph)
    xs, zs = np.concatenate([xs_st, xs_sw]), np.concatenate([zs_st, zs_sw])
    w = int(round(T_BLEND/DT))
    ker = np.ones(w)/w
    xs = np.convolve(np.concatenate([xs[-w:], xs, xs[:w]]), ker, 'same')[w:-w]
    zs = np.convolve(np.concatenate([zs[-w:], zs, zs[:w]]), ker, 'same')[w:-w]
    q = np.array([D.leg_ik(p, x, z) or (np.nan, np.nan) for x, z in zip(xs, zs)], float)
    if np.isnan(q).any():
        raise RuntimeError('foot path leaves the IK reach: lower reach_frac or th_td')
    Fst = np.concatenate([F, np.zeros(n)])
    cop = np.concatenate([np.zeros((len(a), 2)), np.full((n, 2), np.nan)])
    cop[:len(a), 0], cop[:len(a), 1] = -a[:, 0], -a[:, 1]      # COP relative to the hip
    return dict(t=np.arange(len(xs))*DT, tc=tc, T=T, q=q, F=Fst, cop=cop, n_st=len(a))


def joint_loads(p, g: Gait, S):
    """Required joint torques (about +y: positive = pitch-back on the distal segment), relative joint
    angles about +y and their rates, over one stride. Joints: hip, knee, ankle (motors) and mtp (passive)."""
    J = joint_traj(p, g, S)
    q, F, cop, n_st = J['q'], J['F'], J['cop'], J['n_st']
    hip, knee = q[:, 0], q[:, 1]
    # relative joint angles about +y (see design_r02.points: hip + = thigh forward = -y rotation)
    ang = {'hip': -hip, 'knee': knee, 'ankle': -knee}
    vel = {k_: np.gradient(v, DT) for k_, v in ang.items()}
    acc = {k_: np.gradient(v, DT) for k_, v in vel.items()}
    lm = p.tube_density
    L = D.leg_len(p)
    I_leg = (lm*p.femur + p.transmission)*p.femur**2/3 + lm*p.tibia*p.femur**2 + (lm*p.meta + p.foot_toes)*(.7*L)**2
    I_shank = lm*p.tibia*p.tibia**2/3 + (lm*p.meta + p.foot_toes)*p.tibia**2
    N = len(hip)
    tau = {k_: np.zeros(N) for k_ in ('hip', 'knee', 'ankle', 'mtp')}
    for i in range(N):
        K, A, Mp = D.points(p, hip[i], knee[i])
        if i < n_st:
            c = cop[i]
            n = -c/np.linalg.norm(c)                      # COP -> hip, force on the leg
            fx, fz = n[0]*F[i], n[1]*F[i]
            for name, j in (('knee', K), ('ankle', A), ('mtp', Mp)):
                rx, rz = c[0] - j[0], c[1] - j[1]
                My = rz*fx - rx*fz                        # moment of the leg force about the joint, +y
                tau[name][i] = -My                        # the joint must hold it (distal chain quasi-static)
        else:
            tau['hip'][i] = I_leg*acc['hip'][i]
            tau['knee'][i] = I_shank*(acc['hip'][i] + acc['knee'][i])
    return dict(J=J, ang=ang, vel=vel, tau=tau, I_leg=round(I_leg, 4), I_shank=round(I_shank, 4))


# ------------------------------------------------------------------ 4. springs
def spring_search(q, qd, tau, T):
    """(k, q0, mode) minimising peak motor power |(tau - tau_s) qd| with peak motor torque <= no-spring peak."""
    best = (float(np.abs(tau*qd).max()), 0., 0., 'none')
    tau_pk = np.abs(tau).max()
    for mode in ('bi', 'uni+', 'uni-'):
        for k in np.linspace(5, 400, 80):
            for q0 in np.linspace(q.min() - .3, q.max() + .3, 61):
                d = q - q0
                if mode == 'uni+':
                    d = np.maximum(d, 0)
                elif mode == 'uni-':
                    d = np.minimum(d, 0)
                tm = tau + k*d
                if np.abs(tm).max() > tau_pk + 1e-9:
                    continue
                ppk = float(np.abs(tm*qd).max())
                if ppk < best[0]:
                    best = (ppk, float(k), float(q0), mode)
    return best


def clutched_ankle(p, g: Gait, S, Lo=None, swing_fold_rad=None):
    """Ankle spring engaged only in stance (clutch, BirdBot / ostrich engage-disengage ligament; evidence 86 §9).
    Same (k, q0) search as case (b) but the spring torque is zero in swing. Also reports the torque needed to hold a
    swing fold of `swing_fold_rad` (intertarsal included angle) against an always-engaged spring vs the clutched one."""
    Lo = Lo or joint_loads(p, g, S)
    T, n_st = Lo['J']['T'], Lo['J']['n_st']
    tau, q, qd = Lo['tau']['ankle'], Lo['ang']['ankle'], Lo['vel']['ankle']
    stance = np.arange(len(q)) < n_st
    best = (float(np.abs(tau*qd).max()), 0., 0.)
    tau_pk = np.abs(tau).max()
    for k in np.linspace(5, 400, 80):
        for q0 in np.linspace(q.min() - .3, q.max() + .3, 61):
            tm = tau + k*np.minimum(q - q0, 0)*stance
            if np.abs(tm).max() > tau_pk + 1e-9:
                continue
            ppk = float(np.abs(tm*qd).max())
            if ppk < best[0]:
                best = (ppk, float(k), float(q0))
    _, k, q0 = best
    tm = tau + k*np.minimum(q - q0, 0)*stance
    out = dict(**stats(tm, qd, T), spring=dict(k_nm_per_rad=round(k, 1), q0_rad=round(q0, 3), mode='uni- clutched (stance only)'))
    tb = tau + k*np.minimum(q - q0, 0)
    out['same_spring_unclutched'] = stats(tb, qd, T)
    if swing_fold_rad is not None:  # static holding torque in swing at a fold angle (relative ankle coordinate: -knee convention)
        a = -(np.pi - swing_fold_rad)  # joint angle for that included angle, spec convention ang['ankle'] = -knee
        out['swing_fold_hold_torque_nm'] = dict(included_deg=round(float(np.degrees(swing_fold_rad))), unclutched=round(float(abs(k*min(a - q0, 0))), 1), clutched=0.)
    return out


def stats(tau, qd, T):
    P = tau*qd
    return dict(peak_torque_nm=round(float(np.abs(tau).max()), 1), peak_speed_rad_s=round(float(np.abs(qd).max()), 1),
                peak_power_w=round(float(np.abs(P).max())), pos_work_per_stride_j=round(float(np.clip(P, 0, None).sum()*DT), 2),
                mean_pos_power_w=round(float(np.clip(P, 0, None).sum()*DT/T)))


def spring_cases(p, g: Gait, S):
    Lo = joint_loads(p, g, S)
    T = Lo['J']['T']
    cases, springs = {}, {}
    for case, sprung in (('a_no_spring', ()), ('b_achilles', ('ankle',)), ('c_ankle_knee_hip', ('ankle', 'knee', 'hip'))):
        per = {}
        for j in ('hip', 'knee', 'ankle'):
            tau, q, qd = Lo['tau'][j], Lo['ang'][j], Lo['vel'][j]
            if j in sprung:
                key = (j,)
                if key not in springs:
                    springs[key] = spring_search(q, qd, tau, T)
                _, k, q0, mode = springs[key]
                d = q - q0
                d = np.maximum(d, 0) if mode == 'uni+' else np.minimum(d, 0) if mode == 'uni-' else d
                tau = tau + k*d
                per[j] = dict(**stats(tau, qd, T), spring=dict(k_nm_per_rad=round(k, 1), q0_rad=round(q0, 3), mode=mode,
                                                                 energy_j=round(.5*k*float(np.abs(d).max())**2, 2)))
            else:
                per[j] = stats(tau, qd, T)
        per['mtp_passive'] = stats(Lo['tau']['mtp'], Lo['vel']['ankle']*0 + np.gradient(np.zeros_like(Lo['tau']['mtp']), DT), T)
        per['mtp_passive'] = dict(peak_torque_nm=round(float(np.abs(Lo['tau']['mtp']).max()), 1))
        per['sum_peak_power_w'] = sum(per[j]['peak_power_w'] for j in ('hip', 'knee', 'ankle'))
        per['sum_mean_pos_power_w_two_legs'] = 2*sum(per[j]['mean_pos_power_w'] for j in ('hip', 'knee', 'ankle'))
        cases[case] = per
    base = cases['a_no_spring']
    for c in cases.values():
        c['peak_power_reduction_pct'] = round(100*(1 - c['sum_peak_power_w']/base['sum_peak_power_w']), 1)
        c['mean_power_reduction_pct'] = round(100*(1 - c['sum_mean_pos_power_w_two_legs']/base['sum_mean_pos_power_w_two_legs']), 1)
    return cases, Lo


# ------------------------------------------------------------------ 5. mass scenarios
def mass_scenarios(p, g: Gait, S, cases, targets=(11.36, 8., 5.)):
    M0 = S['mass_kg']
    rows = []
    for M in targets:
        s = M/M0
        row = dict(mass_kg=M, scale=round(s, 3), peak_grf_n=round(S['peak_grf_n']*s), leg_stiffness_kn_m=round(S['leg_stiffness_kn_m']*s, 1))
        for case in ('a_no_spring', 'c_ankle_knee_hip'):
            for j in ('hip', 'knee', 'ankle'):
                c = cases[case][j]
                row[f'{case[0]}_{j}'] = dict(torque_nm=round(c['peak_torque_nm']*s, 1), speed_rad_s=c['peak_speed_rad_s'],
                                             power_w=round(c['peak_power_w']*s))
            row[f'{case[0]}_mean_pos_power_w'] = round(cases[case]['sum_mean_pos_power_w_two_legs']*s)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ 6. tail
def tail(p, S, r=None, motor_classes=(17., 36., 60.)):
    """Pitch correction from angular momentum conservation (body + tail isolated in flight):
    dth_body = r/(1+r) * dth_tail, r = I_tail / I_body(no tail). Actuator: what a motor of peak torque
    tau can do over the full +-0.8 rad stroke (bang-bang, rest to rest, gravity on the tail at the worst
    angle subtracted), the angular momentum it can put into the tail and the body pitch rate that cancels.
    Disturbance scale for comparison: a touchdown with the leg force 0.05 m off the CoM for one stance."""
    st = D.stand(p)
    r = r or st['tail_to_body_pitch']
    rng = D.LIMITS['tail_pitch'][1] - D.LIMITS['tail_pitch'][0]
    I_tail, I_body = st['tail_inertia_base'], st['body_pitch_inertia_no_tail']
    grav = G*(p.tail_struct*p.tail_len/2 + p.tail_tip*p.tail_len)
    rows = []
    for tau in motor_classes:
        alpha = (tau - grav)/I_tail
        t_full = 2*sqrt(rng/alpha)                       # bang-bang full stroke
        w = sqrt(rng*alpha)                              # peak rate at mid-stroke
        rows.append(dict(motor_peak_nm=tau, full_stroke_s=round(t_full, 3), peak_rate_rad_s=round(w, 1),
                         peak_power_w=round(tau*w), momentum_capacity_nms=round(I_tail*w, 2),
                         body_rate_cancelled_rad_s=round(I_tail*w/I_body, 1),
                         strokes_per_flight=round(S['flight_s']/t_full, 2)))
    dist = S['peak_grf_n']*.05*2*S['contact_s']/pi     # half-sine force x 5 cm lever over one stance
    return dict(inertia_ratio=r, range_rad=rng, body_correction_rad=round(r/(1 + r)*rng, 3),
                body_correction_rad_at_ratio_0p68=round(.68/1.68*rng, 3), tail_inertia=I_tail, body_inertia_no_tail=I_body,
                touchdown_5cm_error_impulse_nms=round(dist, 2), body_rate_from_it_rad_s=round(dist/I_body, 1),
                yaw_note='tail yaw inertia = pitch inertia (rod); body yaw inertia taken equal to the pitch value (approximation)',
                motors=rows)


# ------------------------------------------------------------------ 7. structure
def structure(p, S, cases):
    F = S['peak_grf_n']
    share = {'III': .70, 'IV': .30}   # running share (assumption; standing measured 55 / 13 %, rest on the pad)
    toes = {}
    for d, sh in share.items():
        L = p.toe3 if d == 'III' else p.toe4
        Ft = F*sh
        M_base = Ft*L/2                              # distributed load along the toe -> cantilever moment at the MTP
        r_o, wall = .008, .0015                       # 16 mm carbon/alu tube toe (assumption)
        I = pi/4*(r_o**4 - (r_o - wall)**4)
        toes[d] = dict(force_n=round(Ft), moment_at_mtp_nm=round(M_base, 1), tube_od_mm=2*r_o*1e3,
                       bending_stress_mpa=round(M_base*r_o/I/1e6), k_toe_for_0p35rad=round(M_base/.35, 1))
    tau_knee = cases['a_no_spring']['knee']['peak_torque_nm']
    tau_ankle = cases['a_no_spring']['ankle']['peak_torque_nm']
    r_o, wall = .0125, .0015                          # 25 x 1.5 mm carbon tube shin (assumption)
    I = pi/4*(r_o**4 - (r_o - wall)**4)
    M_shin = max(tau_knee, tau_ankle)                # bending moment along the tibia varies linearly between the joints
    return dict(toes=toes, shin=dict(axial_n=F, max_bending_nm=round(M_shin, 1), tube='25 x 1.5 mm',
                                     bending_stress_mpa=round(M_shin*r_o/I/1e6), axial_stress_mpa=round(F/(pi*(r_o**2 - (r_o - wall)**2))/1e6, 1),
                                     safety_factor_at_600mpa=round(600/(M_shin*r_o/I/1e6), 1)))


# ------------------------------------------------------------------ 8. actuator classes
CLASSES = [  # name, peak N·m, no-load rad/s, assumed peak mechanical W (2x catalogue rated), mass kg  (catalogue: spec 7)
    ('A  QDD ~0.5 kg (AK80-9 22 N·m / Go1 23.7 / Mini Cheetah 17)', 22, 40, 500, .5),
    ('B  QDD ~1 kg (AK10-9 48 N·m 33 rad/s / RobStride 03 60 N·m 20 rad/s)', 48, 33, 900, .96),
    ('C  QDD ~1.4 kg (RobStride 04 120 N·m 21 rad/s, rated 700 W)', 120, 21, 1400, 1.42),
    ('D  humanoid 2-3 kg (Unitree H1 knee 360 N·m; Cassie 195 N·m 8.5 rad/s)', 360, 12, 4000, 2.5),
]


def actuator_match(reqs):
    out = {}
    for j, r in reqs.items():
        fits = [c[0] for c in CLASSES if r['torque_nm'] <= c[1] and r['speed_rad_s'] <= c[2] and r['power_w'] <= c[3]]
        out[j] = dict(need=r, exists=bool(fits), smallest_class=fits[0] if fits else None)
    return out


# ------------------------------------------------------------------ full actuator budget (all axes)
HIP_Y, FOOT_COP_Y = .10, .02   # m: hip joint lateral offset (R-02 left/right hip roll links at +-0.1 m), assumed lateral
                               # COP offset under the foot; roll loads are quasi-static estimates (no lateral dynamics solved)
RATED_FRACTION = .4            # assumed continuous (rated) torque = 40 % of peak for the QDD classes (catalogue ratios ~0.35-0.45)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def full_budget(p, g, S, Lo, cases, masses=(11.36, 8., 5.)):
    """Actuator mass for whole-robot configurations (every active axis), per total mass, springs (c).
    Pitch joints from the stride loads (peak torque/speed/power and RMS torque for heat); hip roll = peak GRF x HIP_Y,
    ankle roll = peak GRF x FOOT_COP_Y (both at an assumed 5 rad/s); tail axes = spec §8 36 N·m / 12 rad/s / 400 W at
    11.36 kg, scaled with mass. Coupled knee = one motor driving knee + ankle through a 1:1 tendon (horse reciprocal
    apparatus), torque tau_knee - tau_ankle at the same instant, springs (c) on both."""
    T = Lo['J']['T']
    def spr(j, q):
        sp = cases['c_ankle_knee_hip'][j]['spring']; k, q0, mode = sp['k_nm_per_rad'], sp['q0_rad'], sp['mode']
        d = q - q0
        d = np.maximum(d, 0) if mode == 'uni+' else np.minimum(d, 0) if mode == 'uni-' else d
        return k*d
    tau = {j: Lo['tau'][j] + spr(j, Lo['ang'][j]) for j in ('hip', 'knee', 'ankle')}
    vel = {j: Lo['vel'][j] for j in ('hip', 'knee', 'ankle')}
    tau['knee+ankle'], vel['knee+ankle'] = tau['knee'] - tau['ankle'], vel['knee']
    base = {j: dict(stats(tau[j], vel[j], T), rms_torque_nm=round(rms(tau[j]), 1)) for j in tau}
    configs = {  # per side axes, plus tail axes
        '12 axes independent (current T1/R-02 model)': (['hip_roll', 'hip', 'knee', 'ankle', 'ankle_roll'], 2),
        '10 axes (AGENTS base: no ankle roll)': (['hip_roll', 'hip', 'knee', 'ankle'], 2),
        '10 axes = 12 with knee-ankle coupling': (['hip_roll', 'hip', 'knee+ankle', 'ankle_roll'], 2),
        '8 axes = 10 with knee-ankle coupling': (['hip_roll', 'hip', 'knee+ankle'], 2),
        '7 axes = 8 with a single tail axis': (['hip_roll', 'hip', 'knee+ankle'], 1),
    }
    out = []
    for M in masses:
        sc = M/S['mass_kg']
        need = {j: dict(torque_nm=b['peak_torque_nm']*sc, speed_rad_s=b['peak_speed_rad_s'], power_w=b['peak_power_w']*sc,
                        rms_nm=b['rms_torque_nm']*sc) for j, b in base.items()}
        F = S['peak_grf_n']*sc
        need['hip_roll'] = dict(torque_nm=F*HIP_Y, speed_rad_s=5., power_w=F*HIP_Y*5., rms_nm=F*HIP_Y*.45)
        need['ankle_roll'] = dict(torque_nm=F*FOOT_COP_Y, speed_rad_s=5., power_w=F*FOOT_COP_Y*5., rms_nm=F*FOOT_COP_Y*.45)
        need['tail'] = dict(torque_nm=36.*M/11.36, speed_rad_s=12., power_w=400.*M/11.36, rms_nm=36.*M/11.36*.3)
        pick = {}
        for j, r in need.items():
            fits = [c for c in CLASSES if r['torque_nm'] <= c[1] and r['speed_rad_s'] <= c[2] and r['power_w'] <= c[3]
                    and r['rms_nm'] <= RATED_FRACTION*c[1]]
            pick[j] = (fits[0][0][0], fits[0][4]) if fits else (None, None)
        rows = {}
        for name, (axes, n_tail) in configs.items():
            parts = [(a, pick[a]) for a in axes]*2 + [('tail', pick['tail'])]*n_tail
            ok = all(c is not None for _, (c, _) in parts)
            mass = round(sum(m for _, (_, m) in parts), 2) if ok else None
            rows[name] = dict(classes={a: pick[a][0] for a in axes} | {'tail': pick['tail'][0]}, n_actuators=len(parts),
                              actuator_mass_kg=mass, fraction_of_total=round(mass/M, 2) if ok else None)
        out.append(dict(mass_kg=M, needs={j: {k: round(v, 1) for k, v in r.items()} for j, r in need.items()}, configs=rows))
    return out


# ------------------------------------------------------------------ design-point scan
def solve(p, g):
    """SLIP + joint loads; the take-off leg length is lowered until the swing (radial overshoot of the
    matched-rate Hermite = the leg keeps extending briefly after lift-off) stays inside the IK reach."""
    for rf in np.arange(g.reach_frac, .80, -.01):
        g = replace(g, reach_frac=float(rf))
        S = slip(p, g)
        try:
            return g, S, joint_loads(p, g, S)
        except RuntimeError:
            continue
    raise RuntimeError('no reachable stride')


def scan(p, ths=(30., 35., 40.), duties=(.30, .25, .20)):
    rows = []
    for th in ths:
        for d in duties:
            g, S, Lo = solve(p, Gait(th_td_deg=th, duty=d))
            T = Lo['J']['T']
            st = {j: stats(Lo['tau'][j], Lo['vel'][j], T) for j in ('hip', 'knee', 'ankle')}
            rows.append(dict(th_td_deg=th, duty=d, L0_m=S['L0_m'], contact_s=S['contact_s'], stride_hz=S['stride_hz'], stride_m=S['stride_m'],
                             peak_grf_bw=S['peak_grf_bw'], leg_stiffness_kn_m=S['leg_stiffness_kn_m'],
                             hip=st['hip'], knee=st['knee'], ankle=st['ankle'],
                             sum_peak_power_w=sum(x['peak_power_w'] for x in st.values()),
                             mean_pos_power_w_two_legs=2*sum(x['mean_pos_power_w'] for x in st.values())))
    return rows


# ------------------------------------------------------------------ report
def compute(p=None, g=None):
    p = p or D.Params()
    g, S, _ = solve(p, g or Gait())
    cases, Lo = spring_cases(p, g, S)
    rows = mass_scenarios(p, g, S, cases)
    st = structure(p, S, cases)
    tl = tail(p, S)
    act = {}
    for row in rows:
        for case in ('a', 'c'):
            act[f"{row['mass_kg']}kg_{case}"] = actuator_match({j: row[f'{case}_{j}'] for j in ('hip', 'knee', 'ankle')})
    # distal-light leg (ostrich principle: mass proximal): what a lighter shank/foot buys the hip
    pl = replace(p, tube_density=.35, foot_toes=.06)
    gl, Sl, Lol = solve(pl, g)
    light = {j: stats(Lol['tau'][j], Lol['vel'][j], Lol['J']['T']) for j in ('hip', 'knee', 'ankle')}
    light['mass_kg'] = Sl['mass_kg']
    light['I_leg_about_hip'] = Lol['I_leg']
    clutch = clutched_ankle(p, g, S, swing_fold_rad=np.radians(65.))
    S = {k: v for k, v in S.items() if not k.startswith('_')}
    return dict(clutched_ankle_11p36kg=clutch, gait=vars(g), scan=scan(p), slip=S, distal_light_leg=light, joints_11p36kg=cases, mass_scenarios=rows, actuators=act, tail=tl, structure=st,
                inertia=dict(I_leg_about_hip=Lo['I_leg'], I_shank_about_knee=Lo['I_shank']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json')
    ap.add_argument('--th', type=float, default=35.)
    ap.add_argument('--duty', type=float, default=.25)
    a = ap.parse_args()
    r = compute(g=Gait(th_td_deg=a.th, duty=a.duty))
    print(json.dumps(r, indent=1, ensure_ascii=False, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
