"""T1 running environment: R-02 geometry, virtual actuator spec (c) at 5 kg, tail pitch/yaw in the action space.

Tier T1 of docs/design/40kmh-spec.md (§9). Nothing here is the current robot's actuator: every result from this
environment is a T1 (virtual actuator) result and is reported with the spec of the first row of the tier table.

Model: sim/raptor_r02.xml (12 active joints, 12 passive toe joints). Every body mass/inertia is scaled uniformly
from the 11.36 kg design to `mass` kg (default 5, geometry and toe springs fixed). Parallel joint springs of
case (c) (hip unilateral +, knee bilateral, ankle unilateral -) are fixed tendons with a dead-band spring length;
their stiffness scales with mass. `springs='b'` keeps only the Achilles (ankle) spring, 'none' removes all.
Actuator (per joint type, torque N·m / speed rad/s / power W): hip 35/27/350, knee 28/27/225, ankle 8/27/90,
tail 36/12/400; the roll axes take their pitch axis class (assumption). It is a PD position servo evaluated by
MuJoCo at 1 ms (implicit velocity term) whose force range is clamped every physics step to
|tau| <= min(tau_max, P_max/|qd|) and to zero in the driving direction beyond the speed limit.

Policy: 50 Hz position targets for the 12 active joints (10 when the tail is locked). Observation: IMU gyro and
gravity direction, body linear velocity (T1 assumes a state estimator), joint positions/velocities, last action,
command, stride clock whose rate follows the commanded speed. Terrain is flat; `level` in [0, 1] scales the
disturbance curriculum instead: pitch impulses, yaw impulses, foot trips during swing, touchdown impulses and the
one lateral shove of RaptorEnv. Angular momentum of the whole body, the tail and the legs about the CoM is logged.
"""
import json
from pathlib import Path
import sys

import gymnasium as gym
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from raptor_env import RaptorEnv, CONTROL_DT, SIM, joint_type  # noqa: E402
import terrain as tr  # noqa: E402
from realism import Realism, load_config  # noqa: E402

ROOT = SIM.parent
R02_MODEL = SIM/'raptor_r02.xml'
# model rev 3 (stage 0, user decision 2026-10-31, evidence 186-187): R-02 + an active hip yaw axis per leg
# (sim/build_r03.py) and the vendor modules / transmissions / drive masses of one axis config (rev3_b.json)
R03_MODEL = SIM/'raptor_r03.xml'
REV3_CONFIG = HERE/'rev3_b.json'
R02_YAML = ROOT/'src/raptor_description/config/r02_design.yaml'
DESIGN_MASS = 11.36
PHYSICS_DT = .001
# T1 actuator limits (N·m, rad/s, W): spec §9 (5 kg + (c), 1.3x margin); tail: spec §8 proposal.
# Ankle pitch torque 22 N·m (the A-class value, spec §6) instead of the §9 8 N·m: standing still with the case-(c)
# ankle spring needs 9-11 N·m of motor torque against the spring preload (its rest angle is the running touchdown
# angle; deeper rest angles lose the running benefit: q0 -1.3 rad -> 11.5 N·m/159 W at 5 kg), a load case the spec
# did not check (measured 2026-09-29, pose scan). Power and speed limits stay at §9; running needs 5.8 N·m there.
ACTUATOR_T1 = {'hip_pitch': (35., 27., 350.), 'knee_pitch': (28., 27., 225.), 'ankle_pitch': (22., 27., 90.),
               'hip_roll': (35., 27., 350.), 'ankle_roll': (8., 27., 90.), 'hip_yaw': (35., 27., 350.),
               'tail_pitch': (36., 12., 400.), 'tail_yaw': (36., 12., 400.)}
# T1 nominal pose (hip, knee, ankle), toes flat: hip + knee + ankle = -0.5271 (foot mount). Taller than the R-02
# crouch (torso 0.71 m vs 0.58 m) because the springs rest at the running touchdown angles (the crouch would preload
# the ankle spring with 37 N·m). No flat-toe pose stands passively: the zero-action hold topples backward for hip
# <= -0.40 and forward for hip >= -0.35 (pose scan 2026-09-29), so balance is the policy's job; this pose is the
# boundary and gives ~2 s before the fall.
NOMINAL_T1 = (-.375, 1.50, -1.652)
# Parallel springs at the design mass (N·m/rad, rest angle rad, mode), spec §4: (c) = all three, (b) = Achilles only.
# Joint coordinates are the MJCF ones (hip +: thigh back, knee +: flexion, ankle = -knee in the nominal pantograph).
SPRINGS = {'none': {},
           'b': {'ankle_pitch': (130., -1.215, 'uni-')},
           'c': {'hip_pitch': (295., .041, 'uni+'), 'knee_pitch': (90., 1.395, 'bi'), 'ankle_pitch': (130., -1.215, 'uni-')}}
KP_ERR = .1       # rad of position error that saturates the torque limit (kp = tau_max / KP_ERR)
KD_POLE = 45.     # 1/s, kd = kp / KD_POLE
SCALE_RUN = {'hip_roll': .3, 'hip_yaw': .3, 'hip_pitch': 1., 'knee_pitch': 1., 'ankle_pitch': 1., 'ankle_roll': .3,
             'tail_yaw': .8, 'tail_pitch': .8}
RUN_SPEED = 3.    # m/s command above which flight is rewarded
DISTURB_ITEMS = ('pitch', 'yaw', 'trip', 'touchdown', 'push')

# reward weights per second (x CONTROL_DT per step)
# gait: clock-conditioned contact schedule (Siekmann 2021 style): left stance while phase in [0, d), right while in
# [0.5, 0.5+d), d = GAIT_DUTY at running speed. Added 2026-09-30 after the v2/v4 policies converged to a bound
# (left/right touchdown phase 0.85, 4.4 Hz vs the 2.4 Hz clock) - reviews: evidence 86 review_*.md. flight/air_time
# rewards are off by default since they favoured the bound (both reviews); the schedule already implies flight.
GAIT_DUTY, GAIT_EDGE = .28, .03
# joint flexion (ostrich rules scaled to R-02, evidence 86 §9; reviews review_*_kinematics.md): per leg, on the leg's
# clock phase. Swing progress s = (phi - d)/(1 - d). fold: intertarsal ~65 deg at s 0.30-0.60 (ostrich 45 deg);
# extend: ~135 deg at s 0.80-0.98 before touchdown (ostrich 168 deg not imposed: CoM lies behind the hip); both paid
# only while the foot is actually unloaded. static: stance ankle within 12 deg of its touchdown angle over the middle
# 15-85 % of the stance window (no contact there counts as a full violation). retract: world thigh angular velocity
# ~+1.7 rad/s (backward) at s 0.75-0.95. All off by default (weight 0).
# leg_sym: left/right symmetry over time. Each leg's hip, knee and ankle angle and its fold/extend/retract scores are
# low-pass filtered (tau 0.6 s, > 1 stride); for a symmetric gait the filtered values match because the legs only differ
# by half a stride. Penalty = sum of squared left-right differences (angles in rad, scores x2). Added after v6c folded
# the left ankle to 52 deg while the right stayed at 90 deg (the per-leg shaping terms are averaged over the legs).
SYM_TAU = .6
# arch8: the 8-axis hybrid leg from the linkage review (evidence 86 §16, spec §6-2): per leg hip pitch (class B, direct),
# knee driven by a class-B motor mounted on the upper femur through the optimised four-bar (knee_linkage_study.json,
# ratio 1.0-2.05), ankle pitch coupled 1:1 to the knee, ankle roll locked, hip roll class A with a 1.7:1 reduction, tail
# yaw/pitch class B. Motor envelope per class: full torque to half the no-load speed, linear drop to zero at the
# no-load speed, |tau w| <= P (assumed shape); braking torque up to the class peak. Linkage/gear efficiency 0.9.
MOTOR_CLASSES = {'A': (22., 40., 500.), 'B': (48., 33., 900.), 'C': (120., 21., 1400.)}
# vendor modules registered by use_modules() (model rev 3): T-N curve (rad/s, N·m at the module output), rated torque,
# output-side rotor inertia
MODULE_CURVES, MOTOR_CONT, MODULE_J = {}, {}, {}
ARCH8 = {'hip_pitch': ('B', 1.), 'knee_pitch': ('B', 'fourbar'), 'hip_roll': ('A', 1.7), 'tail_yaw': ('B', 1.), 'tail_pitch': ('B', 1.)}
ARCH8_FOURBAR = dict(a=.083, b=.194, c=.102, beta_deg=69., branch=1.)
LINK_ETA = .9
# 'ratchet' spring engagement (springs_override, opt-in): one-way clutch in series with the unilateral spring. The pawl is
# lifted when the foot unloads, drops once the swing joint has come back RATCHET_REARM rad from its swing peak (or at the
# latest when the foot loads), and from then on the spring's rest angle follows the most extended angle reached, rounded
# up to the next tooth (RATCHET_PITCH rad). It always engages at zero deflection, so it cannot inject energy, and it has no
# fixed rest angle that the gait would have to reach (evidence: geometry_probe.py, 2026-10-13).
RATCHET_PITCH, RATCHET_REARM = .02, .05
# spring_store reward term (weight 0 unless given): rate (W) at which a ratchet spring takes up energy while its foot is
# loaded, credited up to SPRING_STORE_CAP J per stance. It pays for letting the stance leg yield into the spring.
SPRING_STORE_CAP = 6.
# heat (arch8 only): per motor, a low-pass (tau HEAT_TAU s) of (motor torque / continuous torque)^2 with the continuous
# torque = 40 % of the class peak - the same RMS criterion as the actuator budget. The term is the sum over motors of
# max(0, heat - 1): zero while every motor stays within its continuous rating. Added 2026-10-01 after the arch8 policy ran
# its knee motor at the torque limit 63 % of the time (RMS 2.2x the continuous rating).
HEAT_TAU = 3.
LP20 = PHYSICS_DT/(PHYSICS_DT + 1/(2*np.pi*20.))  # 20 Hz first-order low-pass per physics step (heat_lp)
# arch12: the 12-axis leg with the same motor classes (equal-condition comparison, 2026-10-01): arch8 plus an ankle pitch
# motor (class B, 1.5:1 cable drive) and an ankle roll motor (class A, direct), both mounted on the pelvis next to the
# hip (most favourable placement: no distal motor mass), ankle not coupled to the knee.
ARCH12_EXTRA = {'ankle_pitch': ('B', 1.5), 'ankle_roll': ('A', 1.)}
# model_rev 2 (2026-10-12, final design report 1.2 section 62; every item is also a separate RunEnv option and all are off
# at model_rev 1, which reproduces the earlier evidence bit for bit):
#   latch_fallback=False  the knee latch engages only in swing when the knee extends down through q0 after having been
#                         above it in that swing (one-way catch), and stays released otherwise (also at reset).
#                         The rev-1 fallback engaged the spring at touchdown at any deflection, which created energy:
#                         5,756 of 5,756 engagements of scr_a8_h2, 39-54 W. Use a 'uni+' latch spring with this option:
#                         it carries no load below q0, so an engagement adds exactly zero energy.
#   friction_dr='foot'    the friction draw scales the foot pads and the floor together (the pads have priority 1, so the
#                         rev-1 draw on the floor alone never reached the foot contacts)
#   self_collision=True   left leg / right leg / tail / torso contact pairs. Links joined through a geomless link stay
#                         excluded (thigh-torso through the hip-roll link, metatarsus-toes and toes of one foot through the
#                         foot link): their box shapes overlap by construction.
#   knee_band=KNEE_BAND   knee targets limited to the four-bar range where both transmission angles are >= 30 deg
#   start_contact=True    reset puts the lowest toe surface on the ground (rev 1 started 13 mm above it)
#   armature=True         reflected rotor inertia on the driven joints: T-Motor AK10-9 V3.0 (rotor 1002 g cm^2, 9:1) times
#                         the external ratio squared. ASSUMED for the class-A hip roll motor (no published value used) and
#                         for the knee, where the mean of N^2 over KNEE_BAND replaces the angle-dependent four-bar ratio.
KNEE_BAND = (.7159, 2.2454)
ROTOR_J_OUT = 81*1.002e-4   # kg m^2 at the module output
# real_mass: explicit part masses instead of uniform scaling of the 11.36 kg design table (2026-10-01; arch8e had been
# trained at 11.36 kg although its real build is 11.02 kg). Payload from design_r02.masses (torso frame 1.0, battery
# 0.6, compute 0.3, mast 0.2, pod 0.45 kg) on the torso; pelvis motors on the hip-roll link (they move with the hip
# mount); knee motor on the upper femur; distal-light leg; both tail motors on the tail-yaw link; tail structure and
# tip as R-02. Drive hardware (ASSUMED): arch8 parallelogram ankle rod 0.08 kg on the shin; arch12 cable drives for the
# two pelvis-mounted ankle motors 0.1 kg on the thigh + 0.1 kg on the shin per leg.
PAYLOAD_KG = 2.55
MOTOR_KG = {'A': .5, 'B': .96, 'C': 1.42}


def use_modules(path=REV3_CONFIG):
    """Model rev 3: read the axis config and register its vendor modules (docs/design/motor_candidates.json) as
    MOTOR_CLASSES entries (peak torque, end of the vendor 48 V T-N curve, peak mechanical power on that curve) with their
    curves, rated torques and output-side rotor inertias. Returns the config."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    mods = json.loads((path.parent/cfg['spec_file']).resolve().read_text())['modules']
    for ax in cfg['axes'].values():
        m = mods[ax['module']]
        w, t = np.asarray(m['tn_curve']['rpm'], float)*np.pi/30, np.asarray(m['tn_curve']['nm'], float)
        MODULE_CURVES[ax['module']] = (w, t)
        MOTOR_CLASSES[ax['module']] = (float(m['tau_peak']), float(w[-1]), float(np.max(w*t)))
        MOTOR_CONT[ax['module']], MODULE_J[ax['module']] = float(m['tau_cont']), float(m['j_out'])
    cfg['module_kg'] = {ax['module']: float(mods[ax['module']]['mass']) for ax in cfg['axes'].values()}
    return cfg


def rev3_motor_masses(cfg):
    """Masses (kg) the rev 3 axis config puts on the hip-roll link (pelvis-mounted modules move with the hip mount),
    the upper femur (knee module) and the tail-yaw link: module + drive hardware (drive_kg, ASSUMED)."""
    out = {'hip_roll_link': 0., 'thigh_link': 0., 'tail_yaw_link': 0.}
    for ax in cfg['axes'].values():
        out[ax['mount']] += cfg['module_kg'][ax['module']] + float(ax.get('drive_kg', 0.))
    return out


def fourbar_band(fb, lo=.1, hi=2.7, min_deg=30.):
    """Knee range of a four-bar where the loop closes and both the output (coupler-lever) and input (crank-coupler)
    transmission angles are >= min_deg (the rev 2 KNEE_BAND rule, final report 1.8 section F1)."""
    sys.path.insert(0, str(ROOT/'modeling'))
    from knee_linkage_study import fourbar_N, FEMUR
    q = np.linspace(lo, hi, 2601)
    beta = np.radians(fb['beta_deg'])
    phi, _N, mu_out, ok = fourbar_N(q, fb['a'], fb['b'], fb['c'], beta, fb['branch'])
    P = np.stack([fb['a']*np.cos(phi), fb['a']*np.sin(phi)], 1)
    Q = np.stack([fb['c']*np.cos(q+beta), -FEMUR+fb['c']*np.sin(q+beta)], 1)
    u = (Q-P)/fb['b']
    mu_in = np.degrees(np.arccos(np.clip(np.abs(np.sum(u*P, 1))/fb['a'], 0, 1)))
    good = ok & (mu_out >= min_deg) & (mu_in >= min_deg)
    if not good.any():
        raise ValueError('four-bar: no knee range with both transmission angles >= %g deg' % min_deg)
    i = np.flatnonzero(good)
    runs = np.split(i, np.flatnonzero(np.diff(i) > 1)+1)
    r = max(runs, key=len)
    return float(q[r[0]]), float(q[r[-1]])


def real_masses(m, arch12, motors=None):
    """Set the explicit part masses (see PAYLOAD_KG) on an R-02 MjModel at the design scale; returns the total.
    motors: {'hip_roll_link', 'thigh_link', 'tail_yaw_link': kg} replacing the class motors (model rev 3)."""
    def set_mass(b, new):
        f = new/m.body_mass[b]
        m.body_mass[b] = new; m.body_inertia[b] *= f
    def add_mass(b, dm, at, r=.03):
        """Add a part of mass dm at `at` (on the link's z axis), own inertia dm*r^2 (r ~ motor/part radius). The body
        inertia is about the NEW centre of mass: both the old body and the part are moved by the parallel-axis theorem
        (until 2026-10-01 the part's offset from the link origin was used and the old body was not moved: shin pitch
        inertia ~2.8x too large; UNI_AI gpt-6-sol review)."""
        mt, ct, at = m.body_mass[b], m.body_ipos[b].copy(), np.asarray(at, float)
        c = (mt*ct + dm*at)/(mt+dm)
        perp = lambda d: np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
        m.body_inertia[b] += mt*perp(ct-c) + dm*perp(at-c) + dm*r*r
        m.body_ipos[b] = c
        m.body_mass[b] = mt+dm
    set_mass(m.body('base_link').id, PAYLOAD_KG)
    for b in range(m.nbody):
        nm = m.body(b).name
        f = .58 if ('shin' in nm or 'metatarsus' in nm) else .6 if ('foot' in nm or 'toe' in nm) else None
        if f is not None:
            m.body_mass[b] *= f; m.body_inertia[b] *= f
    pelvis = MOTOR_KG['A'] + MOTOR_KG['B'] + ((MOTOR_KG['B'] + MOTOR_KG['A']) if arch12 else 0.)
    knee, tail = MOTOR_KG['B'], 2*MOTOR_KG['B']
    if motors is not None:
        pelvis, knee, tail = motors['hip_roll_link'], motors['thigh_link'], motors['tail_yaw_link']
    for side in ('left', 'right'):
        set_mass(m.body(f'{side}_hip_roll_link').id, pelvis)
        add_mass(m.body(f'{side}_thigh_link').id, knee, (0., 0., -.03))
        if arch12:
            add_mass(m.body(f'{side}_thigh_link').id, .1, (0., 0., -.11))
            add_mass(m.body(f'{side}_shin_link').id, .1, (0., 0., -.19))
        else:
            add_mass(m.body(f'{side}_shin_link').id, .08, (0., 0., -.19))
    set_mass(m.body('tail_yaw_link').id, tail)
    return float(m.body_mass.sum())


def motor_available(cls, w):
    if cls in MODULE_CURVES:  # vendor module: its 48 V T-N curve
        return np.interp(np.abs(w), *MODULE_CURVES[cls], right=0.)
    tau, w0, P = MOTOR_CLASSES[cls]
    a = np.abs(w)
    env = np.where(a <= .5*w0, tau, tau*np.clip((w0-a)/(.5*w0), 0., None))
    return np.minimum(env, P/np.maximum(a, 1e-3))
KIN = dict(fold=(65., 15., .30, .60), extend=(135., 12., .80, .98), static_tol=12., static_scale=10., retract=(1.7, .8, .75, .95))
WEIGHTS_RUN = dict(track_lin=3., progress=1., track_yaw=1., yaw_err=0., flight=0., air_time=0., gait=-3., stand=-2., grf=-1., cot=-.05, ang_mom=-.5,
                   fold=0., extend=0., ankle_static=0., retract=0., leg_sym=0., heat=0., spring_store=0., spring_release=0.,
                   lin_vel_z=-.5, ang_vel_xy=-.05, orientation=-5., height=-20., torque=-1e-5, action_rate=-.02,
                   joint_acc=-2e-8, slip=-.2, collision=-5., joint_limit=-5., alive=.5)


def nominal_pose(leg=NOMINAL_T1):
    table = json.loads(R02_YAML.read_text())['nominal_pose']  # R-02 crouch: joint names and the flat-toe convention
    leg = dict(zip(('hip_pitch', 'knee_pitch', 'ankle_pitch'), leg)) if leg else table
    pose = {f'{s}_{j}_joint': leg[j] for s in ('left', 'right') for j in table}
    return {**pose, 'left_hip_roll_joint': 0., 'right_hip_roll_joint': 0., 'left_ankle_roll_joint': 0.,
            'right_ankle_roll_joint': 0., 'tail_yaw_joint': 0., 'tail_pitch_joint': 0.}


class RunEnv(RaptorEnv):
    """T1 running environment (see module docstring). `tail`: 'active' (12 actions) or 'locked' (10 actions, tail
    held at its nominal pose by the same servo). `lock_tail` can also be toggled at run time to evaluate a
    tail-active policy with the tail held."""

    def __init__(self, mass=5., springs='c', actuator=None, tail='active', level=0., cmd_max=(3., .2, .5),
                 episode_s=10., randomize=True, seed=None, render_mode=None, model_path=None, weights=None,
                 zero_cmd=.1, top_cmd=.3, disturb_items=None, grf_cap=4., init_speed=True, obs_vel=True, ankle_clutch=False,
                 kin=None, yaw_impulse=None, couple_ankle=False, track_sigma_frac=.15, arch8=False,
                 springs_override=None, arch8_override=None, arch12=False, real_mass=False, hip_back=0., nominal=None,
                 target_ramp=False, obs_contact=False, model_rev=1, latch_fallback=None, friction_dr=None,
                 self_collision=None, knee_band=None, start_contact=None, armature=None, heat_limit=1., roll_impulse=None, init_noise=.02,
                 realism=None, scale_override=None, range_override=None, gait_duty_min=None,
                 contact_tc=None, terrain=None, obs_scan=None, rev3_config=None):
        self.mass, self.spring_set, self.tail = float(mass), springs, tail
        rev2 = int(model_rev) >= 2
        # model rev 3: R-03 MJCF (hip yaw) + rev3_b.json (modules, transmissions, four-bar, drive masses); implies arch8
        self.rev3 = use_modules(rev3_config or REV3_CONFIG) if int(model_rev) >= 3 else None
        if self.rev3 is not None:
            arch8 = True
        pick = lambda v, new, old: (new if rev2 else old) if v is None else v
        self.model_rev = int(model_rev)
        self.roll_impulse = roll_impulse  # (rate 1/s, lo, hi): training roll impulses about the trunk's own x axis, like yaw_impulse
        self.init_noise = float(init_noise)  # std (rad) of the joint-angle noise at a randomised reset
        self.heat_limit = float(heat_limit)  # the heat term penalises max(0, heat - heat_limit); 1 = continuous rating
        self.latch_fallback, self.friction_dr = bool(pick(latch_fallback, False, True)), pick(friction_dr, 'foot', 'floor')
        self.self_collision, self.start_contact = bool(pick(self_collision, True, False)), bool(pick(start_contact, True, False))
        self.knee_band, self.armature = pick(knee_band, KNEE_BAND, None), bool(pick(armature, True, False))
        if self.rev3 is not None and knee_band is None:
            self.knee_band = fourbar_band(self.rev3['fourbar'])
        # hip_back (m): R-03, the hip mount (pelvis, hip motors, legs) moved rearward along the torso; the torso payload
        # and the tail root stay where they are (raptor layout: pelvis at the rear of the trunk, tail root behind it).
        # nominal: (hip, knee, ankle) action-offset pose replacing NOMINAL_T1 (nominal_scan.py, same rule per variant).
        self.arch12, self.real_mass, self.hip_back = arch12, real_mass, float(hip_back)
        # target_ramp: each new PD target is interpolated linearly over the 20 ms control step instead of stepping. With the
        # step, the stiff servo (saturates at 0.1 rad error) spiked at every new action: arch8e hip pitch mean |tau| 37.5 N·m
        # in the first 3 ms of each step vs 14-17 N·m after, 88 % of its torque power above 20 Hz, so the heat measure was
        # dominated by the control discretisation, not the structure (2026-10-01). Real drives interpolate targets too.
        self.target_ramp = target_ramp
        if arch12:
            arch8 = True
        self.spec_t1 = {**ACTUATOR_T1, **(actuator or {})}
        self.level, self.cmd_max = float(level), np.array(cmd_max, float)
        self.weights = {**WEIGHTS_RUN, **(weights or {})}
        self.zero_cmd, self.top_cmd = zero_cmd, top_cmd
        self.disturb_items = set(disturb_items) if disturb_items is not None else set(DISTURB_ITEMS)
        self.grf_cap, self.init_speed, self.obs_vel = grf_cap, init_speed, obs_vel
        self.obs_contact = bool(obs_contact)
        # ankle_clutch: the ankle spring is engaged only while that foot is loaded (read each 20 ms control step), like the
        # ostrich intertarsal engage-disengage ligament and the BirdBot clutch. Without it the unilateral Achilles spring
        # resists swing flexion: holding a 65 deg intertarsal angle needs 45 N·m against a 22 N·m ankle motor (measured
        # 2026-09-30, v6a policy saturated 46 % at 87 deg). Energy stored at release is lost (a real clutch dissipates it).
        self.ankle_clutch = ankle_clutch
        # kin: overrides of the KIN windows/targets (v7 moved the fold right after toe-off, evidence 86 §10)
        self.kin = {**KIN, **{k: tuple(v) if isinstance(v, list) else v for k, v in (kin or {}).items()}}
        # yaw_impulse: (rate per s, lo, hi) N·m·s at 5 kg, applied during training independently of the curriculum level.
        # The level-scaled yaw impulses stayed below 0.12 N·m·s (level stuck near 0.3) while evaluation uses 1.0-1.5.
        self.yaw_impulse = yaw_impulse
        self.episode_steps, self.randomize, self.render_mode = int(episode_s/CONTROL_DT), randomize, render_mode
        self.model_path = str(model_path or (R03_MODEL if self.rev3 is not None else R02_MODEL))
        self.kinds, self.kind, self.dr_items, self.slew, self.jtc_horizon = ['flat'], 'flat', set(), None, 0.
        probe = mujoco.MjModel.from_xml_path(self.model_path)
        self.active = [probe.actuator(i).name for i in range(probe.nu)]
        pose = nominal_pose(tuple(nominal) if nominal is not None else NOMINAL_T1)
        if real_mass:
            self.mass = real_masses(probe, arch12, rev3_motor_masses(self.rev3) if self.rev3 is not None else None)
        self.q0 = np.array([pose.get(n, 0.) for n in self.active])
        # scale_override {joint type: rad}: action half-range around q0 replacing SCALE_RUN (e.g. a wider hip roll band)
        self.scale = np.array([{**SCALE_RUN, **(scale_override or {})}[joint_type(n)] for n in self.active])
        # range_override {joint type: rad}: symmetric joint stop +-rad replacing the MJCF range (a design change, e.g. hip roll 0.65)
        self.range_override = range_override or {}
        # gait_duty_min: floor of the scheduled stance duty replacing GAIT_DUTY (0.28 leaves ~44 % flight at running speed)
        self.gait_duty_min = GAIT_DUTY if gait_duty_min is None else float(gait_duty_min)
        # contact_tc: solref time constant of every contact replacing the MJCF/MuJoCo default 0.02 s, which lets the feet
        # sink 1-4 cm into flat ground (evidence 174); 0.005 is mujoco_menagerie Cassie's value (>= 2 physics steps)
        self.contact_tc = contact_tc
        # terrain: {"kinds": {kind: weight}, "max_level": L, "vmax": m/s, "size": m}. Each reset draws a kind (terrain.py)
        # and a level U(0, L) (U(lo, L) with "level_min": {kind: lo}; "gait_free": [ahead, height] drops the gait term near edges, see near_edge) and rebuilds the model when the ground changes. Off flat ground the command is straight
        # (no lateral/yaw) at U(0, vmax), the episode is truncated before the robot leaves the terrain, and heights
        # (fall check, sag) are taken above the local ground. None keeps flat ground and the earlier random stream.
        self.terrain = terrain
        self.kind, self.t_level = 'flat', 0.
        # obs_scan: {"dx": [m ahead along the heading], "noise": m, "bias": m, "delay": control steps}. Appends the ground
        # height at those points relative to the ground under the body (base height - spawn base height), clipped to +-1 m: an
        # elevation profile from a front depth camera (R-02 sensor pod). Gaussian noise per reading, a per-episode bias,
        # and `delay` steps on top of the realism sensor delay. A separate random stream keeps the other draws unchanged.
        # None keeps the earlier observation.
        self.obs_scan = obs_scan
        self.scan_dx = np.asarray(obs_scan['dx'], float) if obs_scan else np.zeros(0)
        self.scan_rng = np.random.default_rng(None if seed is None else seed+7919)
        self.scan_buf, self.scan_key, self.scan_bias = [], None, 0.
        if terrain:
            tr.SIZE = float(terrain.get('size', 40.))
        # couple_ankle: horse-style reciprocal apparatus. Each ankle pitch joint follows its knee 1:1 through a joint
        # equality (ankle = c - knee, c from the nominal pose); the ankle pitch motor is removed (no action, zero force),
        # so the knee motor drives both. Active DOF 12 -> 10. Knee limits become 28 N·m / 27 rad/s / 308 W
        # (spec full_budget: coupled requirement 21.7 N·m / 237 W at 5 kg, x1.3 margin as in §9).
        self.arch8 = arch8
        # arch8_override: {joint type: [class, gear]} replacing ARCH8 entries (e.g. hip_pitch B with a 1.5:1 belt)
        base = {jt: (ax['module'], ax['ratio']) for jt, ax in self.rev3['axes'].items()} if self.rev3 is not None else ARCH8
        self.arch8_map = {**base, **(ARCH12_EXTRA if arch12 else {}), **{k: tuple(v) for k, v in (arch8_override or {}).items()}}
        # link efficiency per joint type: LINK_ETA for every arch8 axis (rev <= 2); rev 3 from the config (1 = direct drive)
        self.link_eta = {jt: float(ax.get('link_efficiency', LINK_ETA)) for jt, ax in self.rev3['axes'].items()} if self.rev3 is not None else {}
        # springs_override: {joint type: [k (N·m/rad at the 11.36 kg design mass), q0, mode, engaged]}, engaged 'always',
        # 'stance' (clutch: only while that foot is loaded) or 'swing'. Replaces SPRINGS[springs] (spring_refit.py).
        self.springs_override = springs_override
        if arch8 and not arch12:
            couple_ankle = True
        self.couple_ankle = couple_ankle
        # speed-tracking width sigma = max(0.3, frac * command). With 0.15 the exp term vanished once the speed error
        # grew at high commands and v8c slowed from 6.4 to 3.1 m/s at an 8.1 m/s command without falling (2026-10-01).
        self.track_sigma_frac = track_sigma_frac
        self.tail_idx = np.array([i for i, n in enumerate(self.active) if n.startswith('tail')])
        self.leg_idx = np.array([i for i, n in enumerate(self.active) if not n.startswith('tail')])
        self.policy_idx = self.leg_idx if tail == 'locked' else np.arange(len(self.active))
        if couple_ankle:
            self.policy_idx = np.array([i for i in self.policy_idx if 'ankle_pitch' not in self.active[i]])
        if arch8 and not arch12:
            self.policy_idx = np.array([i for i in self.policy_idx if 'ankle_roll' not in self.active[i]])
        self.lock_tail = tail == 'locked'
        spec = np.array([self.spec_t1[joint_type(n)] for n in self.active])
        if couple_ankle:
            for i, n in enumerate(self.active):
                if 'ankle_pitch' in n:
                    spec[i] = (0., 27., 0.)
                elif 'knee' in n:
                    spec[i] = (28., 27., 308.)
        if arch8:
            for i, n in enumerate(self.active):
                jt = joint_type(n)
                if jt not in self.arch8_map:
                    spec[i] = (0., 99., 0.)
                else:
                    cls, g = self.arch8_map[jt]
                    gear = (self.fourbar_nominal() if self.rev3 is not None else 1.5) if g == 'fourbar' else g
                    spec[i] = (MOTOR_CLASSES[cls][0]*gear*self.link_eta.get(jt, LINK_ETA), MOTOR_CLASSES[cls][1]/gear, MOTOR_CLASSES[cls][2])
        self.tau_nom, self.v_max, self.p_max = spec[:, 0].copy(), spec[:, 1].copy(), spec[:, 2].copy()
        self.tau_max = self.tau_nom.copy()
        self.kp = self.tau_nom/KP_ERR
        self.kd = self.kp/KD_POLE
        n, na = len(self.active), len(self.policy_idx)
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11+3*(self.obs_vel)+2*n+na+2*self.obs_contact+len(self.scan_dx),),
                                                np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (na,), np.float32)
        # realism (opt-in, 2026-10-29): motor current/voltage/thermal limits, friction, backlash, delays, sensor noise,
        # battery sag (realism.py, docs/SIM_REALISM.ko.md). None keeps the earlier code path unchanged.
        cfg = load_config(realism)
        self.rz = Realism(cfg, seed, CONTROL_DT) if cfg else None
        if self.rz is not None and self.rz.obs_temp:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.observation_space.shape[0]+n,), np.float32)
        self.resample_steps = 250
        self.model = self.renderer = None

    # --- model -------------------------------------------------------------------------------
    def _build(self):
        spec = mujoco.MjSpec.from_file(self.model_path)
        self.heights = tr.add_terrain(spec, self.kind, self.t_level, self.rng)
        ratio = self.mass/DESIGN_MASS
        for side in ('left', 'right') if self.hip_back else ():
            b = spec.body(f'{side}_hip_roll_link')
            b.pos = [b.pos[0]-self.hip_back, b.pos[1], b.pos[2]]
        sp = ({j: tuple(v) for j, v in self.springs_override.items()} if self.springs_override
              else {j: (*v, 'stance' if (j == 'ankle_pitch' and self.ankle_clutch) else 'always') for j, v in SPRINGS[self.spring_set].items()})
        self.spring_engage = {}
        for joint, (k, q0, mode, engaged) in sp.items():
            for side in ('left', 'right'):
                band = [-1e6, q0] if mode == 'uni+' else [q0, 1e6] if mode == 'uni-' else [q0, q0]
                t = spec.add_tendon(name=f'spring_{side}_{joint}', stiffness=k*ratio, springlength=band)
                t.wrap_joint(f'{side}_{joint}_joint', 1.)
                self.spring_engage[(side, joint)] = (engaged, k*ratio)
        if self.render_mode == 'rgb_array':
            spec.worldbody.add_light(pos=[0, 0, 5], dir=[.3, .2, -1], diffuse=[.7, .7, .7], ambient=[.35, .35, .35],
                                     specular=[.1, .1, .1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True)
            spec.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                             rgb1=[.82, .84, .86], rgb2=[.62, .65, .68], width=512, height=512)
            spec.add_material(name='grid', textures=['', 'grid'], texrepeat=[16, 16], reflectance=.05)
            for g in spec.geoms:
                if g.parent.name == 'world':
                    g.material = 'grid'
        if self.couple_ankle:
            for side in ('left', 'right'):
                c = self.q0[self.active.index(f'{side}_ankle_pitch_joint')]+self.q0[self.active.index(f'{side}_knee_pitch_joint')]
                eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f'{side}_ankle_pitch_joint', name2=f'{side}_knee_pitch_joint',
                                       data=[c, -1.]+[0.]*9)
                eq.solref = [.004, 1.]          # stiff tendon linkage (default 0.02 s let the ankle lag 0.5 rad)
                eq.solimp = [.99, .999, .001, .5, 2.]
        if self.arch8 and not self.arch12:
            for side in ('left', 'right'):
                eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f'{side}_ankle_roll_joint', data=[0.]*11)
                eq.solref = [.004, 1.]
        if self.self_collision:
            # contype bit 0 keeps every link colliding with the ground; bits 1/2 = left/right leg, 3 = tail, 4 = torso
            for g in spec.geoms:
                nm = g.parent.name
                if nm == 'world':
                    continue
                side = 2 if nm.startswith('left') else 4 if nm.startswith('right') else 0
                if nm == 'base_link':
                    g.contype, g.conaffinity = 1 | 16, 0
                elif nm.startswith('tail'):
                    g.contype, g.conaffinity = 1 | 8, 0
                elif 'thigh' in nm:
                    g.contype, g.conaffinity = 1 | side, (6-side) | 8
                else:
                    g.contype, g.conaffinity = 1 | side, (6-side) | 8 | 16
        m = spec.compile()
        m.opt.timestep = PHYSICS_DT
        for n in self.active:
            if joint_type(n) in self.range_override:
                m.jnt_range[m.joint(n).id] = [-self.range_override[joint_type(n)], self.range_override[joint_type(n)]]
        # uniform mass scaling, geometry fixed (spec §5): masses and inertias. The R-02 toe springs are kept: scaled
        # with mass they fall below the W*h (5 kg x 0.71 m = 34 N·m/rad) a passive stand on the toes needs, and the
        # robot topples forward in ~3 s (measured 2026-09-29).
        if self.real_mass:
            real_masses(m, self.arch12, rev3_motor_masses(self.rev3) if self.rev3 is not None else None)
        else:
            m.body_mass[1:] *= ratio
            m.body_inertia[1:] *= ratio
        if self.arch8 and not self.real_mass:  # distal-light leg (spec: tube 0.6 -> 0.35 kg/m, foot 0.10 -> 0.06 kg) and knee motor on the upper femur
            base = m.body('base_link').id
            for b in range(m.nbody):
                nm = m.body(b).name
                f = .58 if ('shin' in nm or 'metatarsus' in nm) else .6 if ('foot' in nm or 'toe' in nm) else None
                if f is not None:
                    dm = m.body_mass[b]*(1-f)
                    m.body_mass[b] *= f; m.body_inertia[b] *= f; m.body_mass[base] += dm
            mk = .96*ratio  # class-B knee motor mass moved from the pelvis to the upper femur
            for side in ('left', 'right'):
                t = m.body(f'{side}_thigh_link').id
                mt, ct = m.body_mass[t], m.body_ipos[t].copy()
                pm = np.array([0., 0., -.03])
                m.body_ipos[t] = (mt*ct + mk*pm)/(mt+mk)
                m.body_inertia[t] += mk*np.array([.03**2, .03**2, 0.]) + .0005
                m.body_mass[t] = mt+mk; m.body_mass[base] -= mk
        if self.arch8:
            sys.path.insert(0, str(ROOT/'modeling'))
            from knee_linkage_study import fourbar_N
            qq = np.linspace(.1, 2.7, 300)
            fb = self.rev3['fourbar'] if self.rev3 is not None else ARCH8_FOURBAR
            _, N, _trans, ok = fourbar_N(qq, fb['a'], fb['b'], fb['c'], np.radians(fb['beta_deg']), fb['branch'])
            N = np.where(ok, np.abs(N), 1.)
            self.knee_ratio = (qq, np.clip(N, .5, 5.))
            # joint torque -> motor torque / continuous torque: gear x link efficiency x 0.4 peak. The four-bar knee uses its
            # ratio at the current knee angle each physics step (until 2026-10-01 a mean ratio 1.4 was used; Codex review).
            gear, cont = [], []
            for n in self.active:
                jt = joint_type(n)
                if jt in self.arch8_map:
                    cls, g = self.arch8_map[jt]
                    gear.append(1. if g == 'fourbar' else g); cont.append(MOTOR_CONT.get(cls, .4*MOTOR_CLASSES[cls][0]))
                else:
                    gear.append(1.); cont.append(1e9)
            self.gear_const, self.motor_cont = np.array(gear), np.array(cont)
            self.fourbar_idx = np.array([i for i, n in enumerate(self.active) if self.arch8_map.get(joint_type(n), (0, 0))[1] == 'fourbar'], int)
        self.act = np.array([m.actuator(n).id for n in self.active])
        self.ix = {n: i for i, n in enumerate(self.active)}
        self.q_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.active])
        self.v_adr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.active])
        self.lo, self.hi = m.jnt_range[[m.joint(n).id for n in self.active]].T.copy()
        if self.knee_band is not None:
            for i, n in enumerate(self.active):
                if joint_type(n) == 'knee_pitch':
                    self.lo[i], self.hi[i] = self.knee_band
        if self.armature and self.arch8:
            band = (self.knee_ratio[0] >= KNEE_BAND[0]) & (self.knee_ratio[0] <= KNEE_BAND[1])
            n2_knee = float(np.mean(self.knee_ratio[1][band]**2))
            for i, n in enumerate(self.active):
                jt = joint_type(n)
                if jt in self.arch8_map:
                    cls, g = self.arch8_map[jt]
                    m.dof_armature[m.jnt_dofadr[m.joint(n).id]] = MODULE_J.get(cls, ROTOR_J_OUT)*(n2_knee if g == 'fourbar' else g*g)
        # PD position servo as an affine actuator: force = kp (ctrl - q) - kd qd, clamped by forcerange each step
        m.actuator_gainprm[self.act, 0] = self.kp
        m.actuator_biasprm[self.act, :3] = np.c_[np.zeros(len(self.act)), -self.kp, -self.kd]
        m.actuator_ctrllimited[self.act] = 1
        m.actuator_ctrlrange[self.act] = np.c_[self.lo, self.hi]
        m.actuator_forcelimited[self.act] = 1
        m.actuator_forcerange[self.act] = np.c_[-self.tau_nom, self.tau_nom]
        m.jnt_actfrclimited[:] = 0
        if self.rz is not None:
            self.rz.setup(self, m)
        if self.contact_tc is not None:
            m.geom_solref[:, 0] = self.contact_tc
        self.model, self.data = m, mujoco.MjData(m)
        mujoco.mj_setConst(m, self.data)
        self.close()
        self.base = m.body('base_link').id
        self.root = m.body('base_root').id
        self.tail_body = m.body('tail_yaw_link').id
        self.leg_bodies = [m.body(f'{s}_hip_roll_link').id for s in ('left', 'right')]
        self.floor = m.geom('floor').id
        self.ground = {self.floor}|{g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        foot_bodies = [b for b in range(m.nbody) if any(k in m.body(b).name for k in ('foot', 'toe', 'metatarsus'))]
        self.foot_geoms = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] in foot_bodies
                               and m.body(m.geom_bodyid[g]).name.startswith(s)] for s in ('left', 'right')}
        self.foot_body = {s: m.body(f'{s}_foot_link').id for s in ('left', 'right')}
        self.pad_geoms = [g for s in self.foot_geoms for g in self.foot_geoms[s] if m.geom_priority[g] == 1]
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] != 0}
        self.nominal_mass = m.body_mass.copy()
        self.nominal_friction = m.geom_friction.copy()
        self.weight = 9.81*float(m.body_subtreemass[self.root])
        self.ankle_spring = {s_: m.tendon(f'spring_{s_}_ankle_pitch').id for s_ in ('left', 'right')
                             if f'spring_{s_}_ankle_pitch' in [m.tendon(i).name for i in range(m.ntendon)]}
        self.ankle_k = {s_: float(m.tendon_stiffness[t]) for s_, t in self.ankle_spring.items()}

    def _randomize(self):
        m = self.model
        m.body_mass[:] = self.nominal_mass
        m.geom_friction[:] = self.nominal_friction
        self.tau_max = self.tau_nom.copy()
        if not self.randomize:
            return
        m.body_mass[self.base] *= self.rng.uniform(.9, 1.15)
        if self.friction_dr == 'foot':
            idx = sorted(self.ground)+self.pad_geoms
            m.geom_friction[idx, 0] = self.nominal_friction[idx, 0]*self.rng.uniform(.625, 1.5625)
        else:
            m.geom_friction[sorted(self.ground), 0] = self.rng.uniform(.5, 1.25)
        self.tau_max = self.tau_nom*self.rng.uniform(.9, 1.1)

    def set_curriculum(self, level, cmd_max):
        self.level, self.cmd_max = float(level), np.array(cmd_max, float)

    # --- gym API -----------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self.rz is not None:
                self.rz.reseed(seed)
        if self.terrain:
            kinds = self.terrain['kinds']
            k = list(kinds)[self.rng.choice(len(kinds), p=np.array(list(kinds.values()), float)/sum(kinds.values()))]
            if k != 'flat' or self.kind != 'flat':
                self.model = None
            lo = self.terrain.get('level_min', {}).get(k, 0.)
            self.kind, self.t_level = k, (self.rng.uniform(lo, self.terrain.get('max_level', 1.)) if k != 'flat' else 0.)
        if self.model is None:
            self._build()
        self._randomize()
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        mujoco.mj_setConst(m, d)
        self.weight = 9.81*float(m.body_subtreemass[self.root])
        # init_seed (evaluation only, randomize off): a seeded small perturbation of the start so that repeated evaluation
        # episodes are distinct trials. Until 2026-10-01 every deterministic evaluation episode started from the identical
        # state, so "n episodes" were n copies of one trial.
        init_rng = np.random.default_rng(self.init_seed) if (not self.randomize and getattr(self, 'init_seed', None) is not None) else None
        d.qpos[self.q_adr] = self.q0+(self.rng.normal(0, self.init_noise, len(self.q0)) if self.randomize else
                                      init_rng.normal(0, .03, len(self.q0)) if init_rng is not None else 0)
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[g][2]-.02 for s in self.foot_geoms for g in self.foot_geoms[s])
        d.qpos[2] += -lowest+.005
        mujoco.mj_forward(m, d)
        if self.start_contact:
            d.qpos[2] -= self._pad_clearance()-1e-4
            mujoco.mj_forward(m, d)
        if not self.latch_fallback:
            for (s_, joint), (engaged, _) in self.spring_engage.items():
                if engaged == 'latch':
                    m.tendon_stiffness[m.tendon(f'spring_{s_}_{joint}').id] = 0.
        self.h_nom = d.xpos[self.base][2]-.03
        self.scan_buf, self.scan_key = [], None
        self.scan_bias = self.scan_rng.normal(0, self.obs_scan.get('bias', 0.)) if self.obs_scan else 0.
        d.ctrl[self.act] = d.qpos[self.q_adr]
        self.ctrl_prev = d.qpos[self.q_adr].copy()
        self.steps, self.last_action, self.prev_qd = 0, np.zeros(len(self.policy_idx)), d.qvel[self.v_adr].copy()
        self.air = {'left': 0., 'right': 0.}
        self.loaded = {'left': True, 'right': True}
        self.stance_peak = {'left': 0., 'right': 0.}
        self.touchdowns, self.peaks = [], []
        self.ankle_td = {'left': None, 'right': None}
        self.spring_was_on = {}
        self.stance_seen = {}
        self.latch_armed = {}
        self.ratchet = {}
        self.spring_inject_J = self.spring_release_J = self._release_seen = 0.
        self.heat = (self.rng.uniform(0., .8, len(self.q0)) if self.randomize else np.zeros(len(self.q0)))
        self.tau_lp = np.zeros(len(self.q0))
        self.sym_ema = None
        self.leg_scores = {side: dict(fold=0., extend=0., retract=0.) for side in ('left', 'right')}
        self.command = self._sample_command()
        self.phase = float(self.rng.random()) if self.randomize else float(init_rng.random()) if init_rng is not None else 0.
        self.push_at = self.rng.uniform(3., 8.)
        self.events = []
        d.xfrc_applied[:] = 0.
        if self.randomize and self.init_speed:
            d.qvel[0] += self.rng.uniform(0., .3*self.command[0])
        if init_rng is not None:
            d.qvel[0] += init_rng.uniform(0., .5)
        if self.rz is not None:
            self.rz.reset(self)
        return self._observe(), {}

    def _ratchet_arm(self, st, q):
        """Drop the pawl at joint angle q: rest angle = next tooth at or above q (zero deflection, no stored energy)."""
        rest = RATCHET_PITCH*np.ceil(q/RATCHET_PITCH)
        self.model.tendon_lengthspring[st['t']] = [-1e6, rest]
        self.spring_inject_J += .5*st['k']*max(0., q-rest)**2
        st['armed'], st['qmin'] = True, q

    def _pad_clearance(self):
        """Height of the lowest foot-pad surface point above z = 0 (sphere radius or box extent along world z)."""
        m, d = self.model, self.data
        ext = lambda g: (m.geom_size[g][0] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                         else abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g])
        return float(min(d.geom_xpos[g][2]-ext(g) for g in self.pad_geoms))

    def self_contacts(self):
        """Number of active contacts between two robot geoms (0 unless self_collision)."""
        d = self.data
        return sum(1 for i in range(d.ncon) if d.contact[i].geom1 in self.robot_geoms and d.contact[i].geom2 in self.robot_geoms)

    def _sample_command(self):
        c = self.rng.uniform(-1, 1, 3)*self.cmd_max
        u = self.rng.random()
        c[0] = 0. if u < self.zero_cmd else self.cmd_max[0] if u < self.zero_cmd+self.top_cmd \
            else self.rng.uniform(0., self.cmd_max[0])
        if c[0] == 0.:
            c[1:] = 0.
        if self.kind != 'flat':
            c[:] = 0.
            if u >= self.zero_cmd:
                c[0] = self.rng.uniform(0., self.terrain.get('vmax', 3.))
        return c

    def _observe(self):
        if self.rz is not None:
            return self.rz.observe(self)  # delayed and noisy sensors (realism)
        return self._obs()

    def _obs(self):
        return self._compose(self._sense())

    def _sense(self):
        """Raw sensor values of the observation (realism delays and corrupts these): IMU gyro, gravity direction, body
        velocity (state estimator), joint positions and velocities."""
        d = self.data
        R = d.xmat[self.base].reshape(3, 3)
        s = {'gyro': d.qvel[3:6].copy(), 'grav': R.T @ np.array([0, 0, -1.])}
        if self.obs_vel:
            s['vel'] = self.body_velocity(self.base)[1]
        s['q'], s['qd'] = d.qpos[self.q_adr].copy(), d.qvel[self.v_adr].copy()
        if self.obs_scan:
            s['scan'] = self._scan()
        return s

    def _scan(self):
        """Delayed, noisy ground-height profile ahead of the body (obs_scan). One reading per control step: the realism
        layer samples the sensors every physics step, and later calls in the same control step return the same reading
        (with realism delays on, the reading is therefore up to one control step older than the other sensors)."""
        if self.scan_key == self.steps:
            return self.scan_val
        d, cfg = self.data, self.obs_scan
        x, y, z = d.xpos[self.base]
        R = d.xmat[self.base].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        g = np.array([tr.height_at(self.heights, x+dx*np.cos(yaw), y+dx*np.sin(yaw)) for dx in self.scan_dx])
        v = np.clip(g-(z-self.h_nom-.03)+self.scan_bias+self.scan_rng.normal(0, cfg.get('noise', 0.), len(g)), -1., 1.)
        self.scan_buf.append(v)
        self.scan_key = self.steps
        self.scan_val = self.scan_buf.pop(0) if len(self.scan_buf) > int(cfg.get('delay', 0)) else self.scan_buf[0]
        return self.scan_val

    def near_edge(self):
        """terrain "gait_free": [ahead m, height m]: True when the true ground from 0.3 m behind the base to `ahead` m in
        front of it spans more than `height` (a ledge or drop near); the gait schedule term is then not applied."""
        free = self.terrain.get('gait_free') if self.terrain else None
        if not free or self.kind == 'flat':
            return False
        d = self.data
        x, y = d.xpos[self.base][:2]
        R = d.xmat[self.base].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        g = [tr.height_at(self.heights, x+dx*np.cos(yaw), y+dx*np.sin(yaw)) for dx in np.linspace(-.3, free[0], 9)]
        return np.ptp(g) > free[1]

    def ground_z(self):
        """Ground height for the fall check and sag: under the base, except on boxes, where the hip passes over a ledge
        before the feet climb it: there the ground under the mean foot x."""
        d = self.data
        if self.kind == 'flat':
            return 0.
        x, y = d.xpos[self.base][:2]
        if self.kind == 'boxes':
            x = float(np.mean([d.geom_xpos[g][0] for s in self.foot_geoms for g in self.foot_geoms[s]]))
        return tr.height_at(self.heights, x, y)

    def _compose(self, s):
        parts = [s['gyro']*.25, s['grav'], self.command*np.array([.1, 2., .5])]
        if self.obs_vel:
            parts.append(s['vel']*.1)
        parts += [s['q']-self.q0, s['qd']*.05, self.last_action,
                  [np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase)]]
        if self.obs_contact:
            parts.append([float(self.loaded['left']),float(self.loaded['right'])])
        if self.obs_scan:
            parts.append(s['scan'])
        return np.concatenate(parts).astype(np.float32)

    # --- virtual actuator ----------------------------------------------------------------------
    def _clamp(self):
        """Force range per actuator for this physics step: torque limit, power limit P/|qd|, no drive beyond v_max.
        With the realism motor model the motor joints get its current/voltage/thermal limits instead."""
        lo, hi = self._clamp_arch8() if self.arch8 else self._clamp_t1()
        if getattr(self, 'rz', None) is not None:
            lo, hi = self.rz.motor_limits(self, lo, hi)
        return lo, hi

    def _clamp_t1(self):
        qd = self.data.qvel[self.v_adr]
        lim = np.minimum(self.tau_max, self.p_max/np.maximum(np.abs(qd), 1e-2))
        lo, hi = -lim, lim.copy()
        hi = np.where(qd > self.v_max, 0., hi)
        lo = np.where(qd < -self.v_max, 0., lo)
        self.model.actuator_forcerange[self.act, 0] = lo
        self.model.actuator_forcerange[self.act, 1] = hi
        return lo, hi

    def fourbar_nominal(self):
        """Mean |N| of the rev 3 four-bar over its knee band (nominal knee gear for tau_nom and the PD gains)."""
        sys.path.insert(0, str(ROOT/'modeling'))
        from knee_linkage_study import fourbar_N
        fb = self.rev3['fourbar']
        q = np.linspace(*self.knee_band, 200)
        _, N, _, _ = fourbar_N(q, fb['a'], fb['b'], fb['c'], np.radians(fb['beta_deg']), fb['branch'])
        return float(np.mean(np.abs(N)))

    def _clamp_arch8(self):
        """Motor-envelope limits through each joint's transmission (ARCH8): drive torque from the motor curve at the
        motor speed, braking up to the class peak; the random +-10 % actuator factor scales both."""
        q, qd = self.data.qpos[self.q_adr], self.data.qvel[self.v_adr]
        scale = self.tau_max/np.maximum(self.tau_nom, 1e-9)
        lo, hi = np.zeros(len(qd)), np.zeros(len(qd))
        for i, n in enumerate(self.active):
            jt = joint_type(n)
            if jt not in self.arch8_map:
                continue
            cls, g = self.arch8_map[jt]
            gear = float(np.interp(q[i], *self.knee_ratio)) if g == 'fourbar' else g
            eta = self.link_eta.get(jt, LINK_ETA)
            drive = gear*eta*float(motor_available(cls, gear*qd[i]))*scale[i]
            brake = gear*eta*MOTOR_CLASSES[cls][0]*scale[i]
            if qd[i] >= 0:
                hi[i], lo[i] = drive, -brake
            else:
                hi[i], lo[i] = brake, -drive
        self.model.actuator_forcerange[self.act, 0] = lo
        self.model.actuator_forcerange[self.act, 1] = hi
        return lo, hi

    @staticmethod
    def _window(x, a, b, edge=.03):
        """1 inside [a, b], raised-cosine ramps of width `edge` outside, 0 beyond."""
        if x < a-edge or x > b+edge:
            return 0.
        if x < a:
            return .5-.5*np.cos(np.pi*(x-a+edge)/edge)
        if x > b:
            return .5+.5*np.cos(np.pi*(x-b)/edge)
        return 1.

    def gait_duty(self, cx=None):
        cx = self.command[0] if cx is None else cx
        return float(np.clip(.6-.128*(cx-.5), self.gait_duty_min, .6))

    def kinematics_terms(self, loaded, w_pitch):
        """Joint-flexion shaping terms (see KIN), averaged over the two legs; zeros below 0.5 m/s."""
        out = dict(fold=0., extend=0., ankle_static=0., retract=0.)
        self.leg_scores = {side: dict(fold=0., extend=0., retract=0.) for side in ('left', 'right')}
        if self.command[0] < .5:
            return out
        d = self.gait_duty()
        q, qd = self.data.qpos[self.q_adr], self.data.qvel[self.v_adr]
        for side, off in (('left', 0.), ('right', .5)):
            ph = (self.phase-off) % 1.
            a = 180.+np.degrees(q[self.ix[f'{side}_ankle_pitch_joint']])
            if ph < d:
                w = self._window(ph/d, .15, .85)
                if w > 0:
                    if loaded[side] and self.ankle_td.get(side) is not None:
                        e = max(0., abs(a-self.ankle_td[side])-self.kin['static_tol'])/self.kin['static_scale']
                        out['ankle_static'] += w*min(1., e*e)
                    else:
                        out['ankle_static'] += w
            else:
                s_ = (ph-d)/(1.-d)
                sc = self.leg_scores[side]
                if not loaded[side]:
                    t, sig, lo, hi = self.kin['fold']; sc['fold'] = self._window(s_, lo, hi)*np.exp(-(a-t)**2/(2*sig*sig))
                    t, sig, lo, hi = self.kin['extend']; sc['extend'] = self._window(s_, lo, hi)*np.exp(-(a-t)**2/(2*sig*sig))
                t, sig, lo, hi = self.kin['retract']
                om = qd[self.ix[f'{side}_hip_pitch_joint']]+w_pitch  # world thigh rate, + = backward
                # only while the foot is actually in the air (Codex review: the evaluator measures real swing)
                sc['retract'] = self._window(s_, lo, hi)*np.exp(-(om-t)**2/(2*sig*sig)) if not loaded[side] else 0.
                for k_, v_ in sc.items():
                    out[k_] += v_
        return {k: v/2 for k, v in out.items()}

    def leg_symmetry(self):
        """Update the low-pass filtered per-leg angles/scores and return the squared left-right difference."""
        q = self.data.qpos[self.q_adr]
        al = CONTROL_DT/SYM_TAU
        val = {side: np.array([q[self.ix[f'{side}_{j}_joint']] for j in ('hip_pitch', 'knee_pitch', 'ankle_pitch')]
                              + [2*self.leg_scores[side][k] for k in ('fold', 'extend', 'retract')]) for side in ('left', 'right')}
        if self.sym_ema is None:
            self.sym_ema = {k: v.copy() for k, v in val.items()}
        for side in val:
            self.sym_ema[side] += al*(val[side]-self.sym_ema[side])
        return float(np.sum((self.sym_ema['left']-self.sym_ema['right'])**2))

    def gait_schedule(self, cx=None):
        """Expected stance (0..1, smooth edges) for left and right from the stride clock; duty shrinks from 0.6 at
        0.5 m/s to GAIT_DUTY at 3 m/s and above. None below 0.5 m/s (standing is handled by the `stand` term)."""
        cx = self.command[0] if cx is None else cx
        if cx < .5:
            return None
        d = self.gait_duty(cx)
        out = {}
        for side, off in (('left', 0.), ('right', .5)):
            centre = off+d/2
            dist = abs(((self.phase-centre+.5) % 1.)-.5)
            out[side] = float(np.clip((d/2+GAIT_EDGE-dist)/(2*GAIT_EDGE), 0., 1.))
        return out

    # --- disturbances --------------------------------------------------------------------------
    def apply_impulse(self, axis, impulse, duration=.04, body=None):
        """Torque impulse (N·m·s) about a world axis ('x', 'y', 'z') on the torso, spread over `duration` s.
        'y' in the torso frame is used for pitch so the impulse is about the body's own pitch axis."""
        d = self.data
        w = np.zeros(6)
        if axis in ('y', 'r'):  # 'r' = the trunk's own roll (x) axis
            w[3:] = d.xmat[self.base].reshape(3, 3)[:, 1 if axis == 'y' else 0]*impulse/duration
        else:
            w[3+'xyz'.index(axis)] = impulse/duration
        self.events.append((self.base if body is None else body, w, d.time+duration-1e-6))

    def apply_force(self, body, force, duration):
        w = np.zeros(6)
        w[:3] = np.asarray(force, float)/1.
        self.events.append((body, w, self.data.time+duration-1e-6))

    def _disturb(self, feet_loaded):
        d, L, s = self.data, self.level, self.mass/5.
        if self.yaw_impulse and self.rng.random() < self.yaw_impulse[0]*CONTROL_DT:
            self.apply_impulse('z', s*self.rng.uniform(*self.yaw_impulse[1:])*self.rng.choice([-1, 1]))
        if self.roll_impulse and self.rng.random() < self.roll_impulse[0]*CONTROL_DT:
            self.apply_impulse('r', s*self.rng.uniform(*self.roll_impulse[1:])*self.rng.choice([-1, 1]))
        if L <= 0:
            return
        if 'pitch' in self.disturb_items and self.rng.random() < .3*CONTROL_DT:
            self.apply_impulse('y', L*s*self.rng.uniform(.15, .4)*self.rng.choice([-1, 1]))
        if 'yaw' in self.disturb_items and self.rng.random() < .3*CONTROL_DT:
            self.apply_impulse('z', L*s*self.rng.uniform(.15, .4)*self.rng.choice([-1, 1]))
        if 'trip' in self.disturb_items and self.rng.random() < .5*CONTROL_DT:
            swing = [side for side in feet_loaded if not feet_loaded[side]]
            if swing:
                side = swing[int(self.rng.integers(len(swing)))]
                self.apply_force(self.foot_body[side], [-L*self.rng.uniform(.5, 1.5)*self.weight, 0., 0.], .06)

    def _apply_events(self):
        d = self.data
        d.xfrc_applied[:] = 0.
        self.events = [e for e in self.events if d.time < e[2]]
        for body, w, _ in self.events:
            d.xfrc_applied[body] += w

    # --- angular momentum ----------------------------------------------------------------------
    def angular_momentum(self):
        """Whole-body, tail and legs angular momentum about the whole-body CoM (world frame, N·m·s)."""
        m, d = self.model, self.data
        mujoco.mj_subtreeVel(m, d)
        c = d.subtree_com[self.root]

        def part(b):
            return d.subtree_angmom[b]+np.cross(d.subtree_com[b]-c, m.body_subtreemass[b]*d.subtree_linvel[b])
        return d.subtree_angmom[self.root].copy(), part(self.tail_body), sum(part(b) for b in self.leg_bodies)

    # --- step ----------------------------------------------------------------------------------
    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, float), -1, 1)
        target = self.q0.copy()
        target[self.policy_idx] = np.clip(self.q0[self.policy_idx]+self.scale[self.policy_idx]*action,
                                          self.lo[self.policy_idx], self.hi[self.policy_idx])
        if self.lock_tail:
            target[self.tail_idx] = self.q0[self.tail_idx]
        if self.randomize:
            self._disturb(self.loaded)
            if 'push' in self.disturb_items and abs(d.time-self.push_at) < CONTROL_DT/2:
                d.qvel[:2] += self.rng.uniform(-.4, .4, 2)*(1.+self.level)
        self._apply_events()
        start = self.ctrl_prev if self.target_ramp else target
        self.ctrl_prev = target.copy()
        d.ctrl[self.act] = target
        for (s_, joint), (engaged, k_) in self.spring_engage.items():
            if engaged == 'latch':
                # ostrich intertarsal engage-disengage mechanism (Schaller 2009): the spring latches when the joint extends
                # to its rest angle before touchdown (zero deflection, no jolt) and releases at lift-off
                t = m.tendon(f'spring_{s_}_{joint}').id
                qj = d.qpos[m.jnt_qposadr[m.joint(f'{s_}_{joint}_joint').id]]
                q0 = m.tendon_lengthspring[t][0] if m.tendon_lengthspring[t][0] > -1e5 else m.tendon_lengthspring[t][1]
                was = self.spring_was_on.get((s_, joint), False)
                if self.loaded[s_]:
                    on = True if self.latch_fallback else was   # stays engaged through stance (fallback: engage at touchdown)
                else:
                    on = was and self.stance_seen.get(s_, False) is False and False
                    if not self.latch_fallback and qj > q0:
                        self.latch_armed[s_] = True            # the knee has flexed above the rest angle in this swing
                    if not was and qj <= q0 and (self.latch_fallback or self.latch_armed.get(s_, False)):
                        on = True                              # extension past the rest angle in swing: latch
                    elif was and not self.stance_seen.get(s_, False):
                        on = True                              # latched in late swing, waiting for touchdown
                if self.loaded[s_]:
                    self.stance_seen[s_] = True
                elif self.stance_seen.get(s_, False):         # lift-off after a stance: release
                    on, self.stance_seen[s_] = False, False
                    self.latch_armed[s_] = False
                self.spring_was_on[(s_, joint)] = on
                m.tendon_stiffness[t] = k_ if on else 0.
                continue
            if engaged == 'ratchet':
                t = m.tendon(f'spring_{s_}_{joint}').id
                adr = m.jnt_qposadr[m.joint(f'{s_}_{joint}_joint').id]
                qj = d.qpos[adr]
                st = self.ratchet.get((s_, joint))
                if st is None:  # start of the episode: engaged at the current angle
                    st = self.ratchet[(s_, joint)] = dict(t=t, adr=adr, k=k_, armed=False, qmin=qj, qpeak=qj, stance=False)
                    self._ratchet_arm(st, qj)
                if self.loaded[s_]:
                    st['stance'] = True
                    if not st['armed']:  # the swing never came back RATCHET_REARM: the load drops the pawl
                        self._ratchet_arm(st, qj)
                else:
                    if st['stance']:  # lift-off: pawl lifted, whatever is still stored is lost
                        st['stance'] = False
                        if st['armed']:
                            self.spring_release_J += .5*k_*max(0., qj-m.tendon_lengthspring[t][1])**2
                        st['armed'], st['qpeak'] = False, qj
                    if not st['armed']:
                        st['qpeak'] = max(st['qpeak'], qj)
                        if qj < st['qpeak']-RATCHET_REARM:
                            self._ratchet_arm(st, qj)
                self.spring_was_on[(s_, joint)] = st['armed']
                m.tendon_stiffness[t] = k_ if st['armed'] else 0.
                continue
            if engaged != 'always':
                t = m.tendon(f'spring_{s_}_{joint}').id
                on = self.loaded[s_] if engaged in ('stance', 'touchdown') else not self.loaded[s_]
                if engaged == 'touchdown' and on and not self.spring_was_on.get((s_, joint), False):
                    qj = d.qpos[m.jnt_qposadr[m.joint(f'{s_}_{joint}_joint').id]]
                    lo_, hi_ = m.tendon_lengthspring[t]
                    m.tendon_lengthspring[t] = [qj, qj] if lo_ == hi_ or (lo_ > -1e5 and hi_ < 1e5) else ([-1e6, qj] if lo_ < -1e5 else [qj, 1e6])
                self.spring_was_on[(s_, joint)] = on
                m.tendon_stiffness[t] = k_ if on else 0.
        n_sub = int(round(CONTROL_DT/PHYSICS_DT))
        power, torque_sq = 0., 0.
        heat_sq, heat_lp_sq = np.zeros(len(self.act)), np.zeros(len(self.act))
        for k_sub in range(n_sub):
            if self.target_ramp:
                d.ctrl[self.act] = start+(k_sub+1)/n_sub*(target-start)
            if self.rz is not None:
                d.ctrl[self.act] = self.rz.command(self, start+(k_sub+1)/n_sub*(target-start) if self.target_ramp else target)
            lo_, hi_ = self._clamp()
            if self.rz is not None:
                self.rz.before_substep(self, lo_, hi_, PHYSICS_DT)
            mujoco.mj_step(m, d)
            if self.rz is not None:
                self.rz.after_substep(self, PHYSICS_DT)
            for st in self.ratchet.values():  # the rest angle follows the most extended angle, one tooth at a time
                if st['armed'] and d.qpos[st['adr']] < st['qmin']:
                    st['qmin'] = d.qpos[st['adr']]
                    m.tendon_lengthspring[st['t']] = [-1e6, RATCHET_PITCH*np.ceil(st['qmin']/RATCHET_PITCH)]
            tau, qd = d.actuator_force[self.act], d.qvel[self.v_adr]
            power += np.sum(np.clip(tau*qd, 0, None))
            torque_sq += np.sum(tau**2)
            if self.arch8:
                g = self.gear_const.copy()
                g[self.fourbar_idx] = np.interp(d.qpos[self.q_adr[self.fourbar_idx]], *self.knee_ratio)
                tm = tau/(g*LINK_ETA*self.motor_cont)
                heat_sq += tm**2
                # reporting only: the same ratio after a 20 Hz first-order low-pass. ~35 % of the 1 ms knee torque power
                # of arch8e lies above 50 Hz (servo ringing, impacts; stride ~3 Hz) that a real current loop and
                # transmission compliance would partly filter, so heat_lp is the lower bound and heat the upper.
                self.tau_lp += LP20*(tm - self.tau_lp)
                heat_lp_sq += self.tau_lp**2
        power, torque_sq = power/n_sub, torque_sq/n_sub
        if self.arch8:
            self.heat_inst = heat_sq/n_sub  # (motor torque / continuous torque)^2 this control step (heat_test.py)
            self.heat_lp_inst = heat_lp_sq/n_sub
            self.heat += CONTROL_DT/HEAT_TAU*(self.heat_inst - self.heat)
        self.steps += 1
        cx = self.command[0]
        self.phase = (self.phase+CONTROL_DT*(1.6+.14*abs(cx))) % 1.
        if self.resample_steps and self.steps % self.resample_steps == 0:
            self.command = self._sample_command()

        R = d.xmat[self.base].reshape(3, 3)
        w, v_body = self.body_velocity(self.base)
        grav = R.T @ np.array([0, 0, -1.])
        qd = d.qvel[self.v_adr]
        feet, body_hit = self._contacts()
        W = self.weight
        L, L_tail, L_legs = self.angular_momentum()
        terms = {}
        sigma = max(.3, self.track_sigma_frac*abs(cx))
        terms['track_lin'] = np.exp(-np.sum((self.command[:2]-v_body[:2])**2)/sigma**2)
        if 'lin_err' in self.weights:
            # Optional dense penalty, including the large-error braking interval.
            # Absent by default: preserve the original reward terms and sums.
            terms['lin_err'] = float(np.linalg.norm(self.command[:2]-v_body[:2]))
        if 'lateral_err' in self.weights:
            terms['lateral_err'] = float(abs(self.command[1]-v_body[1]))
        terms['progress'] = np.clip(v_body[0], -1., cx)/max(cx, 1.)
        terms['track_yaw'] = np.exp(-(self.command[2]-w[2])**2/.2)
        terms['yaw_err'] = abs(self.command[2]-w[2])  # linear term: gradient for large turn commands (off by default)
        terms['lin_vel_z'] = v_body[2]**2
        terms['ang_vel_xy'] = np.sum(w[:2]**2)
        terms['orientation'] = np.sum(grav[:2]**2)
        bz = d.xpos[self.base][2]-self.ground_z()
        sag = self.h_nom-bz-.1  # 0.1 m free band: SLIP stance compression (spec §2: 0.15 m at the CoM)
        terms['height'] = sag**2 if sag > 0 else 0.
        terms['torque'] = torque_sq
        terms['action_rate'] = np.sum((action-self.last_action)**2)
        terms['joint_acc'] = np.sum(((qd-self.prev_qd)/CONTROL_DT)**2)
        terms['cot'] = power/(W*max(abs(cx), 1.))
        terms['ang_mom'] = L[1]**2+L[2]**2
        air_reward, slip, grf = 0., 0., 0.
        moving = cx > .05
        loaded_now = {}
        for s in feet:
            f = feet[s]/W
            loaded = f > .15
            loaded_now[s] = loaded
            grf += max(0., f-self.grf_cap)**2
            if loaded:
                self.stance_peak[s] = max(self.stance_peak[s], f)
                if not self.loaded[s]:  # touchdown
                    self.touchdowns.append((d.time, s))
                    self.ankle_td[s] = 180.+np.degrees(d.qpos[self.q_adr[self.ix[f'{s}_ankle_pitch_joint']]])
                    if self.randomize and self.level > 0 and 'touchdown' in self.disturb_items and self.rng.random() < .3:
                        self.apply_impulse('y', self.level*(self.mass/5.)*self.rng.uniform(0., .3)*self.rng.choice([-1, 1]), .02)
                    if self.air[s] > 0 and moving:
                        air_reward += min(self.air[s], .35)-.12  # reward swings of 0.12..0.35 s
                self.air[s] = 0.
                slip += np.sum(self.body_velocity(self.foot_body[s], local=False)[1][:2]**2)
            else:
                if self.loaded[s] and self.stance_peak[s] > 0:  # lift-off
                    self.peaks.append(self.stance_peak[s])
                    self.stance_peak[s] = 0.
                self.air[s] += CONTROL_DT
        self.loaded = loaded_now
        flight = not any(loaded_now.values())
        terms['flight'] = float(flight and cx >= RUN_SPEED and v_body[0] > .5*cx)
        sched = self.gait_schedule()
        terms['gait'] = sum((min(feet[s_]/(.2*W), 1.)-sched[s_])**2 for s_ in feet) if sched and not self.near_edge() else 0.
        terms.update(self.kinematics_terms(loaded_now, w[1]))
        terms['leg_sym'] = self.leg_symmetry() if self.command[0] >= .5 else 0.
        terms['heat'] = float(np.sum(np.clip(self.heat-self.heat_limit, 0., None))) if self.arch8 else 0.
        terms['stand'] = float(flight) if cx < .05 else 0.  # hopping in place at a zero command (v1 defect)
        if 'stand_support' in self.weights:
            # Optional double-support objective, only for a fully zero command.
            # In-place turning and lateral motion retain their original rewards.
            terms['stand_support'] = (sum(not v for v in loaded_now.values())/len(loaded_now)
                                      if np.all(self.command == 0.) else 0.)
        terms['air_time'] = air_reward/CONTROL_DT if moving else 0.
        terms['grf'] = grf
        terms['slip'] = slip
        terms['collision'] = float(body_hit)
        if 'self_collision' in self.weights:
            terms['self_collision'] = float(self.self_contacts() > 0)
        q = d.qpos[self.q_adr]
        margin = .9*(self.hi-self.lo)/2
        mid = (self.hi+self.lo)/2
        terms['joint_limit'] = np.sum(np.clip(np.abs(q-mid)-margin, 0, None))
        terms['alive'] = 1.
        if self.weights.get('spring_store'):
            gain = 0.
            for (s_, _j), st in self.ratchet.items():
                u = .5*st['k']*max(0., d.qpos[st['adr']]-m.tendon_lengthspring[st['t']][1])**2 if st['armed'] else 0.
                if loaded_now[s_]:
                    add = min(max(0., u-st.get('u', 0.)), SPRING_STORE_CAP-st.get('credit', 0.))
                    st['credit'] = st.get('credit', 0.)+add
                    gain += add
                else:
                    st['credit'] = 0.
                st['u'] = u
            terms['spring_store'] = gain/CONTROL_DT
        if self.weights.get('spring_release'):  # rate (W) at which the ratchet throws stored energy away at lift-off
            terms['spring_release'] = (self.spring_release_J-getattr(self, '_release_seen', 0.))/CONTROL_DT
            self._release_seen = self.spring_release_J
        reward = CONTROL_DT*sum(self.weights[k]*v for k, v in terms.items())

        tilt = np.arccos(np.clip(-grav[2], -1, 1))
        fallen = (body_hit or tilt > .8 or bz < .5*self.h_nom or not np.isfinite(d.qpos).all()
                  or np.abs(v_body).max() > 30.)
        if fallen:
            reward -= 10.
        self.last_action, self.prev_qd = action, qd.copy()
        truncated = self.steps >= self.episode_steps
        if self.kind != 'flat':
            bx, by = d.xpos[self.base][:2]
            truncated |= bx > tr.SIZE/2-1. or (self.kind == 'rough' and abs(by) > tr.ROUGH_HALF_WIDTH-.2)
        info = {'terms': terms, 'tilt': tilt, 'v_body': v_body, 'command': self.command.copy(), 'kind': self.kind,
                'x': d.xpos[self.base][0], 'power': power, 'grf': {s: feet[s]/W for s in feet}, 'flight': flight,
                'loaded': loaded_now, 'L': L, 'L_tail': L_tail, 'L_legs': L_legs,
                'tail_qd': d.qvel[self.v_adr][self.tail_idx].copy(), 'phase': self.phase,
                'heat': self.heat.copy() if self.arch8 else None,
                'latched': {s_: bool(on_) for (s_, _j), on_ in self.spring_was_on.items()}}
        if self.rz is not None:
            info['realism'] = self.rz.info()
        return self._observe(), float(reward), bool(fallen), bool(truncated), info
