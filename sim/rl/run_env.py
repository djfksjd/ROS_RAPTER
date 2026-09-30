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

ROOT = SIM.parent
R02_MODEL = SIM/'raptor_r02.xml'
R02_YAML = ROOT/'src/raptor_description/config/r02_design.yaml'
DESIGN_MASS = 11.36
PHYSICS_DT = .001
# T1 actuator limits (N·m, rad/s, W): spec §9 (5 kg + (c), 1.3x margin); tail: spec §8 proposal.
# Ankle pitch torque 22 N·m (the A-class value, spec §6) instead of the §9 8 N·m: standing still with the case-(c)
# ankle spring needs 9-11 N·m of motor torque against the spring preload (its rest angle is the running touchdown
# angle; deeper rest angles lose the running benefit: q0 -1.3 rad -> 11.5 N·m/159 W at 5 kg), a load case the spec
# did not check (measured 2026-09-29, pose scan). Power and speed limits stay at §9; running needs 5.8 N·m there.
ACTUATOR_T1 = {'hip_pitch': (35., 27., 350.), 'knee_pitch': (28., 27., 225.), 'ankle_pitch': (22., 27., 90.),
               'hip_roll': (35., 27., 350.), 'ankle_roll': (8., 27., 90.),
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
SCALE_RUN = {'hip_roll': .3, 'hip_pitch': 1., 'knee_pitch': 1., 'ankle_pitch': 1., 'ankle_roll': .3,
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
KIN = dict(fold=(65., 15., .30, .60), extend=(135., 12., .80, .98), static_tol=12., static_scale=10., retract=(1.7, .8, .75, .95))
WEIGHTS_RUN = dict(track_lin=3., progress=1., track_yaw=1., yaw_err=0., flight=0., air_time=0., gait=-3., stand=-2., grf=-1., cot=-.05, ang_mom=-.5,
                   fold=0., extend=0., ankle_static=0., retract=0., leg_sym=0.,
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
                 kin=None, yaw_impulse=None, couple_ankle=False):
        self.mass, self.spring_set, self.tail = float(mass), springs, tail
        self.spec_t1 = {**ACTUATOR_T1, **(actuator or {})}
        self.level, self.cmd_max = float(level), np.array(cmd_max, float)
        self.weights = {**WEIGHTS_RUN, **(weights or {})}
        self.zero_cmd, self.top_cmd = zero_cmd, top_cmd
        self.disturb_items = set(disturb_items) if disturb_items is not None else set(DISTURB_ITEMS)
        self.grf_cap, self.init_speed, self.obs_vel = grf_cap, init_speed, obs_vel
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
        self.model_path = str(model_path or R02_MODEL)
        self.kinds, self.kind, self.dr_items, self.slew, self.jtc_horizon = ['flat'], 'flat', set(), None, 0.
        probe = mujoco.MjModel.from_xml_path(self.model_path)
        self.active = [probe.actuator(i).name for i in range(probe.nu)]
        pose = nominal_pose()
        self.q0 = np.array([pose.get(n, 0.) for n in self.active])
        self.scale = np.array([SCALE_RUN[joint_type(n)] for n in self.active])
        # couple_ankle: horse-style reciprocal apparatus. Each ankle pitch joint follows its knee 1:1 through a joint
        # equality (ankle = c - knee, c from the nominal pose); the ankle pitch motor is removed (no action, zero force),
        # so the knee motor drives both. Active DOF 12 -> 10. Knee limits become 28 N·m / 27 rad/s / 308 W
        # (spec full_budget: coupled requirement 21.7 N·m / 237 W at 5 kg, x1.3 margin as in §9).
        self.couple_ankle = couple_ankle
        self.tail_idx = np.array([i for i, n in enumerate(self.active) if n.startswith('tail')])
        self.leg_idx = np.array([i for i, n in enumerate(self.active) if not n.startswith('tail')])
        self.policy_idx = self.leg_idx if tail == 'locked' else np.arange(len(self.active))
        if couple_ankle:
            self.policy_idx = np.array([i for i in self.policy_idx if 'ankle_pitch' not in self.active[i]])
        self.lock_tail = tail == 'locked'
        spec = np.array([self.spec_t1[joint_type(n)] for n in self.active])
        if couple_ankle:
            for i, n in enumerate(self.active):
                if 'ankle_pitch' in n:
                    spec[i] = (0., 27., 0.)
                elif 'knee' in n:
                    spec[i] = (28., 27., 308.)
        self.tau_nom, self.v_max, self.p_max = spec[:, 0].copy(), spec[:, 1].copy(), spec[:, 2].copy()
        self.tau_max = self.tau_nom.copy()
        self.kp = self.tau_nom/KP_ERR
        self.kd = self.kp/KD_POLE
        n, na = len(self.active), len(self.policy_idx)
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11+3*(self.obs_vel)+2*n+na,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (na,), np.float32)
        self.resample_steps = 250
        self.model = self.renderer = None

    # --- model -------------------------------------------------------------------------------
    def _build(self):
        spec = mujoco.MjSpec.from_file(self.model_path)
        self.heights = tr.add_terrain(spec, 'flat', 0., self.rng)
        ratio = self.mass/DESIGN_MASS
        for joint, (k, q0, mode) in SPRINGS[self.spring_set].items():
            for side in ('left', 'right'):
                band = [-1e6, q0] if mode == 'uni+' else [q0, 1e6] if mode == 'uni-' else [q0, q0]
                t = spec.add_tendon(name=f'spring_{side}_{joint}', stiffness=k*ratio, springlength=band)
                t.wrap_joint(f'{side}_{joint}_joint', 1.)
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
        m = spec.compile()
        m.opt.timestep = PHYSICS_DT
        # uniform mass scaling, geometry fixed (spec §5): masses and inertias. The R-02 toe springs are kept: scaled
        # with mass they fall below the W*h (5 kg x 0.71 m = 34 N·m/rad) a passive stand on the toes needs, and the
        # robot topples forward in ~3 s (measured 2026-09-29).
        m.body_mass[1:] *= ratio
        m.body_inertia[1:] *= ratio
        self.act = np.array([m.actuator(n).id for n in self.active])
        self.ix = {n: i for i, n in enumerate(self.active)}
        self.q_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.active])
        self.v_adr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.active])
        self.lo, self.hi = m.jnt_range[[m.joint(n).id for n in self.active]].T.copy()
        # PD position servo as an affine actuator: force = kp (ctrl - q) - kd qd, clamped by forcerange each step
        m.actuator_gainprm[self.act, 0] = self.kp
        m.actuator_biasprm[self.act, :3] = np.c_[np.zeros(len(self.act)), -self.kp, -self.kd]
        m.actuator_ctrllimited[self.act] = 1
        m.actuator_ctrlrange[self.act] = np.c_[self.lo, self.hi]
        m.actuator_forcelimited[self.act] = 1
        m.actuator_forcerange[self.act] = np.c_[-self.tau_nom, self.tau_nom]
        m.jnt_actfrclimited[:] = 0
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
        m.geom_friction[sorted(self.ground), 0] = self.rng.uniform(.5, 1.25)
        self.tau_max = self.tau_nom*self.rng.uniform(.9, 1.1)

    def set_curriculum(self, level, cmd_max):
        self.level, self.cmd_max = float(level), np.array(cmd_max, float)

    # --- gym API -----------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.model is None:
            self._build()
        self._randomize()
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        mujoco.mj_setConst(m, d)
        self.weight = 9.81*float(m.body_subtreemass[self.root])
        d.qpos[self.q_adr] = self.q0+(self.rng.normal(0, .02, len(self.q0)) if self.randomize else 0)
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[g][2]-.02 for s in self.foot_geoms for g in self.foot_geoms[s])
        d.qpos[2] += -lowest+.005
        mujoco.mj_forward(m, d)
        self.h_nom = d.xpos[self.base][2]-.03
        d.ctrl[self.act] = d.qpos[self.q_adr]
        self.steps, self.last_action, self.prev_qd = 0, np.zeros(len(self.policy_idx)), d.qvel[self.v_adr].copy()
        self.air = {'left': 0., 'right': 0.}
        self.loaded = {'left': True, 'right': True}
        self.stance_peak = {'left': 0., 'right': 0.}
        self.touchdowns, self.peaks = [], []
        self.ankle_td = {'left': None, 'right': None}
        self.sym_ema = None
        self.leg_scores = {side: dict(fold=0., extend=0., retract=0.) for side in ('left', 'right')}
        self.command = self._sample_command()
        self.phase = float(self.rng.random()) if self.randomize else 0.
        self.push_at = self.rng.uniform(3., 8.)
        self.events = []
        d.xfrc_applied[:] = 0.
        if self.randomize and self.init_speed:
            d.qvel[0] += self.rng.uniform(0., .3*self.command[0])
        return self._observe(), {}

    def _sample_command(self):
        c = self.rng.uniform(-1, 1, 3)*self.cmd_max
        u = self.rng.random()
        c[0] = 0. if u < self.zero_cmd else self.cmd_max[0] if u < self.zero_cmd+self.top_cmd \
            else self.rng.uniform(0., self.cmd_max[0])
        if c[0] == 0.:
            c[1:] = 0.
        return c

    def _observe(self):
        return self._obs()

    def _obs(self):
        d = self.data
        R = d.xmat[self.base].reshape(3, 3)
        gyro = d.qvel[3:6]
        grav = R.T @ np.array([0, 0, -1.])
        parts = [gyro*.25, grav, self.command*np.array([.1, 2., .5])]
        if self.obs_vel:
            parts.append(self.body_velocity(self.base)[1]*.1)
        parts += [d.qpos[self.q_adr]-self.q0, d.qvel[self.v_adr]*.05, self.last_action,
                  [np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase)]]
        return np.concatenate(parts).astype(np.float32)

    # --- virtual actuator ----------------------------------------------------------------------
    def _clamp(self):
        """Force range per actuator for this physics step: torque limit, power limit P/|qd|, no drive beyond v_max."""
        qd = self.data.qvel[self.v_adr]
        lim = np.minimum(self.tau_max, self.p_max/np.maximum(np.abs(qd), 1e-2))
        lo, hi = -lim, lim.copy()
        hi = np.where(qd > self.v_max, 0., hi)
        lo = np.where(qd < -self.v_max, 0., lo)
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
        return float(np.clip(.6-.128*(cx-.5), GAIT_DUTY, .6))

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
        if axis == 'y':
            w[3:] = d.xmat[self.base].reshape(3, 3)[:, 1]*impulse/duration
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
        d.ctrl[self.act] = target
        if self.ankle_clutch:
            for s_, t in self.ankle_spring.items():
                m.tendon_stiffness[t] = self.ankle_k[s_] if self.loaded[s_] else 0.
        n_sub = int(round(CONTROL_DT/PHYSICS_DT))
        power, torque_sq = 0., 0.
        for _ in range(n_sub):
            self._clamp()
            mujoco.mj_step(m, d)
            tau, qd = d.actuator_force[self.act], d.qvel[self.v_adr]
            power += np.sum(np.clip(tau*qd, 0, None))
            torque_sq += np.sum(tau**2)
        power, torque_sq = power/n_sub, torque_sq/n_sub
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
        sigma = max(.3, .15*abs(cx))
        terms['track_lin'] = np.exp(-np.sum((self.command[:2]-v_body[:2])**2)/sigma**2)
        terms['progress'] = np.clip(v_body[0], -1., cx)/max(cx, 1.)
        terms['track_yaw'] = np.exp(-(self.command[2]-w[2])**2/.2)
        terms['yaw_err'] = abs(self.command[2]-w[2])  # linear term: gradient for large turn commands (off by default)
        terms['lin_vel_z'] = v_body[2]**2
        terms['ang_vel_xy'] = np.sum(w[:2]**2)
        terms['orientation'] = np.sum(grav[:2]**2)
        bz = d.xpos[self.base][2]
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
        terms['gait'] = sum((min(feet[s_]/(.2*W), 1.)-sched[s_])**2 for s_ in feet) if sched else 0.
        terms.update(self.kinematics_terms(loaded_now, w[1]))
        terms['leg_sym'] = self.leg_symmetry() if self.command[0] >= .5 else 0.
        terms['stand'] = float(flight) if cx < .05 else 0.  # hopping in place at a zero command (v1 defect)
        terms['air_time'] = air_reward/CONTROL_DT if moving else 0.
        terms['grf'] = grf
        terms['slip'] = slip
        terms['collision'] = float(body_hit)
        q = d.qpos[self.q_adr]
        margin = .9*(self.hi-self.lo)/2
        mid = (self.hi+self.lo)/2
        terms['joint_limit'] = np.sum(np.clip(np.abs(q-mid)-margin, 0, None))
        terms['alive'] = 1.
        reward = CONTROL_DT*sum(self.weights[k]*v for k, v in terms.items())

        tilt = np.arccos(np.clip(-grav[2], -1, 1))
        fallen = (body_hit or tilt > .8 or bz < .5*self.h_nom or not np.isfinite(d.qpos).all()
                  or np.abs(v_body).max() > 30.)
        if fallen:
            reward -= 10.
        self.last_action, self.prev_qd = action, qd.copy()
        truncated = self.steps >= self.episode_steps
        info = {'terms': terms, 'tilt': tilt, 'v_body': v_body, 'command': self.command.copy(), 'kind': 'flat',
                'x': d.xpos[self.base][0], 'power': power, 'grf': {s: feet[s]/W for s in feet}, 'flight': flight,
                'loaded': loaded_now, 'L': L, 'L_tail': L_tail, 'L_legs': L_legs,
                'tail_qd': d.qvel[self.v_adr][self.tail_idx].copy(), 'phase': self.phase}
        return self._observe(), float(reward), bool(fallen), bool(truncated), info
