"""Gymnasium environment: velocity-command locomotion for the 10-DOF Raptor (MuJoCo).

The policy outputs position targets for the active joints (10, or 12 with ankle roll) at 50 Hz; they go through the same
Gazebo-like position servo used for the rocking experiments (velocity limit, effort limit, 30/s gain).
Observations are proprioceptive only (what the ROS robot has): IMU gyro and gravity direction,
joint positions/velocities, last action, command and a gait clock. The language model never produces
these actions; STOP stays a hold outside the policy.

`vel_scale` scales the joint velocity limits to study actuator specs (1.0 = URDF). Results obtained
with vel_scale != 1 describe a different (hypothetical) actuator, not the current robot.
"""
from pathlib import Path
import sys

import gymnasium as gym
import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from stand_check import crouch  # noqa: E402
import terrain as tr  # noqa: E402

DEFAULT = {**crouch(-.1, .5), 'left_hip_roll_joint': 0., 'right_hip_roll_joint': 0.,
           'tail_yaw_joint': 0., 'tail_pitch_joint': 0.}
# rad per unit action, by joint type; the active joints are read from the model's actuators
SCALE = {'hip_roll': .3, 'hip_pitch': .6, 'knee_pitch': .8, 'ankle_pitch': .5, 'ankle_roll': .3,
         'tail_yaw': .4, 'tail_pitch': .3}
MODELS = {10: SIM/'raptor_digitigrade.xml', 12: SIM/'raptor_digitigrade_ankleroll.xml'}


def joint_type(name):
    return name.removesuffix('_joint').removeprefix('left_').removeprefix('right_')
CONTROL_DT = .02
SERVO_DT, SERVO_GAIN = .01, 30.  # gz_ros2_control update 100 Hz, gain 0.3 x 100
# Velocity-actuator stiffness. kv 30 is the value compared against Gazebo in evidence 74; the model
# default 100 makes the foot chatter and fall under 0.02 rad ankle-target noise (measured 2026-09-29).
SERVO_KV = 30.
# Joint speed limits (rad/s) per actuator spec. 'r01a' is the evidence-75 recommendation for jumps and
# running (hypothetical actuator, torque limits unchanged); results with it are not the current robot's.
ACTUATORS = {'urdf': None,
             'r01a': {'hip_roll': 5., 'hip_pitch': 11., 'knee_pitch': 18., 'ankle_pitch': 10., 'ankle_roll': 10.}}

# weights: reward per second (multiplied by CONTROL_DT each step)
WEIGHTS = dict(track_lin=2., track_yaw=1., lin_vel_z=-2., ang_vel_xy=-.05, orientation=-5.,
               height=-20., torque=-2e-5, action_rate=-.02, joint_acc=-2.5e-7, air_time=1.,
               slip=-.2, collision=-5., joint_limit=-5., alive=.5)


class RaptorEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': int(1/CONTROL_DT)}

    def __init__(self, terrain='flat', level=0., cmd_max=(.5, .2, .5), vel_scale=1., episode_s=20.,
                 randomize=True, seed=None, render_mode=None, model_path=None, servo_kv=SERVO_KV,
                 actuator='urdf', dof=10):
        self.kinds = [terrain] if isinstance(terrain, str) else list(terrain)
        self.level, self.cmd_max, self.vel_scale = level, np.array(cmd_max, float), vel_scale
        self.servo_kv, self.actuator = servo_kv, actuator
        self.episode_steps, self.randomize, self.render_mode = int(episode_s/CONTROL_DT), randomize, render_mode
        self.model_path = str(model_path or MODELS[dof])
        probe = mujoco.MjModel.from_xml_path(self.model_path)
        names = [probe.actuator(i).name for i in range(probe.nu)]
        self.active = names
        self.q0 = np.array([DEFAULT.get(n, 0.) for n in names])
        self.scale = np.array([SCALE[joint_type(n)] for n in names])
        n = len(names)
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11+3*n,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (n,), np.float32)
        self.resample_steps = 250  # new random command every 5 s (0 = keep)
        self.model = None
        self.renderer = None

    # --- model -------------------------------------------------------------------------------
    def _build(self):
        spec = mujoco.MjSpec.from_file(self.model_path)
        self.kind = self.rng.choice(self.kinds)
        self.heights = tr.add_terrain(spec, self.kind, self.level, self.rng)
        m = spec.compile()
        m.actuator_ctrlrange[:] *= self.vel_scale
        for i, name in enumerate(self.active):
            speed = (ACTUATORS[self.actuator] or {}).get(joint_type(name))
            if speed:
                m.actuator_ctrlrange[m.actuator(name).id] = [-speed, speed]
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 2] = self.servo_kv, -self.servo_kv
        self.model, self.data = m, mujoco.MjData(m)
        self.close()  # a renderer is bound to the previous model
        self.base = m.body('base_link').id
        self.q_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.active])
        self.v_adr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.active])
        self.act = np.array([m.actuator(n).id for n in self.active])
        self.lo, self.hi = m.jnt_range[[m.joint(n).id for n in self.active]].T
        self.floor = m.geom('floor').id
        self.ground = {self.floor}|{g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        foot_bodies = [b for b in range(m.nbody) if 'foot' in m.body(b).name or 'toe' in m.body(b).name]
        self.foot_geoms = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] in foot_bodies
                               and m.body(m.geom_bodyid[g]).name.startswith(s)] for s in ('left', 'right')}
        self.foot_body = {s: m.body(f'{s}_foot_link').id for s in ('left', 'right')}
        self.nominal_mass = m.body_mass.copy()
        self.nominal_friction = m.geom_friction.copy()
        self.nominal_force = m.actuator_forcerange.copy()

    def _randomize(self):
        m = self.model
        m.body_mass[:] = self.nominal_mass
        m.geom_friction[:] = self.nominal_friction
        m.actuator_forcerange[:] = self.nominal_force
        if not self.randomize:
            return
        m.body_mass[self.base] += self.rng.uniform(-1., 1.5)
        m.geom_friction[sorted(self.ground), 0] = self.rng.uniform(.5, 1.25)
        m.actuator_forcerange[:] *= self.rng.uniform(.9, 1.1)

    def set_curriculum(self, level, cmd_max):
        self.level, self.cmd_max = level, np.array(cmd_max, float)

    # --- gym API -----------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.model is None or len(self.kinds) > 1 or self.kind != 'flat':
            self._build()  # new terrain sample each episode
        self._randomize()
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.q_adr] = self.q0 + (self.rng.normal(0, .02, len(self.q0)) if self.randomize else 0)
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[g][2]-.02 for s in self.foot_geoms for g in self.foot_geoms[s])
        d.qpos[2] += tr.height_at(self.heights, 0., 0.)-lowest+.005
        mujoco.mj_forward(m, d)
        self.h_nom = d.xpos[self.base][2]-tr.height_at(self.heights, 0., 0.)-.03  # allow a little crouch
        self.substeps = int(round(SERVO_DT/m.opt.timestep))
        d.ctrl[:] = 0.
        self.steps, self.last_action, self.prev_qd = 0, np.zeros(len(self.q0)), d.qvel[self.v_adr].copy()
        self.air = {'left': 0., 'right': 0.}
        self.command = self._sample_command()
        self.phase = 0.
        self.push_at = self.rng.uniform(4., 10.)
        return self._obs(), {}

    def _sample_command(self):
        c = self.rng.uniform(-1, 1, 3)*self.cmd_max
        c[0] = self.rng.uniform(-.3*self.cmd_max[0], self.cmd_max[0])
        if self.rng.random() < .1:
            c[:] = 0.
        return c

    def _obs(self):
        d = self.data
        R = d.xmat[self.base].reshape(3, 3)
        gyro = d.qvel[3:6]  # free-joint angular velocity is expressed in the body frame (IMU gyro)
        grav = R.T @ np.array([0, 0, -1.])
        return np.concatenate([gyro*.25, grav, self.command*np.array([2., 2., .5]),
                               d.qpos[self.q_adr]-self.q0, d.qvel[self.v_adr]*.05, self.last_action,
                               [np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase)]]).astype(np.float32)

    def _contacts(self):
        m, d = self.model, self.data
        feet = {'left': 0., 'right': 0.}
        body_hit = False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self.ground:
                other = c.geom2
            elif c.geom2 in self.ground:
                other = c.geom1
            else:
                continue
            side = next((s for s in feet if other in self.foot_geoms[s]), None)
            if side is None:
                body_hit = True
                continue
            f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
            feet[side] += abs(f[0])
        return feet, body_hit

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, float), -1, 1)
        target = np.clip(self.q0+self.scale*action, self.lo, self.hi)
        if self.randomize and abs(d.time-self.push_at) < CONTROL_DT/2:  # one lateral/fore-aft shove
            d.qvel[:2] += self.rng.uniform(-.4, .4, 2)
        # Same law as GazeboLikeServo (positions read every 10 ms, ctrl held in between), batched.
        vmax = m.actuator_ctrlrange[self.act, 1]
        for _ in range(int(round(CONTROL_DT/SERVO_DT))):
            read = d.qpos[self.q_adr]
            d.ctrl[self.act] = np.clip(-SERVO_GAIN*(read-target), -vmax, vmax)
            mujoco.mj_step(m, d, nstep=self.substeps)
        self.steps += 1
        self.phase = (self.phase+CONTROL_DT*1.6) % 1.  # 1.6 Hz nominal stride clock (advisory input)
        if self.resample_steps and self.steps % self.resample_steps == 0:
            self.command = self._sample_command()

        R = d.xmat[self.base].reshape(3, 3)
        v_body = R.T @ d.qvel[:3]
        w = d.qvel[3:6]
        grav = R.T @ np.array([0, 0, -1.])
        qd = d.qvel[self.v_adr]
        torque = d.actuator_force[self.act]
        feet, body_hit = self._contacts()
        weight = 9.81*m.body_subtreemass[1]
        terms = {}
        terms['track_lin'] = np.exp(-np.sum((self.command[:2]-v_body[:2])**2)/.1)
        terms['track_yaw'] = np.exp(-(self.command[2]-w[2])**2/.2)
        terms['lin_vel_z'] = v_body[2]**2
        terms['ang_vel_xy'] = np.sum(w[:2]**2)
        terms['orientation'] = np.sum(grav[:2]**2)
        bx, by, bz = d.xpos[self.base]
        ground = tr.height_at(self.heights, bx, by)
        sag = self.h_nom-(bz-ground)
        terms['height'] = sag**2 if sag > 0 else 0.
        terms['torque'] = np.sum(torque**2)
        terms['action_rate'] = np.sum((action-self.last_action)**2)
        terms['joint_acc'] = np.sum(((qd-self.prev_qd)/CONTROL_DT)**2)
        air_reward, slip = 0., 0.
        moving = np.linalg.norm(self.command[:2]) > .05
        for s in feet:
            loaded = feet[s] > .15*weight
            if loaded:
                if self.air[s] > 0 and moving:
                    air_reward += min(self.air[s], .5)-.25  # reward swings of 0.25..0.5 s
                self.air[s] = 0.
                slip += np.sum(d.cvel[self.foot_body[s]][3:5]**2)
            else:
                self.air[s] += CONTROL_DT
        terms['air_time'] = air_reward/CONTROL_DT if moving else 0.  # event reward, per-second scaled
        terms['slip'] = slip
        terms['collision'] = float(body_hit)
        q = d.qpos[self.q_adr]
        margin = .9*(self.hi-self.lo)/2
        mid = (self.hi+self.lo)/2
        terms['joint_limit'] = np.sum(np.clip(np.abs(q-mid)-margin, 0, None))
        terms['alive'] = 1.
        reward = CONTROL_DT*sum(WEIGHTS[k]*v for k, v in terms.items())

        tilt = np.arccos(np.clip(-grav[2], -1, 1))
        fallen = body_hit or tilt > .8 or bz-ground < .5 or not np.isfinite(d.qpos).all()
        if fallen:
            reward -= 10.
        self.last_action, self.prev_qd = action, qd.copy()
        truncated = self.steps >= self.episode_steps
        info = {'terms': terms, 'tilt': tilt, 'v_body': v_body, 'command': self.command.copy(), 'kind': self.kind,
                'x': bx}
        return self._obs(), float(reward), bool(fallen), bool(truncated), info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
            self.cam = mujoco.MjvCamera()
            self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.6, -15., 130.
        self.cam.lookat[:] = self.data.xpos[self.base]
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
