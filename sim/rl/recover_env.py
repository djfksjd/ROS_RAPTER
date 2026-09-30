"""T1 get-up environment: the R-02 (5 kg, springs (c), T1 clamped actuator of run_env) starts lying on the ground
after a random fall and must stand up.

Same model, actuator clamps, springs, observation layout and action mapping as RunEnv (tail included), but:
- reset drops the robot from a random orientation (side, back, front or random) with random joints and lets it settle
  for 1 s, so episodes start in a realistic fallen pose;
- ground contact of the torso/tail does not end the episode; the episode ends with success once the torso is upright
  (tilt < 0.3 rad) and above 80 % of the nominal height for 1 s, or after `episode_s`;
- reward: torso height and uprightness progress, a standing bonus, feet loaded when standing, small torque,
  action-rate and joint-limit penalties.
Every result is a T1 (virtual actuator) result.
"""
import mujoco
import numpy as np

from run_env import RunEnv, CONTROL_DT, PHYSICS_DT

FALL_MODES = ('side', 'back', 'front', 'random')
# 'easy' starts: dropped almost upright (tilt 0.3-0.9 rad) so the last part of getting up is learned first
# (reference-state initialisation). recover_t1_v1 (level rewards only, fallen starts only) plateaued at 0 successes
# after 4M steps while random joint sequences reached 0.5-1.0 m torso height, so the reward now pays progress.
WEIGHTS_RECOVER = dict(height=2., upright=2., d_height=5., d_upright=5., standing=5., feet=1., torque=-1e-5,
                       action_rate=-.02, joint_limit=-2., ang_vel=-.02)


class RecoverEnv(RunEnv):
    def __init__(self, fall_modes=FALL_MODES, easy_frac=.4, episode_s=6., stand_hold_s=1., weights=None, **kw):
        kw.setdefault('cmd_max', (0., 0., 0.))
        kw.setdefault('init_speed', False)
        kw.setdefault('obs_vel', False)
        super().__init__(episode_s=episode_s, **kw)
        self.fall_modes, self.easy_frac = list(fall_modes), easy_frac
        self.stand_hold = int(stand_hold_s/CONTROL_DT)
        self.weights = {**WEIGHTS_RECOVER, **(weights or {})}
        self.disturb_items = set()

    def _sample_command(self):
        return np.zeros(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        m, d = self.model, self.data
        self.h_stand = self.h_nom+.03  # torso height of the nominal stand (reset measured it before the drop)
        mode = (options or {}).get('mode') or ('easy' if self.rng.random() < self.easy_frac else self.rng.choice(self.fall_modes))
        self.mode = mode
        if mode == 'easy':
            a = self.rng.uniform(.3, .9)*self.rng.choice([-1, 1]); roll, pitch = (a, 0.) if self.rng.random() < .5 else (0., a)
        else:
            roll, pitch = {'side': (self.rng.choice([-1, 1])*np.pi/2, 0.), 'back': (0., -np.pi/2), 'front': (0., np.pi/2)}.get(
                mode, (self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-np.pi/2, np.pi/2)))
            roll += self.rng.normal(0, .2); pitch += self.rng.normal(0, .2)
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([roll, pitch, self.rng.uniform(-np.pi, np.pi)]), 'xyz')
        mujoco.mj_resetData(m, d)
        d.qpos[3:7] = q
        d.qpos[2] = .6  # base_root origin: the torso hangs 0.58 m above it in the model frame; drop from ~0.6 m
        joints = self.q0+self.rng.uniform(-.5, .5, len(self.q0))*np.array([1. if 'pitch' in n or 'knee' in n else .3 for n in self.active])
        d.qpos[self.q_adr] = np.clip(joints, self.lo, self.hi)
        d.ctrl[self.act] = d.qpos[self.q_adr]
        mujoco.mj_forward(m, d)
        low = min(d.geom_xpos[g][2]-.03 for g in range(m.ngeom) if m.geom_bodyid[g] != 0)
        d.qpos[2] += -low+.05
        mujoco.mj_forward(m, d)
        for _ in range(int((.3 if mode == 'easy' else 1.)/PHYSICS_DT)):  # settle into the fallen pose with the joints held
            self._clamp(); mujoco.mj_step(m, d)
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        self.steps, self.last_action, self.prev_qd = 0, np.zeros(len(self.policy_idx)), d.qvel[self.v_adr].copy()
        self.stand_count, self.events = 0, []
        self.prev_h, self.prev_up = None, None
        self.obs_buffer = []
        return self._observe(), {'mode': mode}

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, float), -1, 1)
        target = self.q0.copy()
        target[self.policy_idx] = np.clip(self.q0[self.policy_idx]+self.scale[self.policy_idx]*action,
                                          self.lo[self.policy_idx], self.hi[self.policy_idx])
        if self.lock_tail:
            target[self.tail_idx] = self.q0[self.tail_idx]
        d.ctrl[self.act] = target
        torque_sq = 0.
        for _ in range(int(round(CONTROL_DT/PHYSICS_DT))):
            self._clamp(); mujoco.mj_step(m, d)
            torque_sq += np.sum(d.actuator_force[self.act]**2)
        self.steps += 1
        R = d.xmat[self.base].reshape(3, 3)
        grav = R.T @ np.array([0, 0, -1.])
        tilt = float(np.arccos(np.clip(-grav[2], -1, 1)))
        bz = float(d.xpos[self.base][2])
        feet, _ = self._contacts()
        qd = d.qvel[self.v_adr]
        q = d.qpos[self.q_adr]
        standing = tilt < .3 and bz > .8*self.h_stand
        self.stand_count = self.stand_count+1 if standing else 0
        h, up = float(np.clip(bz/self.h_stand, 0, 1)), (1-grav[2])/2
        if self.prev_h is None:
            self.prev_h, self.prev_up = h, up
        terms = {'height': h, 'upright': up,  # upright: 1 upright, 0 upside down
                 'd_height': (h-self.prev_h)/CONTROL_DT, 'd_upright': (up-self.prev_up)/CONTROL_DT,
                 'standing': float(standing), 'feet': float(standing and all(feet[s] > .15*self.weight for s in feet)),
                 'torque': torque_sq/20, 'action_rate': float(np.sum((action-self.last_action)**2)),
                 'ang_vel': float(np.sum(d.qvel[3:6]**2))}
        margin = .9*(self.hi-self.lo)/2; mid = (self.hi+self.lo)/2
        terms['joint_limit'] = float(np.sum(np.clip(np.abs(q-mid)-margin, 0, None)))
        reward = CONTROL_DT*sum(self.weights[k]*v for k, v in terms.items())
        success = self.stand_count >= self.stand_hold
        bad = not np.isfinite(d.qpos).all()
        if success:
            reward += 10.
        if bad:
            reward -= 10.
        self.last_action, self.prev_qd = action, qd.copy()
        self.prev_h, self.prev_up = h, up
        truncated = self.steps >= self.episode_steps
        info = {'terms': terms, 'tilt': tilt, 'height': bz, 'standing': standing, 'success': success, 'mode': self.mode,
                'v_body': np.zeros(3), 'command': self.command.copy(), 'kind': 'flat', 'x': d.xpos[self.base][0]}
        return self._observe(), float(reward), bool(success or bad), bool(truncated), info
