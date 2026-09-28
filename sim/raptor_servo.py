"""Gazebo-like position path for the MuJoCo Raptor model.

Mirrors gz_ros2_control 1.2.20: every physics step the velocity command is
-gain*(q_read - q_cmd)*update_rate, where q_read is refreshed only every control
period. The actuator ctrlrange clips it to the URDF velocity limit (like DART SERVO
setCommand). DART's separate velocity-limit constraint on externally driven motion is
not reproduced. STOP holds the last read positions; no learned motor commands here.
"""
import numpy as np
import mujoco

ACTIVE = [f'{s}_{j}_joint' for s in ('left', 'right')
          for j in ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')] + ['tail_yaw_joint', 'tail_pitch_joint']


class GazeboLikeServo:
    def __init__(self, model, gain=.3, update_rate=100.):
        self.model = model
        self.gain, self.update_rate = gain, update_rate
        self.period_steps = max(1, round(1/(update_rate*model.opt.timestep)))
        self.qpos = np.array([model.jnt_qposadr[model.joint(n).id] for n in ACTIVE])
        self.actuators = np.array([model.actuator(n).id for n in ACTIVE])
        self.read = None
        self.command = None
        self.steps = 0

    def set_target(self, positions):
        """positions: dict joint->rad for a subset; others keep the previous command."""
        if self.command is None:
            raise RuntimeError('call step() once before set_target()')
        for name, value in positions.items():
            self.command[ACTIVE.index(name)] = value

    def stop(self):
        self.command = self.read.copy()

    def step(self, data):
        if self.steps % self.period_steps == 0:
            self.read = data.qpos[self.qpos].copy()
            if self.command is None:
                self.command = self.read.copy()
        data.ctrl[self.actuators] = -self.gain*(self.read-self.command)*self.update_rate
        mujoco.mj_step(self.model, data)
        self.steps += 1


class JTCLikeServo(GazeboLikeServo):
    """GazeboLikeServo behind a streamed JointTrajectoryController (rock_gz_probe.py path).

    Targets are sampled every `sample` s (zero-order hold, like the probe's stream); each new goal replaces
    the trajectory and, at every controller tick, the command moves linearly from its value when the goal
    arrived to the goal over `horizon` s. Approximation of JTC single-point replacement, not a JTC port.
    """

    def __init__(self, model, sample=.02, horizon=.02, **kwargs):
        super().__init__(model, **kwargs)
        self.sample, self.horizon = sample, horizon
        self.time, self.last_sample, self.goal, self.segment = 0., -np.inf, None, None

    def set_target(self, positions):
        if self.command is None:
            raise RuntimeError('call step() once before set_target()')
        if self.time-self.last_sample < self.sample-1e-9:
            return
        self.last_sample = self.time
        goal = (self.command if self.goal is None else self.goal).copy()
        for name, value in positions.items():
            goal[ACTIVE.index(name)] = value
        self.goal, self.segment = goal, (self.time, self.command.copy(), goal)

    def stop(self):
        super().stop()
        self.goal, self.segment = self.command.copy(), None

    def step(self, data):
        if self.steps % self.period_steps == 0 and self.segment is not None:
            start, origin, goal = self.segment
            self.command = origin+(goal-origin)*min(1., (self.time-start)/self.horizon)
        super().step(data)
        self.time += self.model.opt.timestep
