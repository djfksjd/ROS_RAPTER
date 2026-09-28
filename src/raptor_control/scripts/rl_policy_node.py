#!/usr/bin/env python3
"""Run an exported RL locomotion policy in Gazebo (development experiment; not a mission, not a certificate).

Every 20 ms of sim time: IMU + joint states + a fixed velocity command -> rl_policy.PolicyRunner ->
position targets on the joint trajectory controller (the observation/action contract of sim/rl/raptor_env.py).
The language models never produce these targets. Watchdog -> HOLD measured positions and end: IMU tilt,
stale joint/IMU data, sim clock stall/backwards, nonfinite state, a joint at >= 90% of its velocity limit
for longer than --saturation-hold, or any exception. Refuses to run next to the operator mission gate.
Writes /raptor_ws/log/rl-policy.json with the start/end world pose (gz), so distance and heading are measured.
"""
import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from motion_guard import TiltGuard
from motion_probe import pose
from rl_policy import PolicyRunner

VELOCITY_LIMIT = {'hip_roll': 2.0, 'hip_pitch': 2.5, 'knee_pitch': 3.0, 'ankle_pitch': 2.5, 'ankle_roll': 2.5,
                  'tail_yaw': 2.0, 'tail_pitch': 2.0}
STALE, CLOCK_STALL, PERIOD = .1, 2., .02


def limit_of(joint):
    return next(v for k, v in VELOCITY_LIMIT.items() if k in joint)


class PolicyNode(Node):
    def __init__(self, args):
        super().__init__('raptor_rl_policy')
        self.args, self.policy = args, PolicyRunner(args.policy)
        self.joints_order = self.policy.joints
        self.guard = TiltGuard(limit=args.tilt_limit, clock=lambda: self.sim_time or 0.)
        self.sim_time, self.joints, self.imu, self.joints_at, self.imu_at = None, {}, None, None, None
        self.clock_wall, self.clock_fault = time.monotonic(), None
        self.start, self.last_sent, self.stop_reason, self.over_since = None, -math.inf, None, {}
        self.target, self.rows = dict(zip(self.joints_order, self.policy.q0)), []
        self.pub = self.create_publisher(JointTrajectory, '/raptor_joint_controller/joint_trajectory', 10)
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.create_subscription(Imu, '/raptor/imu', self.on_imu, 10)

    def on_clock(self, msg):
        t = msg.clock.sec+msg.clock.nanosec*1e-9
        if self.sim_time is not None and t < self.sim_time:
            self.clock_fault = f'sim time went backwards {self.sim_time:.3f} -> {t:.3f}'
        if self.sim_time is None or t > self.sim_time:
            self.clock_wall = time.monotonic()
        self.sim_time = t

    def on_joints(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.target and i < len(msg.position):
                self.joints[name] = (msg.position[i], msg.velocity[i] if i < len(msg.velocity) else float('nan'))
        self.joints_at = self.sim_time

    def on_imu(self, msg):
        q, w = msg.orientation, msg.angular_velocity
        self.imu_at = self.sim_time
        self.guard.observe(q.x, q.y, q.z, q.w, msg.orientation_covariance[0] != -1)
        self.imu = ((w.x, w.y, w.z), (q.w, q.x, q.y, q.z))

    def send(self, positions, horizon):
        msg = JointTrajectory()
        msg.joint_names = list(self.joints_order)
        point = JointTrajectoryPoint()
        point.positions = [float(positions[j]) for j in self.joints_order]
        point.time_from_start.nanosec = int(horizon*1e9)
        msg.points = [point]
        self.pub.publish(msg)

    def hold(self):
        positions = {j: self.joints[j][0] if j in self.joints and math.isfinite(self.joints[j][0]) else self.target[j]
                     for j in self.joints_order}
        self.target = positions
        for _ in range(3):  # topic interface has no acknowledgement; repeat the hold
            self.send(positions, .1)
            time.sleep(.05)

    def watchdog(self):
        if self.guard.failure:
            return self.guard.failure
        if self.clock_fault:
            return self.clock_fault
        if time.monotonic()-self.clock_wall > CLOCK_STALL:
            return 'sim clock stalled'
        if self.sim_time-(self.joints_at or -math.inf) > STALE or self.sim_time-(self.imu_at or -math.inf) > STALE:
            return 'stale joint states or IMU'
        for j, (q, v) in self.joints.items():
            if not math.isfinite(q) or not math.isfinite(v):
                return f'nonfinite state {j}'
            if abs(v) >= .9*limit_of(j):
                first = self.over_since.setdefault(j, self.joints_at)
                if self.joints_at-first >= self.args.saturation_hold:
                    return f'velocity saturation {j} {v:.3f} rad/s for {self.joints_at-first:.3f} s'
            else:
                self.over_since.pop(j, None)
        return None

    def step(self):
        t = self.sim_time
        if t is None or len(self.joints) < len(self.joints_order) or self.joints_at is None or self.imu_at is None:
            return False
        if self.start is None:
            if t-self.joints_at > STALE/2 or t-self.imu_at > STALE/2:
                return False
            error = max(abs(self.joints[j][0]-q0) for j, q0 in zip(self.joints_order, self.policy.q0))
            if error > .05:
                raise RuntimeError(f'not at the policy default pose (max error {error:.3f} rad); use crouched_start')
            self.start = t
        phase = t-self.start
        if self.stop_reason is None:
            reason = self.watchdog()
            if reason:
                self.stop_reason = f'{reason} at sim {phase:.3f} s'
                self.hold()
                return True
        if phase > self.args.duration:
            self.hold()
            return True
        if t-self.last_sent < PERIOD:
            return False
        horizon = min(max(self.args.horizon, t-self.last_sent), .1)
        self.last_sent = t
        command = self.args.command if phase >= self.args.settle else (0., 0., 0.)
        q = [self.joints[j][0] for j in self.joints_order]
        qd = [self.joints[j][1] for j in self.joints_order]
        gyro, quat = self.imu
        action = self.policy.act(self.policy.observation(gyro, quat, command, q, qd))
        self.target = dict(zip(self.joints_order, self.policy.targets(action)))
        self.send(self.target, horizon)
        self.rows.append({'t': round(phase, 3), 'tilt': round(self.guard.tilt or 0., 4), 'command': list(command),
                          'gyro': [round(v, 4) for v in gyro], 'q': [round(v, 4) for v in q],
                          'qd': [round(v, 3) for v in qd], 'action': [round(float(v), 4) for v in action]})
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True, help='.npz from sim/rl/export_policy.py')
    parser.add_argument('--command', type=float, nargs=3, default=[.3, 0., 0.], metavar=('VX', 'VY', 'YAW'))
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--settle', type=float, default=1., help='zero command first (s)')
    parser.add_argument('--horizon', type=float, default=.02, help='JTC time_from_start of each target (s)')
    parser.add_argument('--tilt-limit', type=float, default=.6)
    parser.add_argument('--saturation-hold', type=float, default=.2)
    parser.add_argument('--out', default='/raptor_ws/log/rl-policy.json')
    args = parser.parse_args()
    if not (abs(args.command[0]) <= .8 and abs(args.command[1]) <= .3 and abs(args.command[2]) <= .8
            and 1 <= args.duration <= 120 and .01 <= args.horizon <= .1 and .1 <= args.tilt_limit <= .8
            and .03 <= args.saturation_hold <= .5):
        parser.error('outside experimental bounds')
    rclpy.init()
    node = PolicyNode(args)
    error, wall, poses = None, time.monotonic(), {}
    try:
        if 'raptor_mission_gate' in node.get_node_names():
            raise RuntimeError('stop the operator mission gate before a development experiment')
        poses['start'] = pose()
        while not node.step():
            rclpy.spin_once(node, timeout_sec=.005)
            if time.monotonic()-wall > 60*20:
                raise RuntimeError('wall-clock limit')
        poses['end'] = pose()
    except BaseException as exc:  # any failure holds the measured pose
        error = repr(exc)
        try:
            node.hold()
        except Exception:
            pass
    Path(args.out).write_text(json.dumps({'arguments': vars(args), 'stop_reason': node.stop_reason, 'error': error,
                                          'poses': poses, 'wall_s': round(time.monotonic()-wall, 1),
                                          'limitation': 'Gazebo DART, exported MuJoCo-trained policy; not a mission.',
                                          'rows': node.rows}, indent=1))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
