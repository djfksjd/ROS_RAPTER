#!/usr/bin/env python3
"""Small, bounded simulation-only joint test. No learned motor commands."""
import json
import argparse
import math
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from mission_policy import active_positions
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = [f'{side}_{joint}_joint' for side in ('left', 'right')
          for joint in ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')]
JOINTS += ['tail_yaw_joint', 'tail_pitch_joint']


class Probe(Node):
    def __init__(self):
        super().__init__('raptor_motion_probe')
        self.state = {}
        self.received = 0.
        self.create_subscription(JointState, '/joint_states', self.observe, 10)
        self.client = ActionClient(self, FollowJointTrajectory,
                                   '/raptor_joint_controller/follow_joint_trajectory')

    def observe(self, msg):
        values = active_positions(msg.name, msg.position, JOINTS)
        if values is not None:
            self.state = values
            self.received = time.monotonic()

    def settle(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.05)
        if set(JOINTS) - self.state.keys() or time.monotonic() - self.received > 1:
            raise RuntimeError('Missing or stale joint states')
        if not all(math.isfinite(self.state[j]) for j in JOINTS):
            raise RuntimeError('Nonfinite joint state')

    def move(self, positions, seconds=2):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = [positions[j] for j in JOINTS]
        point.time_from_start.sec = seconds
        goal.trajectory.points = [point]
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10)
        if not future.done() or not future.result().accepted:
            raise RuntimeError('Trajectory not accepted')
        handle = future.result()
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=20)
        if not result.done():
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel, timeout_sec=3)
            raise RuntimeError('Trajectory timed out and cancellation requested')
        if result.result().result.error_code != 0:
            raise RuntimeError(result.result().result.error_string)
        self.settle(.5)
        error = max(abs(self.state[j] - positions[j]) for j in JOINTS)
        if error > .025:
            raise RuntimeError(f'Tracking error {error:.5f} rad')
        return error


def pose():
    data = subprocess.check_output(['gz', 'topic', '-e', '-t',
        '/world/raptor_world/dynamic_pose/info', '-n', '1', '--json-output'], timeout=8)
    # gz transport may deliver a second queued sample even with --num 1.
    msg, _ = json.JSONDecoder().raw_decode(data.decode().lstrip())
    return next(p for p in msg['pose'] if p['name'] == 'raptor')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--standing-only', action='store_true')
    args = parser.parse_args()
    rclpy.init()
    node = Probe()
    report = {'joint_tests': [], 'standing_samples': []}
    try:
        if not node.client.wait_for_server(timeout_sec=15):
            raise RuntimeError('Trajectory server unavailable')
        node.settle(1)
        if 'raptor_mission_gate' in node.get_node_names():
            raise RuntimeError('Stop operator mission gate before a development probe')
        neutral = dict.fromkeys(JOINTS, 0.)
        node.move(neutral)
        for joint in ([] if args.standing_only else JOINTS):
            target = neutral | {joint: .03}
            error = node.move(target)
            report['joint_tests'].append({'joint': joint, 'target': .03,
                'actual': node.state[joint], 'max_error': error, 'pose': pose()})
            node.move(neutral)
        for _ in range(15):
            node.settle(1)
            report['standing_samples'].append(pose())
        report['passed_joint_tracking'] = True
        report['standing_only'] = args.standing_only
        if args.standing_only:
            for sample in report['standing_samples']:
                q = sample['orientation']
                tilt = 2*math.acos(min(1., abs(q.get('w', 1.))))
                if tilt > .1 or abs(sample['position'].get('z', 0)) > .04:
                    raise RuntimeError('Free standing pose exceeded tilt/height limits')
            report['passed_static_standing'] = True
    finally:
        Path('/raptor_ws/log/motion-probe.json').write_text(json.dumps(report, indent=2))
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
