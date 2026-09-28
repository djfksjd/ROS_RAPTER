#!/usr/bin/env python3
"""Servo step-response identification on a fixed base (test_fixture:=true). Simulation only.

For each joint/amplitude: hold zero, command a near-step (single JTC point after `ramp` s),
hold, return. Records every /joint_states and controller_state message with sim stamps.
Bounded amplitudes; no learned commands; refuses to run beside the operator mission gate.
"""
import argparse
import json
import time
from pathlib import Path
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState
from motion_probe import Probe, JOINTS
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


def stamp(msg):
    return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joints', nargs='+', default=['left_ankle_pitch_joint', 'left_knee_pitch_joint'])
    parser.add_argument('--amplitudes', type=float, nargs='+', default=[.05, .15])
    parser.add_argument('--ramp', type=float, default=.01)
    parser.add_argument('--hold', type=float, default=1.5)
    parser.add_argument('--output', default='/raptor_ws/log/servo-id.json')
    args = parser.parse_args()
    if set(args.joints)-set(JOINTS) or not all(0 < abs(a) <= .2 for a in args.amplitudes):
        parser.error('Unknown joint or |amplitude| outside (0, 0.2] rad')
    rclpy.init()
    node = Probe()
    states, commands = [], []
    node.create_subscription(JointState, '/joint_states', lambda m: states.append(
        {'t': stamp(m), **{n: [p, v, e] for n, p, v, e in zip(m.name, m.position, m.velocity, m.effort)
                           if n in args.joints}}), 100)
    node.create_subscription(JointTrajectoryControllerState, '/raptor_joint_controller/controller_state',
        lambda m: commands.append({'t': stamp(m), **{n: [r, o] for n, r, o in zip(
            m.joint_names, m.reference.positions, m.output.positions or m.reference.positions)
            if n in args.joints}}), 100)
    trials = []
    try:
        if not node.client.wait_for_server(timeout_sec=15):
            raise RuntimeError('Trajectory server unavailable')
        node.settle(1)
        if 'raptor_mission_gate' in node.get_node_names():
            raise RuntimeError('Stop operator mission gate before a development probe')

        def send(target, seconds):
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = JOINTS
            point = JointTrajectoryPoint(positions=[target.get(j, 0.) for j in JOINTS])
            point.time_from_start.sec, point.time_from_start.nanosec = int(seconds), int(seconds % 1*1e9)
            goal.trajectory.points = [point]
            future = node.client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, future, timeout_sec=10)
            if not future.done() or not future.result().accepted:
                raise RuntimeError('Trajectory not accepted')
            return time.monotonic()

        send({}, 2.)
        node.settle(3)
        for joint in args.joints:
            for amplitude in args.amplitudes:
                for target in (amplitude, 0.):
                    trials.append({'joint': joint, 'target': target, 'sent_wall': send({joint: target}, args.ramp)})
                    node.settle(args.hold)
    finally:
        Path(args.output).write_text(json.dumps({'arguments': vars(args), 'trials': trials,
            'joint_states': states, 'controller_state': commands}))
        node.destroy_node()
        rclpy.shutdown()
    print(f'wrote {args.output}: {len(states)} joint_states, {len(commands)} controller_state')


if __name__ == '__main__':
    main()
