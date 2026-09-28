#!/usr/bin/env python3
"""Read-only gait phase estimator: publishes /gait/phase, never commands motors.

Left-foot contact (any pad/toe contact message in the last 0.1 s of sim time) feeds rock_law.GaitPhase
(phase 0 at a left touchdown). Output: std_msgs/Float64MultiArray [phase or -1 if unknown, period s or -1,
duty or -1]. Contact messages are not a force threshold, so the estimate inherits their flicker (evidence 71).
"""
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Float64MultiArray

from rock_law import GaitPhase

RATE = .02  # s of sim time between publications


class GaitPhaseNode(Node):
    def __init__(self):
        super().__init__('raptor_gait_phase')
        self.gait, self.sim_time, self.last, self.contacts = GaitPhase(), None, None, {}
        self.pub = self.create_publisher(Float64MultiArray, '/gait/phase', 10)
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        topics = ['/raptor/left_foot/contact'] + [f'/raptor/left/toe_{d}_{p}/contact'
                                                  for d in range(1, 4) for p in ('proximal', 'distal')]
        for topic in topics:
            self.create_subscription(Contacts, topic, lambda m, k=topic: self.on_contact(m, k), 10)

    def on_contact(self, msg, key):
        if self.sim_time is not None:
            self.contacts[key] = (self.sim_time, len(msg.contacts))

    def on_clock(self, msg):
        self.sim_time = msg.clock.sec+msg.clock.nanosec*1e-9
        if self.last is not None and self.sim_time-self.last < RATE:
            return
        self.last = self.sim_time
        left = any(n > 0 and self.sim_time-t < .1 for t, n in self.contacts.values())
        phase = self.gait.update(self.sim_time, left)
        out = Float64MultiArray()
        out.data = [-1. if v is None else float(v) for v in (phase, self.gait.period, self.gait.duty)]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = GaitPhaseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
