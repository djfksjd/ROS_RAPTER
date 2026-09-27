"""URDF-derived center of mass and foot-edge geometry for simulation experiments."""
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    result = np.eye(4)
    result[:3, :3] += np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)
    return result


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = [float(v) for v in element.get('xyz', '0 0 0').split()]
        roll, pitch, yaw = map(float, element.get('rpy', '0 0 0').split())
        result = result@rotation([0, 0, 1], yaw)@rotation([0, 1, 0], pitch)@rotation([1, 0, 0], roll)
    return result


class SupportModel:
    def __init__(self, urdf):
        root = ET.fromstring(urdf)
        self.joints = [(j.get('name'), j.find('parent').get('link'),
                       j.find('child').get('link'), origin(j.find('origin')),
                       [float(v) for v in j.find('axis').get('xyz').split()]
                       if j.get('type') == 'revolute' else None)
                      for j in root.findall('joint')]
        self.masses = []
        for link in root.findall('link'):
            inertial = link.find('inertial')
            if inertial is not None:
                self.masses.append((link.get('name'), float(inertial.find('mass').get('value')),
                                    origin(inertial.find('origin'))[:, 3]))
        foot = root.find("link[@name='left_foot_link']/collision")
        half_width = float(foot.find('geometry/box').get('size').split()[1])/2
        half_height = float(foot.find('geometry/box').get('size').split()[2])/2
        self.edge_points = [origin(foot.find('origin'))@np.array([0, y, -half_height, 1])
                            for y in [-half_width, half_width]]

    def margin(self, joints, body_rotation):
        transforms = {'base_root': body_rotation}
        pending = self.joints.copy()
        while pending:
            previous = len(pending)
            for joint in pending[:]:
                name, parent, child, offset, axis = joint
                if parent not in transforms:
                    continue
                transforms[child] = transforms[parent]@offset
                if axis is not None:
                    transforms[child] = transforms[child]@rotation(axis, joints[name])
                pending.remove(joint)
            if len(pending) == previous:
                raise ValueError('URDF is not a connected base_root tree')
        mass = sum(m for _, m, _ in self.masses)
        center = sum(m*(transforms[name]@point)[:3] for name, m, point in self.masses)/mass
        edges = [transforms['left_foot_link']@point for point in self.edge_points]
        edge = min(edges, key=lambda point: point[2])
        return float(center[1]-edge[1])
