"""Contact wrench on a free-body robot root, in world coordinates.

mj_contactForce returns force:torque in the contact frame. Positive contact
force acts on geom2; reverse it when the subtree contains geom1 instead.
Internal contacts are excluded from the whole-robot external wrench.
"""
import mujoco
import numpy as np


def contact_wrench(model, data, root):
    if root <= 0 or model.body_rootid[root] != root:
        raise ValueError('root must be the non-world root of the robot kinematic tree')
    center = data.subtree_com[root]
    force = np.zeros(3)
    moment = np.zeros(3)
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        inside1 = model.body_rootid[b1] == root
        inside2 = model.body_rootid[b2] == root
        if inside1 == inside2:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        sign = 1. if inside2 else -1.
        frame = c.frame.reshape(3, 3)
        f = sign*(frame.T @ wrench[:3])
        tau = sign*(frame.T @ wrench[3:])
        arm_moment = np.cross(c.pos-center, f)
        force += f
        moment += tau+arm_moment
        contacts.append(dict(body=int(b2 if inside2 else b1),
                             force_world_N=f.tolist(),
                             direct_moment_world_Nm=tau.tolist(),
                             arm_moment_world_Nm=arm_moment.tolist(),
                             position_world_m=c.pos.tolist()))
    return force, moment, contacts
