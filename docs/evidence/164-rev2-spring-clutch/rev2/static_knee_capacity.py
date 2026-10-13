"""Knee load per unit vertical foot force versus knee angle (level torso, flat toes: hip fixed at -0.5273 rad, shin world
angle = knee - 0.5273). Closed form of the final report: tau_knee = [F l2 - g (m_shin r_shin + m_distal l2)] sin(theta_shin).
Gives the single-leg vertical force (in body weights) the knee drive holds at its continuous and peak rating without a
spring, and the spring torque of a (k, q0) latch at that angle."""
import json, math, sys
from pathlib import Path
import numpy as np
a_, b_, c_, beta, lf = .083, .194, .102, math.radians(69), .22


def fourbar(q):
    Qx, Qz = c_*math.cos(q+beta), -lf+c_*math.sin(q+beta); d = math.hypot(Qx, Qz)
    phi = math.atan2(Qz, Qx)+math.acos((a_*a_+d*d-b_*b_)/(2*a_*d))
    return c_*(lf*math.cos(q+beta)-a_*math.sin(q+beta-phi))/(a_*(lf*math.cos(phi)-c_*math.sin(q+beta-phi)))
M, g, l2 = 11.2912, 9.81, .38; W = M*g
m_shin, m_dist = .228*.58+.08, .192*.58+.04*.6+6*.01*.6
grav = g*(m_shin*.19+m_dist*.38)
rows = []
for q in (.716, .956, 1.1, 1.2, 1.3, 1.5):
    N = abs(fourbar(q)); s = math.sin(q-.5273)
    cont, peak = .9*19.2*N, .9*48*N
    F = lambda tau: (tau/s+grav)/l2/W
    rows.append(dict(knee=q, N=N, joint_cont=cont, joint_peak=peak, torque_per_BW=(W*l2-grav)*s, F_cont_BW=F(cont), F_peak_BW=F(peak),
                     spring170_from_0p956=168.970423*max(0., q-.956)))
    print({k: round(v, 3) for k, v in rows[-1].items()})
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
