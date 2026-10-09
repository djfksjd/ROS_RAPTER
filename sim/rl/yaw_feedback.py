"""Bounded heading-rate command correction for a simulated gait policy.

No motor commands or changed actuator limits. Zero-command behavior and the
fixed feedforward adapter remain unchanged when feedback is disabled.
"""
import math

class YawCommandFeedback:
    def __init__(self, scale=6.4, gain=0., time_constant=.2, limit=8., correction_limit=None):
        if not all(math.isfinite(x) for x in (scale,gain,time_constant,limit)):
            raise ValueError('feedback parameters must be finite')
        if scale<=0 or gain<0 or time_constant<=0 or limit<=0:
            raise ValueError('invalid feedback parameters')
        if correction_limit is not None and (not math.isfinite(correction_limit) or correction_limit<=0):
            raise ValueError("correction limit must be finite and positive")
        self.correction_limit=correction_limit
        self.scale,self.gain,self.time_constant,self.limit=scale,gain,time_constant,limit
        self.filtered_rate=0.

    def update(self, desired, measured, dt=.02):
        if not all(math.isfinite(x) for x in (desired,measured,dt)) or dt<=0:
            raise ValueError('rate and positive timestep must be finite')
        alpha=-math.expm1(-dt/self.time_constant)
        self.filtered_rate += alpha*(measured-self.filtered_rate)
        if self.gain==0:
            return self.scale*desired  # preserve the old fixed adapter exactly
        if desired==0:
            return 0.  # straight/zero-command behavior is retained in this ablation
        correction=self.gain*(desired-self.filtered_rate)
        if self.correction_limit is not None:
            correction=max(-self.correction_limit,min(self.correction_limit,correction))
        command=self.scale*desired+correction
        return max(-self.limit,min(self.limit,command))


def heading_rate(rotation, angular_local):
    """Derivative of atan2 of the body's forward axis, in rad/s.

    Body gyro z alone is not heading rate when the body banks or pitches.
    Uses world angular velocity crossed with the forward axis; no Euler angles.
    """
    import numpy as np
    rotation=np.asarray(rotation).reshape(3,3)
    angular_local=np.asarray(angular_local)
    forward=rotation[:,0]
    derivative=np.cross(rotation@angular_local,forward)
    denominator=forward[0]**2+forward[1]**2
    if denominator<1e-8 or not np.isfinite(denominator):
        raise ValueError('heading is undefined for vertical forward axis')
    return float((forward[0]*derivative[1]-forward[1]*derivative[0])/denominator)
