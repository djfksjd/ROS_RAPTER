"""World-referenced gravity tilt guard for bounded simulation probes."""
import math
import time


class TiltGuard:
    def __init__(self, limit=.25, max_age=1., clock=time.monotonic):
        self.limit = limit
        self.max_age = max_age
        self.clock = clock
        self.received = None
        self.tilt = None
        self.failure = None

    def observe(self, x, y, z, w, valid=True):
        self.received = self.clock()
        norm = x*x+y*y+z*z+w*w
        if not valid or not all(math.isfinite(v) for v in (x,y,z,w)) or abs(norm-1)> .02:
            self.failure = 'Invalid IMU orientation'
            return
        self.tilt = math.acos(max(-1., min(1., 1-2*(x*x+y*y)/norm)))
        if self.tilt > self.limit:
            self.failure = f'IMU tilt {self.tilt:.6f} exceeded {self.limit:.3f} rad'

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.received is None or self.clock()-self.received > self.max_age:
            raise RuntimeError('Missing or stale IMU orientation')
