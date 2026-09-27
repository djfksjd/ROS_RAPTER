"""Pure state policy for the simulation command gate."""
class Policy:
    def __init__(self):
        self.stopped = False
        self.paused = False

    def check(self, action):
        if action == 'STOP':
            self.stopped = True
            return 'hold'
        if self.stopped:
            return 'rejected_stop_latched'
        if action == 'PAUSE':
            self.paused = True
            return 'hold'
        if action == 'RESUME':
            if not self.paused:
                return 'rejected_not_paused'
            self.paused = False
            return 'hold'  # Never resume an old trajectory automatically.
        if self.paused:
            return 'rejected_paused'
        if action == 'STAND':
            return 'stand'
        return 'rejected_unverified_mission'


def active_positions(names, positions, required):
    """Ignore partial/passive JointState messages; do not refresh active feedback."""
    import math
    if len(names) != len(positions):
        return None
    values = dict(zip(names, positions))
    if not set(required).issubset(values):
        return None
    if not all(math.isfinite(values[j]) for j in required):
        return None
    return {j: values[j] for j in required}
