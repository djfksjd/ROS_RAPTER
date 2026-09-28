"""Roll-phase feedback oscillator for lateral rocking: hip-roll command only, no motor-level AI."""
import numpy as np


def wrap(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class RollPhaseOscillator:
    """Hip-roll oscillator with phase and peak feedback."""

    def __init__(
        self,
        frequency=2.0,
        amplitude=0.08,
        gain=1.0,
        correction_limit=0.5,
        phase_offset=0.0,
        target_peak=0.09,
        amplitude_step=0.005,
        amplitude_limits=(0.05, 0.10),
        filter_tau=0.01,
        dt=0.001,
        enable_after=1.0,
        amplitude_feedback=True,
    ):
        self.omega0 = 2 * np.pi * frequency
        self.amplitude = amplitude
        self.gain = gain
        self.correction_limit = correction_limit
        self.phase_offset = phase_offset
        self.target_peak = target_peak
        self.amplitude_step = amplitude_step
        self.amplitude_limits = amplitude_limits
        self.alpha = dt / (filter_tau + dt)
        self.dt = dt
        self.enable_after = enable_after
        self.amplitude_feedback = amplitude_feedback

        self.theta = 0.0
        self.time = 0.0
        self.roll_f = 0.0
        self.roll_rate_f = 0.0
        self.peak = 0.0
        self.last_peak = 0.0
        self.cycles = 0

    def step(self, roll):
        """Filter roll and advance one step."""
        previous = self.roll_f
        self.roll_f += self.alpha * (roll - self.roll_f)
        rate = (self.roll_f - previous) / self.dt
        self.roll_rate_f += self.alpha * (rate - self.roll_rate_f)
        self.peak = max(self.peak, abs(self.roll_f))

        correction = 0.0
        if self.time >= self.enable_after:
            phi = np.arctan2(self.roll_f, self.roll_rate_f / self.omega0)
            correction = np.clip(
                self.gain * wrap(phi - self.theta - self.phase_offset),
                -self.correction_limit,
                self.correction_limit,
            )

        previous_cycle = int(np.floor(self.theta / (2 * np.pi)))
        self.theta += self.dt * (self.omega0 + correction)
        self.time += self.dt

        cycle = int(np.floor(self.theta / (2 * np.pi)))
        if cycle > previous_cycle:
            self.cycles += cycle - previous_cycle
            self.last_peak = self.peak
            if self.amplitude_feedback:
                self.amplitude = np.clip(
                    self.amplitude
                    + self.amplitude_step * np.sign(self.target_peak - self.last_peak),
                    *self.amplitude_limits,
                )
            self.peak = 0.0

        return self.amplitude * np.sin(self.theta)
