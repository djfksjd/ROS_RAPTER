# tests/test_step_metrics.py
"""Tests for footstep metrics."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from step_metrics import steps, summarize


class StepMetricsTests(unittest.TestCase):
    dt = 0.001

    def signal(self, period=0.5):
        t = np.arange(5000) * self.dt
        loaded = (t % period) < period / 2
        force = np.where(loaded, 100.0, 0.0)
        slip = np.where(loaded, 0.02, np.nan)
        x = np.floor(t / period) * 0.1
        return force, slip, x

    def test_touchdowns_and_steps(self):
        result = steps(*self.signal(), self.dt, 100)
        self.assertEqual(result["count"], 9)
        np.testing.assert_allclose(
            result["touchdown_times"], np.arange(1, 10) * 0.5,
            atol=1e-9, rtol=0,
        )
        np.testing.assert_allclose(
            result["step_lengths"], 0.1, atol=1e-9, rtol=0,
        )
        self.assertAlmostEqual(result["median_step"], 0.1)

    def test_short_spikes_do_not_create_touchdowns(self):
        force, slip, x = self.signal()
        for start in range(300, 5000, 500):
            force[start:start + 20] = 100
        result = steps(force, slip, x, self.dt, 100)
        self.assertEqual(result["count"], 9)
        self.assertEqual(len(result["stances"]), 9)

    def test_chatter_does_not_split_stances(self):
        force, slip, x = self.signal()
        for start in range(100, 5000, 500):
            force[start:start + 10] = 0
        result = steps(force, slip, x, self.dt, 100)
        self.assertEqual(result["count"], 9)
        self.assertEqual(len(result["stances"]), 9)

    def test_slip_and_summary(self):
        # Half-second loaded phases give 0.02 * 0.5 m per full stance.
        force, slip, x = self.signal(period=1.0)
        result = steps(force, slip, x, self.dt, 100)
        self.assertEqual(result["count"], 4)
        for stance in result["stances"]:
            self.assertAlmostEqual(stance["slip_m"], 0.01, delta=1e-6)
        self.assertAlmostEqual(result["median_step"], 0.1)
        summary = summarize(result, result, 2)
        self.assertEqual(
            summary["touchdowns_per_foot_per_cycle"],
            {"left": 2, "right": 2},
        )
        self.assertAlmostEqual(summary["slip_to_step"]["left"], 0.1)
        self.assertAlmostEqual(summary["slip_to_step"]["right"], 0.1)


if __name__ == "__main__":
    unittest.main()
