"""rock_law.targets must follow the sim/rock_probe.py law (lift = 0)."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/raptor_control/scripts'))
from rock_law import (ContactEvents, GaitPhase, SlewLimiter, TailSync, TouchdownPLL, WindowMax, antipump,  # noqa: E402
                      crouch_pose,
                      step_roll,
                      tail_sync_targets, tail_targets, targets)

JOINTS = [f'{s}_{j}_joint' for s in ('left', 'right') for j in ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')]
JOINTS += ['tail_yaw_joint', 'tail_pitch_joint']
POSE = crouch_pose(JOINTS, -.10, .5)


def reference(a, stride, window=.35, ramp=1., f=2.):
    """Inline copy of the rock_probe.py stride branch (lift = 0, smooth_swing False)."""
    out = {}
    cycle = (a*f) % 1
    for side, center in (('left', .28), ('right', .78)):
        offset = (cycle-center+.5) % 1-.5
        u = (offset+window/2)/window
        swing = -stride*(u-.5) if 0 <= u <= 1 else -stride*(.5-((offset-window/2) % 1)/(1-window))
        out[side] = swing*min(1., max(0., a-ramp))
    return out


class RockLawTest(unittest.TestCase):
    def test_before_start_is_the_crouch(self):
        self.assertEqual(targets(POSE, -.1, .08, 2., 1., stride=.05), POSE)

    def test_hip_rolls_mirror(self):
        t = targets(POSE, 1.125, .08, 2., 1.)
        self.assertAlmostEqual(t['left_hip_roll_joint'], -.08*math.sin(2*math.pi*2*1.125))
        self.assertAlmostEqual(t['right_hip_roll_joint'], -t['left_hip_roll_joint'])

    def test_stride_matches_rock_probe_and_keeps_sole_level(self):
        for k in range(200):
            a = 1.5+k*.013
            t, ref = targets(POSE, a, .08, 2., 1., stride=.05), reference(a, .05)
            for side in ('left', 'right'):
                swing = t[f'{side}_hip_pitch_joint']-POSE[f'{side}_hip_pitch_joint']
                self.assertAlmostEqual(swing, ref[side])
                level = sum(t[f'{side}_{j}_joint'] for j in ('hip_pitch', 'knee_pitch', 'ankle_pitch'))
                self.assertAlmostEqual(level, 0.)

    def test_mirror_swaps_sides(self):
        for a in (2.1, 2.23, 2.37, 2.55):
            t, m = targets(POSE, a, .08, 2.5, 1., stride=.08), targets(POSE, a, .08, 2.5, 1., stride=.08, mirror=True)
            self.assertAlmostEqual(m['left_hip_roll_joint'], -t['left_hip_roll_joint'])
            self.assertAlmostEqual(m['right_hip_roll_joint'], -t['right_hip_roll_joint'])
            for j in ('hip_pitch', 'ankle_pitch'):  # same instant: the legs exchange stride roles
                self.assertAlmostEqual(m[f'left_{j}_joint'], t[f'right_{j}_joint'])
                self.assertAlmostEqual(m[f'right_{j}_joint'], t[f'left_{j}_joint'])

    def test_abduction_ramps_in_on_both_hip_rolls(self):
        half = targets(POSE, .5, 0., 2.5, 1., abduction=.04)
        full = targets(POSE, 3., 0., 2.5, 1., abduction=.04)
        self.assertAlmostEqual(half['left_hip_roll_joint'], .02)
        self.assertAlmostEqual(half['right_hip_roll_joint'], .02)
        self.assertAlmostEqual(full['left_hip_roll_joint'], .04)
        self.assertEqual(targets(POSE, 3., 0., 2.5, 1.)['left_hip_roll_joint'], 0.)

    def test_external_cycle_matches_time_clock(self):
        for a in (1.3, 2.07, 3.91):
            self.assertEqual(targets(POSE, a, .08, 2.5, 1., stride=.08), targets(POSE, a, .08, 2.5, 1., stride=.08,
                                                                              cycle=a*2.5))

    def test_pitch_feedback_is_clipped(self):
        t = targets(POSE, 2., .08, 2., 1., pitch=1., feedback=(.5, .05, .15))
        self.assertAlmostEqual(t['left_ankle_pitch_joint']-POSE['left_ankle_pitch_joint'], .15)

    def test_tail_targets_signs_and_limits(self):
        t = tail_targets(.05, roll=.1, pitch=.02, pitch_rate=.1, gains=(1.5, 2.5, 2., .1))
        self.assertAlmostEqual(t['tail_yaw_joint'], 1.5*.05+2.5*.1)
        self.assertAlmostEqual(t['tail_pitch_joint'], 2*.02+.1*.1)
        big = tail_targets(1., roll=1., pitch=1., gains=(3., 3., 3., 0.))
        self.assertEqual((big['tail_yaw_joint'], big['tail_pitch_joint']), (.6, .4))
        self.assertEqual(tail_targets(.05), {'tail_yaw_joint': 0., 'tail_pitch_joint': 0.})

    def test_gait_phase_from_left_touchdowns(self):
        gait, phases = GaitPhase(), []
        for k in range(400):  # 2 Hz gait, left stance 55 % of the cycle, 10 ms samples
            t = k*.01
            phases.append(gait.update(t, (t % .5) < .275))
        self.assertIsNone(phases[10])
        self.assertAlmostEqual(gait.period, .5, places=6)
        self.assertAlmostEqual(gait.duty, .55, delta=.03)
        self.assertAlmostEqual(phases[-1], (3.99 % .5)/.5, delta=.03)

    def test_tail_sync_sign_and_unknown_phase(self):
        left = tail_sync_targets(.55, 0., .1, .3, 0.)  # sin(2*pi*.25) = 1: tail to the left = negative joint
        self.assertAlmostEqual(left['tail_yaw_joint'], -.1)
        self.assertEqual(tail_sync_targets(None, 0., .1, .3, 0.)['tail_yaw_joint'], 0.)

    def test_tail_sync_filter_and_slew_limit(self):
        law = TailSync(0., 0., -.1, tau=.05, max_rate=1.)
        law.update(0., None, 0.)
        out = law.update(.02, None, 10.)  # yaw-rate step: filtered, then slew limited to 1 rad/s * 20 ms
        self.assertAlmostEqual(out['tail_yaw_joint'], .02)
        self.assertLess(law.yaw_rate, 10.)

    def test_step_roll_profile_is_rate_limited(self):
        values = [step_roll(k*.001) for k in range(-100, 4000)]
        self.assertAlmostEqual(max(values), .08)
        self.assertAlmostEqual(min(values), -.08)
        self.assertLessEqual(max(abs(b-a) for a, b in zip(values, values[1:]))/.001, 1.6+1e-6)
        self.assertEqual(step_roll(1.5), 0.)
        self.assertAlmostEqual(step_roll(.3), .08)

    def test_contact_events_debounce_and_hysteresis(self):
        det, events = ContactEvents(), []
        forces = [0.]*5+[50.]*1+[0.]*3+[50.]*5+[30.]*5+[20.]*5  # 10 ms samples
        for k, f in enumerate(forces):
            ev = det.update(k*.01, f)
            if ev:
                events.append((round(k*.01, 2), ev))
        # a single 10 ms spike is ignored; touchdown after 20 ms above 44 N; 30 N (inside the band) keeps contact;
        # liftoff after 20 ms below 24 N
        self.assertEqual(events, [(.11, 'touchdown'), (.21, 'liftoff')])

    def test_window_max_bridges_zero_force_frames(self):
        wm, det, events = WindowMax(.03), ContactEvents(), []
        forces = [0.]*3+[150., 0., 150., 0., 0., 150., 0., 150.]+[0.]*6  # Gazebo-like flicker while loaded
        for k, f in enumerate(forces):
            ev = det.update(k*.01, wm.update(k*.01, f))
            if ev:
                events.append(ev)
        self.assertEqual(events, ['touchdown', 'liftoff'])

    def test_pll_speeds_up_when_touchdown_comes_early_and_clamps(self):
        pll = TouchdownPLL(2.5, k=2.5, clamp=.4)
        t, dt = 0., .001
        forces = {'left': 100., 'right': 100.}
        pll.update(t, forces)
        forces['left'] = 0.
        while pll.c < .45:  # left in the air until the clock shows 0.45 (< nominal 0.595): touchdown is early
            t += dt
            pll.update(t, forces)
        forces['left'] = 100.
        for _ in range(40):
            t += dt
            pll.update(t, forces)
        side, e, f = pll.events[-1][1:]
        self.assertEqual(side, 'left')
        self.assertGreater(e, 0)
        self.assertAlmostEqual(f, 2.5+2.5*e, places=3)
        wide = TouchdownPLL(2.5, k=10., clamp=.4)
        wide.detect['left'].loaded, wide.c, wide.t = False, .1, 0.
        for k in range(30):
            wide.update(.001*k, {'left': 100., 'right': 100.})
        self.assertAlmostEqual(wide.f, 2.9)  # e = 0.495 -> clamped at f0 + 0.4

    def test_antipump_stance_only_mirrored_and_clamped(self):
        self.assertEqual(antipump(1., None, .05), {})
        self.assertAlmostEqual(antipump(.5, 'left', .05)['left_hip_roll_joint'], .025)
        self.assertAlmostEqual(antipump(.5, 'right', .05)['right_hip_roll_joint'], -.025)
        self.assertAlmostEqual(antipump(5., 'left', .05)['left_hip_roll_joint'], .03)
        self.assertAlmostEqual(antipump(.5, 'left', .05, sign=-1)['left_hip_roll_joint'], -.025)

    def test_slew_limiter_limits_and_decays(self):
        lim = SlewLimiter(.5)
        lim.update(0., {})
        self.assertAlmostEqual(lim.update(.02, {'a': .03})['a'], .01)
        self.assertAlmostEqual(lim.update(.04, {'a': .03})['a'], .02)
        self.assertAlmostEqual(lim.update(.06, {})['a'], .01)  # missing key decays toward 0

    def test_pll_ignores_bounces_and_waits_for_ramp_end(self):
        pll = TouchdownPLL(2.5, k=3.5, active_after=1.)
        for t, f in ((0., 100.), (.5, 0.), (.53, 100.), (1.5, 0.), (1.53, 100.), (1.6, 0.), (1.63, 100.)):
            for k in range(30):
                pll.update(t+k*.001, {'left': f, 'right': 100.})
        # touchdown at ~0.55 s is before active_after; ~1.55 s is accepted; ~1.65 s is a bounce (< 0.24 s later)
        self.assertEqual([e[0] for e in pll.events], [1.55])


if __name__ == '__main__':
    unittest.main()
