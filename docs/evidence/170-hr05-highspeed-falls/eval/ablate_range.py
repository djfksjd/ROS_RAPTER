"""Evaluation-only joint-stop test: run evidence 166 ablate.py with the hip-roll joint stop widened on the outward side
(abduction; both axes are +x, so +q left, -q right) to OUT rad. Command targets stay clipped to the trained +-0.5 band
(self.lo/hi are read before the change), so the policy and its commands are unchanged; only the mechanical stop moves.
argv: OUT then the ablate.py argv (run_dir group cmd episodes [seed0])."""
import runpy, sys
sys.path.insert(0, 'sim/rl')
from run_env import RunEnv
OUT = float(sys.argv.pop(1))
_build = RunEnv._build
def _wide(self):
    _build(self)
    m = self.model
    m.jnt_range[m.joint('left_hip_roll_joint').id, 1] = OUT
    m.jnt_range[m.joint('right_hip_roll_joint').id, 0] = -OUT
RunEnv._build = _wide
runpy.run_path('docs/evidence/166-realism-dr-fall-ablation/eval/ablate.py', run_name='__main__')
