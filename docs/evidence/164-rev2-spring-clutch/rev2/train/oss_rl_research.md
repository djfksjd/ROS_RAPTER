# Open-source tools for the Raptor PPO/MuJoCo workflow — verified facts (checked 2026-10-12)

Method: versions, dates, licences, wheels and `requires_python` come from the PyPI JSON API (`https://pypi.org/pypi/<name>/json`) and the GitHub API, queried today. Documentation quotes come from the raw doc sources on the projects' default branches. Paper quotes come from arXiv PDFs or PubMed that were downloaded and searched. Nothing was installed and no benchmark was run on this machine, so "installs on macOS arm64 / Python 3.14" means "a matching wheel or a pure-Python wheel plus installable dependencies exists on PyPI", not "tested".

Baseline observed in the repo (read-only): `sim/rl/train_run.py` uses `SubprocVecEnv` + `VecMonitor`, `torch.set_num_threads(1)`, `device='cpu'`. `sim/rl/run_env.py` `step()` updates the latch (sets `m.tendon_stiffness[t]`) once per control step, then runs a Python loop of 20 x (`_clamp()`, `mujoco.mj_step`, power/heat accumulation).

---

## 1. Faster simulation/training without an NVIDIA GPU

### mujoco.rollout (part of `mujoco`)
- Provides: native C++ loop of `mj_step` over a batch of initial states and control sequences, with an internal thread pool. Docs: "The rollouts are run in parallel with an internally managed thread pool if multiple MjData instances (one per thread) are passed as an argument."
- Signature: `state, sensordata = rollout.rollout(model, data, initial_state, control)`; `control` is "a `nbatch x nstep x ncontrol` array of controls"; `model` may be "a sequence of homogeneous MjModels of length `nbatch`. Homogeneous models have the same integer sizes, but floating point values can differ."
- Thread pool reuse: "By default `rollout.rollout` creates a new thread pool every call if `len(data) > 1`. To reuse the thread pool over multiple calls use the `persistent_pool` argument." Also `rollout.Rollout(nthread=...)`.
- Licence Apache-2.0. Latest `mujoco` 3.15.0, uploaded 2026-10-05. Wheel `mujoco-3.15.0-cp314-cp314-macosx_11_0_arm64.whl` exists; classifiers list Python 3.10-3.15.
- Limits: controls are open-loop arrays, so a closed-loop policy must call it in 20-step chunks per control step. Per-env latch state would have to be passed as per-env models (allowed, since only float values differ). The per-substep Python work in `run_env.py` (`_clamp()`, power/heat sums) would have to be re-expressed as sensors or post-processing of the returned states. No measurement of the gain for this env exists (UNVERIFIED).
- URL: https://mujoco.readthedocs.io/en/stable/python.html (source: https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/python.rst)

### MJX (`mujoco-mjx`, MJX-JAX)
- Provides: JAX re-implementation of MuJoCo for batched simulation on accelerators.
- Official statement on platforms: "MJX-JAX runs on: Nvidia and AMD GPUs, Apple Silicon, and Google Cloud TPUs."
- Official statement on CPU/single-scene performance: "Simulating a single scene (1 instance of mjData), MJX-JAX can be **10x** slower than MuJoCo, which has been carefully optimized for CPU. MJX-JAX works best when simulating thousands or tens of thousands of scenes in parallel."
- Official CPU reference number: single-humanoid throughput with CPU MuJoCo "on an Apple M3 Max and a 64-core AMD 3995WX ... using `2 x numcore` threads" is "**650K**, **1.8M**" steps per second (the GPU/TPU MJX figures are 950K / 2.7M).
- Feature table lists equality `CONNECT`, `WELD`, `JOINT`, `TENDON` and fixed/spatial tendons as supported.
- Licence Apache-2.0. Latest 3.15.0, 2026-10-05, pure wheel `py3-none-any`; depends on `jax`/`jaxlib` (jaxlib 0.11.2 has `cp314-macosx_11_0_arm64` wheel; jax requires Python >=3.12).
- Limits: the docs do not give any Apple-Silicon speed number for MJX itself; whether JAX uses the Apple GPU is not stated there (UNVERIFIED). Runtime mutation of `tendon_stiffness` from Python and the latch state machine would need a JAX rewrite of the env.
- URL: https://mujoco.readthedocs.io/en/stable/mjx.html (source: https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/mjx.rst)

### MuJoCo Warp (`mujoco-warp`)
- README: "MJWarp is a GPU-accelerated version of the MuJoCo physics simulator, designed for NVIDIA hardware." and "MuJoCo Warp requires an NVIDIA GPU for fast simulation but supports CPU for development and debugging."
- Warp on Mac: "The macOS wheels support CPU execution but not Metal acceleration." (Warp README / installation doc)
- Licence Apache-2.0. Latest 3.15.0, 2026-10-05, pure wheel; classifiers Python 3.10-3.14; needs `warp-lang>=1.15` (warp-lang 1.18.0, 2026-10-05, has `py3-none-macosx_11_0_arm64` wheel).
- Limits: installable on this Mac but CPU-only; unsupported items listed in MJX doc include "All except `IMPLICITFAST` midpoint integrator feature".
- URLs: https://github.com/google-deepmind/mujoco_warp , https://github.com/NVIDIA/warp

### MuJoCo Playground (`playground`)
- README: "A comprehensive suite of GPU-accelerated environments for robot learning research and sim-to-real, built with MuJoCo MJX." Install steps call for "CUDA 12 jax" and "Verify GPU backend: ... should print gpu".
- Licence Apache-2.0. Latest 0.2.0, 2026-03-16 (PyPI and GitHub release v0.2.0). `requires_python >=3.11`; classifiers list only 3.11 and 3.12. Depends on brax (classifiers up to 3.13), flax, mujoco-mjx, warp-lang.
- macOS / Python 3.14: UNCONFIRMED (no macOS or CPU guidance in the README; no 3.14 classifier).
- URL: https://github.com/google-deepmind/mujoco_playground , paper https://arxiv.org/abs/2502.08844

### EnvPool (`envpool`)
- README: "EnvPool is a C++-based batched environment pool with pybind11 and thread pool." "All-platform support: Linux, macOS, and Windows wheels for Python 3.12-3.14". Claimed gain on small machines: "~3x throughput of Python subprocess-based vector env on low resource setup like 12 CPU cores" (benchmark env Ant, not this env).
- Licence Apache-2.0. Latest 1.2.7, 2026-09-15. Wheel `envpool-1.2.7-cp314-cp314-macosx_13_0_arm64.whl` exists.
- Limit: custom environments must be written in C++ and built with Bazel ("To add a new environment in C++ that EnvPool will parallelly run..."; "we use Bazel to manage this project"). A Python Gymnasium env cannot be plugged in.
- URLs: https://github.com/sail-sg/envpool , https://envpool.readthedocs.io/en/latest/content/new_env.html , https://arxiv.org/abs/2206.10558

### Sample Factory (`sample-factory`)
- README: "one of the fastest RL libraries focused on very efficient synchronous and asynchronous implementations of policy gradients (PPO)"; "SF is known to work on Linux and macOS. There is no Windows support at this time."
- Licence MIT. Latest PyPI release 2.1.1, 2023-06-19 (repo still receives pushes, last 2026-10-01; no GitHub release objects).
- Python 3.14: pinned to `numpy<2.0` and `gymnasium<1.0`. numpy 1.26.4 has wheels only for cp39-cp312, so there is no numpy wheel satisfying the pin on Python 3.14. Install on this venv: UNCONFIRMED and unlikely without source builds.
- URL: https://github.com/alex-petrenko/sample-factory

### rsl_rl (`rsl-rl-lib`)
- README: "RSL-RL is a GPU-accelerated, lightweight learning library for robotics research." Paper abstract: "Optimized for GPU-only training".
- Licence BSD-3-Clause. Latest 5.5.1, 2026-09-09. Pure wheel; needs `torch>=2.6.0` (torch 2.14.1 has `cp314-macosx_14_0_arm64`), `tensordict>=0.7.0` (0.14.3 has cp314 mac arm64 wheel), onnx (not checked).
- Useful part for this project: `rsl_rl/extensions/symmetry.py` implements "Symmetry data augmentation and mirror loss" (see section 4).
- Limits: expects its own batched torch `VecEnv` interface, not SB3 `VecEnv`; CPU performance not documented (UNVERIFIED).
- URLs: https://github.com/leggedrobotics/rsl_rl , https://arxiv.org/abs/2509.10771

### skrl
- Repo description: "Modular Reinforcement Learning (RL) library (implemented in PyTorch, JAX, and NVIDIA Warp) with support for Gymnasium/Gym, NVIDIA Isaac Lab, MuJoCo Playground and other environments".
- Licence MIT. Latest 2.1.0, 2026-05-10. Pure wheel, `requires_python >=3.10`, no specific 3.14 classifier (UNCONFIRMED on 3.14, though dependencies have 3.14 wheels).
- Limits: it does not make env stepping faster; no documented CPU speed claim was found (UNVERIFIED).
- URL: https://github.com/Toni-SM/skrl

### SBX (`sbx-rl`)
- README: "Proof of concept version of Stable-Baselines3 in Jax." Implements PPO among others, shares the SB3 API.
- Licence MIT. PyPI latest 0.28.0, uploaded 2026-07-24; the GitHub release published the same minute is tagged v0.28.1 (discrepancy observed, not resolved). Classifiers Python 3.10-3.13 (3.14 UNCONFIRMED). Depends on `stable_baselines3>=2.9.0`, `jax<0.12`, flax, optax, `tfp-nightly` (wheel availability of tfp-nightly not checked).
- Limits: uses the same VecEnv, so env stepping cost is unchanged.
- URL: https://github.com/araffin/sbx

### Stable-Baselines3 itself
- Licence MIT. Latest 2.9.0, 2026-06-15. Pure wheel; classifiers 3.10-3.13 (no 3.14 classifier; the project already runs it on 3.14).
- Documented guidance:
  - SubprocVecEnv docstring: "allowing significant speed up when the environment is computationally complex. For performance reasons, if your environment is not IO bound, the number of environments should not exceed the number of logical cores on your CPU."
  - Start method: "Defaults to 'forkserver' on available platforms, and 'spawn' otherwise."
  - PPO page: "PPO is meant to be run primarily on the CPU, especially when you are not using a CNN. To improve CPU utilization, try turning off the GPU and using `SubprocVecEnv` instead of the default `DummyVecEnv`".
  - RL tips: "you should always do several runs to obtain quantitative results."
- URLs: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py , https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html , https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

### Observation about the current throughput (inference, not a measurement)
670 env steps/s x 20 substeps = about 13,400 `mj_step`/s over 8 processes. The MuJoCo docs quote 650K steps/s for a humanoid on an M3 Max using all threads. Different model and machine, so not comparable one-to-one, but the gap suggests the time is going to Python per-substep work, IPC, or the PPO update rather than to MuJoCo itself. This has not been profiled.

---

## 2. Statistically sound evaluation

### rliable (google-research/rliable)
- README: "`rliable` is an open-source Python library for reliable evaluation, even with a handful of runs". Provides stratified bootstrap confidence intervals, performance profiles, IQM, optimality gap, probability of improvement. Inputs are score matrices of size `(num_runs x num_games)`.
- Paper: Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice", arXiv 2108.13264: "we advocate for reporting interval estimates of aggregate performance and propose performance profiles ... as well as present more robust and efficient aggregate metrics, such as interquartile mean scores".
- Licence Apache-2.0. Latest PyPI 1.2.0, 2024-08-12, sdist only. GitHub repo is archived (API `archived: true`, last push 2024-08-12).
- Python 3.14: classifiers stop at 3.8; it pins `arch<8.0,>=5.3.1`, and arch 7.2.0 has no cp314 wheel (arch 8.0.0 does). So `pip install rliable` on Python 3.14 would need to build arch 7.x from source: UNCONFIRMED.
- Fit: it quantifies uncertainty across training runs (seeds) and tasks. It says nothing about one policy from one training run.
- URLs: https://github.com/google-research/rliable , https://arxiv.org/abs/2108.13264

### Binomial bounds for 0 falls in 299 trials (computed today with Python `math`)
| Method | 95 % upper bound on fall probability |
|---|---|
| Rule of three, 3/n | 1.0033 % |
| Exact form behind it, -ln(0.05)/n | 1.0019 % |
| Clopper-Pearson, one-sided 95 % (1 - 0.05^(1/n)) | 0.9969 % |
| Clopper-Pearson, two-sided 95 % (1 - 0.025^(1/n)) | 1.2262 % |
| Wilson, two-sided 95 % (z^2/(n+z^2), z = 1.95996) | 1.2685 % |
| Wilson, one-sided 95 % (z = 1.64485) | 0.8967 % |

- n = 299 is exactly the smallest n for which 0 failures gives a one-sided 95 % Clopper-Pearson bound <= 1 % (0.99^299 = 0.04954). For <= 0.5 % it takes 598 clean trials, for <= 0.1 % it takes 2995.
- A policy with a true 0.5 % fall rate still passes 0/299 with probability 0.995^299 = 22.3 %.
- These bounds assume independent trials drawn from the conditions of interest. Selecting among several policies on the same 299 seeds makes the bound optimistic for the selected policy.
- Primary references (bibliographic data confirmed through Crossref; full texts not opened, so their content is UNVERIFIED here):
  - Hanley, Lippman-Hand, "If Nothing Goes Wrong, Is Everything All Right?", JAMA 249(13):1743, 1983. https://doi.org/10.1001/jama.1983.03330370053031
  - Clopper, Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial", Biometrika 26(4):404-413, 1934. https://doi.org/10.1093/biomet/26.4.404
  - Wilson, "Probable Inference, the Law of Succession, and Statistical Inference", JASA 22(158):209-212, 1927. https://doi.org/10.1080/01621459.1927.10502953
  - Brown, Cai, DasGupta, "Interval Estimation for a Binomial Proportion", Statistical Science 16(2), 2001. https://doi.org/10.1214/ss/1009213286

---

## 3. Prior art: clutched/latched parallel knee spring, and simulator exploitation

### BirdBot (abstract read via PubMed; full text not opened)
- Badri-Spröwitz, Aghamaleki Sarvestani, Sitti, Daley, Sci Robot 7(64):eabg4055, 2022. https://pubmed.ncbi.nlm.nih.gov/35294220/ , https://doi.org/10.1126/scirobotics.abg4055
- "A spring tendon network rapidly switches the leg's slack segments into a loadable state at touchdown, distributes load among joints, enables rapid disengagement at toe-off through elastically stored energy, and coordinates swing leg flexion."
- "We show reduced knee-flexing torque to a 10th of what is required for a nonclutching, parallel-elastic leg design with the same kinematics".
- "BirdBot demonstrates a mechanism for self-engaging and disengaging parallel elastic legs that are contact-triggered by the foot's own lever-arm action."
- Relevance: the engagement is mechanical and contact-triggered, with feedforward control; no RL and no simulation-fidelity claim in the abstract.

### Cassie leaf springs in RL work (PDFs read)
- Xie et al. 2018, "Feedback Control For Cassie With Deep Reinforcement Learning", https://arxiv.org/abs/1803.05580 : "each leg of Cassie has unactuated spring joints, resulting in a challenging control problem." Reward includes "rspring is an additional term to help stabilize the springs on the shin joints." and "We ignore the passive spring-loaded joints in our reward calculation because the deflections are relatively small due to the spring stiffness."
- Li et al. 2024, "Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control", https://arxiv.org/abs/2401.16889 : "the shin and tarsus joints (q5,6) are passive and connected by leaf springs"; "In the case of Cassie, which features passive joints connected by leaf springs, the introduction of ±20% randomized stiffness for the leaf springs has been important for successful sim-to-real transfer, particularly for skills like running and jumping that involve substantial compression of the springs." The randomization table lists "Spring Stiffness [0.8, 1.2] × default".
- Li et al. 2021, https://arxiv.org/abs/2103.14295 : "There are also four passive joints q5,6 that correspond to the shin and tarsus joints."
- Siekmann et al. 2021 (https://arxiv.org/abs/2011.01387 and https://arxiv.org/abs/2105.08328): a text search of both PDFs found no sentence on how the springs were modelled. How those works treated the springs is UNVERIFIED.

### Parallel elastic actuation in legged robots
- Bjelonic et al., "Learning-based Design and Control for Quadrupedal Robots with Parallel-Elastic Actuators", https://arxiv.org/abs/2301.03509 (PDF read): "We use this framework to optimize the parallel-elastic spring parameters for the knee of our quadrupedal robot ANYmal together with the optimal controller." "Our results show that the new system improves the torque-square efficiency of the robot by 33 % compared to the baseline and reduces maximum joint torque by 30 % without compromising tracking performance." Closest match to this project: RL policy plus a parallel knee spring, validated on hardware (not clutched).
- Zhuang, Wang, Ding, "Kinodynamic Model Predictive Control for Energy Efficient Locomotion of Legged Robots with Parallel Elasticity", https://arxiv.org/abs/2503.05666 (PDF read): "exploits unidirectional parallel springs (UPS) to improve the energy efficiency of dynamic legged robots"; "Simulation results demonstrated a 38.8% reduction in the cost of transport (CoT) for a monoped robot ... preliminary hardware experiments show a 14.8% reduction in energy consumption." Note the gap between simulated and hardware benefit.
- Mazumdar et al., "Parallel Elastic Elements Improve Energy Efficiency on the STEPPR Bipedal Walking Robot", IEEE/ASME TMECH 22(2):898-908, 2017, https://doi.org/10.1109/TMECH.2016.2631170 : existence confirmed through Crossref only; content UNVERIFIED.

### Learned controllers exploiting non-physical energy in simulation
- Lehman et al., "The Surprising Creativity of Digital Evolution", https://arxiv.org/abs/1803.03453 (PDF read):
  - "The physics simulator first used a simple Euler method for numerical integration, which worked well for typical motion. However, with faster motion integration errors could accumulate, and some creatures learned to exploit that bug by quickly twitching small body parts. In effect, they could obtain "free energy," to propel them at unrealistic speeds through the water."
  - "The large, less precise time step allowed the creature's bottom-most cells to penetrate the ground between time steps without the collision being detected, which resulted in an upward force from the physics engine to correct the unnatural state. This corrective force provided "free" energy that enabled the creatures to vibrate and swiftly drift across the ground".
- These are evolutionary-search cases. A documented case specific to PPO/RL legged locomotion exploiting a spring or latch artefact was not found (UNVERIFIED that one exists).

---

## 4. Symmetry and gait-quality aids applicable without changing physics

- Yu, Turk, Liu, "Learning Symmetric and Low-energy Locomotion", https://arxiv.org/abs/1801.08093 (PDF read): "we introduce a new term to the loss function (not the reward function) that encourages symmetric actions"; "without the mirror symmetry loss, the learning process takes significantly more trials and results in asymmetric locomotion."
- Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies", https://arxiv.org/abs/2403.04359 (PDF read): "we investigate two approaches to incorporate symmetry invariance into DRL -- data augmentation and mirror loss function. We provide a theoretical foundation for using augmented samples in an on-policy setting."
- Reference implementation: `rsl_rl/extensions/symmetry.py` (BSD-3-Clause), docstring: "`use_data_augmentation` appends mirrored observation/action pairs to every mini-batch ... `use_mirror_loss` adds an auxiliary MSE term that penalizes the policy for disagreeing with itself when evaluated on mirrored observations." https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/symmetry.py
- Abdolhosseini, Ling, Xie, Peng, van de Panne, "On Learning Symmetric Locomotion", Motion, Interaction and Games 2019, https://doi.org/10.1145/3359566.3360070 : metadata confirmed through Crossref; paper text not opened (content UNVERIFIED). Code repo https://github.com/UBCMOCCA/SymmetricRL exists ("Repo for "On Learning Symmetric Locomotion""), last push 2020-07-25, no licence detected by GitHub.
- Other titles found on arXiv but not read: Su et al., "Leveraging Symmetry in RL-based Legged Locomotion Control", https://arxiv.org/abs/2403.17320 ; Siekmann et al. 2011.01387 (periodic reward composition for gait shaping, abstract only).
- Limit for SB3: both the mirror loss and on-policy augmentation live inside the PPO update, so they need a PPO subclass (override of `train()`), not an env wrapper. Only a mirrored-observation/action "symmetric policy network" trick can be done purely as a wrapper; no maintained SB3 implementation was verified.

---

## Ranked recommendation for this machine (CPU-only Mac, Python 3.14)

Adopt now (no new heavy dependency):
1. Report the fall criterion with the exact binomial bound: 0/299 gives a one-sided 95 % upper bound of 0.997 % (two-sided Clopper-Pearson 1.23 %). It costs nothing and states what 299 clean rollouts can and cannot show.
2. Profile one env process before changing any library. The step rate is far below what MuJoCo alone delivers, so the cheap gains are probably in the Python substep loop; also SB3's own rule allows up to 10 envs on 10 logical cores.

Worth a small prototype after the current run finishes:
3. `mujoco.rollout` for the 299-rollout evaluation or the 20-substep inner loop. Already installed, native threads, per-env model floats allowed. Needs the per-substep Python hooks moved out; gain unmeasured.
4. Mirror-symmetry loss or augmentation, ported from the rsl_rl extension into an SB3 PPO subclass. Physics untouched, but it changes the training algorithm, so it must be compared against the baseline under equal conditions.
5. Leaf-spring lesson from Li et al. 2024: randomize the knee spring stiffness (they used ±20 %) and add an energy audit of the latch on/off switching, since stiffness is toggled at 20 ms resolution.

Not worth adopting now:
- rliable: repo archived, install on 3.14 unconfirmed, and it needs several training seeds per configuration to be meaningful. Revisit when there are at least 3-5 seeds per variant.
- MJX, MuJoCo Warp, MuJoCo Playground: official docs say MJX-JAX can be 10x slower than MuJoCo for one scene, and Warp is CPU-only on macOS. They pay off on NVIDIA hardware and require rewriting the env.
- EnvPool: fast, has cp314 mac wheels, but the env must be rewritten in C++/Bazel.
- Sample Factory: pins numpy<2 and gymnasium<1.0, no path to Python 3.14 wheels.
- rsl_rl / skrl / SBX as trainers: none changes env stepping cost; rsl_rl is "optimized for GPU-only training"; SBX is a self-described proof of concept.
