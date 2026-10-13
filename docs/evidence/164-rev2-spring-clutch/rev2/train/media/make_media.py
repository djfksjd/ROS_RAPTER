"""Render deterministic rollouts of the model_rev 2 policies as evidence media (no training, no new acceptance trials).
For each run: 10 s at the given command, init seed 96001, side camera following the trunk. Writes <run>_<cmd>mps.mp4, a 3 s GIF,
a touchdown snapshot and a knee-angle chart, and checks that the rendered rollout equals the same rollout without rendering.
Usage (project root): .venv-sim/bin/python make_media.py <cmd> <run> [<run> ...]"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
ROOT = Path.cwd(); OUT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
FONT = '/Library/Fonts/Arial Unicode.ttf'; font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(), 'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
W, H, SECONDS, SEED = 640, 480, 10., 96001
LABEL = {'rev2_a8_h2': '걸쇠 0.956 rad, 명목 무릎 1.5 rad', 'rev2_n10': '걸쇠 0.956 rad, 명목 무릎 1.0 rad', 'rev2_td340': '접지각 클러치 k 340, 명목 무릎 1.0 rad',
         'rev2_r3_ck7M': '래칫 클러치 k 500, 좌우 대칭 정책, 꼬리·고관절 1.5:1', 'rev2_r4b': '래칫 클러치 k 500, 대칭 정책, 롤 충격 학습'}


def rollout(run, cmd, render):
    a = json.loads((ROOT/'sim/rl/runs'/run/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(a), 'episode_s': SECONDS+5., 'level': 0.}, randomize=False, seed=7, render_mode='rgb_array' if render else None); env.init_seed = SEED
    model, venv = load(str(ROOT/'sim/rl/runs'/run/'model.zip'), env); venv.reset(); venv.reset(); m, d = env.model, env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; obs = venv.normalize_obs(env._obs()[None])
    kq = [env.q_adr[env.ix[f'{s}_knee_pitch_joint']] for s in ('left', 'right')]; rows, frames = [], []
    if render:
        ren = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 2.6, 90., -8.
    for k in range(int(SECONDS/.02)):
        raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
        on = info.get('latched', {}); rows.append([d.time, d.qpos[kq[0]], d.qpos[kq[1]], info['loaded']['left'], info['loaded']['right'], on.get('left', False), on.get('right', False), info['v_body'][0], d.qpos[0], d.qpos[2]])
        if render and k % 2 == 1:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], .55]; ren.update_scene(d, cam); frames.append(ren.render().copy())
        if term: break
    if render: ren.close()
    env.close(); return np.array(rows, float), frames, bool(term), a


def main(cmd, run):
    ref, _, _, _ = rollout(run, cmd, False); rows, frames, fell, a = rollout(run, cmd, True)
    assert rows.shape == ref.shape and np.array_equal(rows, ref), 'rendered rollout differs from the plain rollout'
    so = a['springs_override']; spring = (json.loads(so) if isinstance(so, str) else so)['knee_pitch']; q0 = spring[1]; clutch = spring[3] == 'touchdown'; stem = f'{run}_{cmd:g}mps'
    proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H+80}', '-r', '25', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                             '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p', str(OUT/f'{stem}.mp4')], stdin=subprocess.PIPE)
    f18, f14 = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14); gif = []; snap = None
    td = [i for i in range(251, len(rows)) if rows[i, 3] and not rows[i-1, 3]]
    for n, fr in enumerate(frames):
        i = 2*n+1; r = rows[i]; im = Image.new('RGB', (W, H+80), '#132C42'); im.paste(Image.fromarray(fr), (0, 80)); dr = ImageDraw.Draw(im)
        dr.text((14, 8), f'{run} | 명령 {cmd:g} m/s | seed {SEED}', font=f18, fill='white')
        dr.text((14, 34), f'{LABEL.get(run, "")} | t={r[0]:.2f}s  속도 {r[7]:.2f} m/s', font=f14, fill='#C9DFE6')
        st = lambda j: ('입각' if r[3+j] else '유각')+(' · 스프링 연결' if r[5+j] else '')
        dr.text((14, 56), f'무릎 좌 {r[1]:.2f} rad ({st(0)})  우 {r[2]:.2f} rad ({st(1)})  | MuJoCo 시뮬레이션, 1배속, 실물 아님', font=f14, fill='#C9DFE6')
        proc.stdin.write(np.asarray(im).tobytes())
        if 5. <= r[0] < 8. and n % 2 == 0: gif.append(im.resize((400, 350)).quantize(64))
        if snap is None and td and i >= td[0]: snap = im; im.save(OUT/f'{stem}_touchdown.png')
    proc.stdin.close(); assert proc.wait() == 0
    gif[0].save(OUT/f'{stem}.gif', save_all=True, append_images=gif[1:], duration=80, loop=0, optimize=True)
    # knee angle chart, 1.5 s window
    s = slice(300, 376); t = rows[s, 0]; fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), dpi=150, sharex=True)
    for j, (ax, name) in enumerate(zip(axes, ('왼 무릎', '오른 무릎'))):
        ld, on = rows[s, 3+j] > 0, rows[s, 5+j] > 0
        ax.fill_between(t, .6, 2., where=ld, step='mid', color='#147F86', alpha=.13, label='입각')
        ax.fill_between(t, .6, .68, where=on, step='mid', color='#BC6333', alpha=.9, label='스프링 연결')
        ax.plot(t, rows[s, 1+j], color='#132C42', lw=1.6, marker='o', ms=2.5, label='무릎각')
        if not clutch: ax.axhline(q0, color='#BC6333', ls='--', lw=1, label=f'걸쇠 정지각 {q0} rad')
        ax.set(ylabel=f'{name} (rad)', ylim=(.6, 2.)); ax.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=8, loc='upper right'); axes[1].set_xlabel('시간 (s)')
    axes[0].set_title(f'{run} · 명령 {cmd:g} m/s · {LABEL.get(run, "")}')
    fig.text(.02, .01, f'결정적 rollout, seed {SEED}, 20 ms 표본. 시뮬레이션이며 실물 성능이 아님.', fontsize=8); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(OUT/f'{stem}_knee.png'); plt.close(fig)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = dict(run=run, cmd=cmd, seed=SEED, seconds=float(rows[-1, 0]), fell=fell, frames=len(frames), speed_last5s=float(rows[250:, 7].mean()), spring=spring,
                spring_engaged_stance_frac=float(((rows[250:, 5] > 0) & (rows[250:, 3] > 0)).sum()+((rows[250:, 6] > 0) & (rows[250:, 4] > 0)).sum())/max(1., float((rows[250:, 3] > 0).sum()+(rows[250:, 4] > 0).sum())),
                render_equals_plain_rollout=True, model_sha256=sha(ROOT/'sim/rl/runs'/run/'model.zip'),
                files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.glob(f'{stem}*')) if p.suffix in ('.mp4', '.gif', '.png')})
    (OUT/f'{stem}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1)); print(json.dumps({k: v for k, v in meta.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    for run in sys.argv[2:]:
        main(float(sys.argv[1]), run)
