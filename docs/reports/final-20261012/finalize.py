"""Build report.ko.md (final 1.1) from the integrated 1.0 source: asserted text corrections plus new sections whose numbers
are read from evidence/final/*.json (no hand-copied tables). Run from the project root: python3 <this file>."""
from pathlib import Path
import json, hashlib

HERE = Path(__file__).resolve().parent
SRC = HERE.parent/'RAPTOR_통합문서_20261011'/'report.ko.md'
EV = HERE/'evidence'/'final'
J = lambda n: json.loads((EV/n).read_text())
GR = json.loads((HERE/'evidence'/'gates'/'gate_results.json').read_text()); GC = GR['conditions']
R2 = json.loads((HERE/'evidence'/'rev2'/'rev2_results.json').read_text())
RT = json.loads((HERE/'evidence'/'rev2'/'train'/'train_results.json').read_text())
RN = json.loads((HERE/'evidence'/'rev2'/'train'/'n10_results.json').read_text())
GP = json.loads((HERE/'evidence'/'rev2'/'train'/'geometry_probe_n10.json').read_text())
GY = json.loads((HERE/'evidence'/'rev2'/'train'/'geometry_replay_n10.json').read_text())
RC = json.loads((HERE/'evidence'/'rev2'/'train'/'td340_results.json').read_text())
TL = json.loads((HERE/'evidence'/'rev2'/'train'/'tail_results.json').read_text())
cc, fb, sm, kl, pc = J('crosscheck_pinocchio.json'), J('fourbar_closed_form.json'), J('static_stand_map.json'), J('knee_static_limits.json'), J('parser_check.json')
s = SRC.read_text()
changes = []


def fix(old, new, cid, count=1):
    global s
    assert s.count(old) == count, (cid, s.count(old), old[:60])
    s = s.replace(old, new)
    changes.append({'id': cid, 'old': old, 'new': new})


def insert_page_after(title, page, cid):
    """Insert a new ---PAGE--- block right after the page whose heading is `title`."""
    global s
    pages = s.split('---PAGE---')
    ix = [i for i, p in enumerate(pages) if p.strip().startswith('# '+title)]
    assert len(ix) == 1, (cid, title, ix)
    pages.insert(ix[0]+1, '\n\n'+page.strip()+'\n\n')
    s = '---PAGE---'.join(pages)
    changes.append({'id': cid, 'old': None, 'new': 'new section after: '+title, 'heading': page.strip().splitlines()[0]})


nom, lim, bands = sm['nominal_pose'], kl, kl['fourbar_bands']
f = lambda x, n=4: f'{x:.{n}f}'
row = lambda q: next(r for r in sm['rows'] if abs(r['qk']-q) < 1e-9)
tb = {round(r['q'], 4): r for r in fb['table']}

# ---------------------------------------------------------------- header and key judgement
fix('2026-10-11 | 문서 통합판 1.0 | 기준 코드 6c30fd6 | 우선 검증 후보 scr_a8_h2',
    '2026-10-12 | 최종판 1.8 (통합판 1.0의 정정·보강, 게이트 측정, 모델 개정 2와 그 학습 결과 포함) | 기준 코드 6c30fd6 + 모델 개정 2(미커밋) | scr_a8_h2는 개정 2에서 무효; 개정 2 정책 rev2_a8_h2·rev2_n10·rev2_td340은 과열로 미채택', 'F00')
fix('기존 정적 계산과 시험 기록을 재사용했다. [E1,E2,V1-V4]',
    '기존 정적 계산과 시험 기록을 재사용했다. [E1,E2,V1-V4]\n\n'
    '최종판에서는 1.0의 수치를 모두 다시 계산해 대조했고, 같은 모델을 독립 동역학 라이브러리 Pinocchio '
    f'{cc["pinocchio"]}로 다시 구성해 MuJoCo {cc["mujoco"]}와 비교했다. 4절 링크 전달비와 무릎 정적 유지 토크는 닫힌 식으로 유도해 '
    f'수치 계산과 맞췄다. 이 계산들은 모두 정적 계산이다. 1.2에서는 착수 전 게이트를 닫기 위해 동결된 기준선 정책으로 진단 rollout {GR["timing"]["rollouts"]}회를 실행했다. 1.3에서는 그 결론을 sim/rl/run_env.py의 선택 옵션(model_rev 2)으로 구현하고, 학습 없이 동결 정책·역동역학·낙하 시험으로 검증했다. 기본값은 이전 동작과 같다. 1.3까지는 학습을 하지 않았다. 1.4에서는 사용자 지시로 개정 2에서 학습 1회와 후속 미세조정 1회를 실행하고 고정 시드로 판정했다. 1.5에서는 사용자 선택에 따라 명목 무릎각만 1.0 rad로 옮겨 학습 1회를 더 실행하고 같은 시드로 판정했다. 1.6에서는 학습 없이 그 정책의 접지 기하를 점검했다. 1.7에서는 사용자 선택에 따라 접지각 클러치로 학습 1회를 더 실행하고 같은 시드로 판정했으며, 영상과 차트를 근거로 남겼다. 1.8에서는 학습 없이 그 정책의 꼬리 과열 원인을 조사했다. 정정 내용은 마지막의 정정 이력 표에 있다. [F1,F2,F3,F9,F10]', 'F01')
fix('| 내용 구분 | 이 문서에서의 의미 |',
    '최종판에서 새로 확정한 계산 결과가 하나 있다. 걸쇠 스프링이 연결된 명목 자세(무릎 1.5 rad)는 구동기가 정적으로 유지할 수 없다. '
    f'필요한 무릎 구동 토크 {f(nom["knee_motor_spring_on"], 2)} Nm가 그 자세의 관절 피크 {f(nom["knee_peak"], 2)} Nm의 {f(lim["ratio_nominal_to_peak"], 2)}배, '
    f'가정 연속 토크 {f(nom["knee_cont"], 2)} Nm의 {f(lim["ratio_nominal_to_cont"], 2)}배다. 이것은 정적 계산이며 주행 중 성능이나 기립 실패의 단일 원인을 뜻하지 않는다. [F1]\n\n'
    f'게이트 측정에서 더 큰 문제가 확인됐다. 기준선 정책의 주행은 물리적으로 성립하지 않는 두 가지에 기대고 있다. 걸쇠 스프링이 변형된 상태에서 연결되면서 생기는 에너지가 4 m/s에서 {GC["base@4"]["inject_W"]:.1f} W, 7 m/s에서 {GC["base@7"]["inject_W"]:.1f} W로 모터 양의 기계동력의 {GC["base@4"]["inject_over_motor"]*100:.0f}%와 {GC["base@7"]["inject_over_motor"]*100:.0f}%에 이르고, 이 에너지를 없애면 두 속도 모두 10회 중 10회 전도한다. '
    f'7 m/s에서는 좌우 다리가 서로 통과하며, 자기충돌을 켜면 10회 중 {GC["selfcol@7"]["falls"]}회 전도한다. 따라서 기존 주행 기록을 물리적으로 유효한 성능 근거로 쓸 수 없고, 모델을 고친 뒤 다시 학습해야 한다. [F9]\n\n'
    '| 내용 구분 | 이 문서에서의 의미 |', 'F02')

# ---------------------------------------------------------------- joint range note (R20)
fix('scale: hip roll 0.3 / hip pitch 1.0 / knee 1.0 / tail 0.8 rad\n```',
    'scale: hip roll 0.3 / hip pitch 1.0 / knee 1.0 / tail 0.8 rad\n```\n\n'
    '무릎 모델 상한 2.6 rad에는 실제로 도달할 수 없다. 연동식 q_a = c − q_k(c = 0.0003)와 발목 하한 −2.5 rad 때문에 무릎의 실효 상한은 2.5003 rad다. '
    f'그보다 앞서 4절 링크 폐루프가 {f(fb["closure_upper_rad"], 6)} rad에서 끝난다. 정책 목표 상한 2.5 rad는 두 한계 가운데 링크 쪽을 넘는다. [F1]', 'R20')

# ---------------------------------------------------------------- reset height (R21)
fix('총높이 또한 현재 명목 몸체 원점 높이 0.743674 m와',
    f'총높이 또한 리셋 직후 몸체 원점 높이 0.743674 m(발바닥이 바닥에서 13.0 mm 떠 있는 값이며, 발가락이 닿은 강체 기준은 약 {f(lim["standing_origin_height_rigid"], 4)} m)와', 'R21a')
fix('대칭 중심선과 한쪽 고관절의 위치 차이다. [E19]',
    '대칭 중심선과 한쪽 고관절의 위치 차이다. [E19]\n\n'
    '위 z값은 RunEnv 리셋 직후의 값이다. 리셋은 가장 낮은 발 geom 중심에서 0.02 m를 뺀 높이를 바닥 위 5 mm에 맞추는데, 발가락 상자의 반두께는 0.012 m라서 '
    '가장 낮은 발가락 바닥이 지면에서 12.997 mm 떠 있고 접촉은 0개다. 발가락이 닿은 강체 기준으로는 모든 z가 약 13.0 mm 낮다. x좌표와 상대 위치에는 영향이 없다. [F1]', 'R21b')

# ---------------------------------------------------------------- |D| rendering defect (R19)
fix('|D|≤1은 수학적인 도달 조건이다.', '수학적인 도달 조건은 −1 ≤ D ≤ 1, 곧 절댓값 |D| ≤ 1이다.', 'R19a')
fix('이 값은 자기충돌을 검사했다는 의미가 아니다. [E5,E19]\n| --- | --- |\n', '이 값은 자기충돌을 검사했다는 의미가 아니다. [E5,E19]\n', 'R19b')

# ---------------------------------------------------------------- contact point of the static force (R17)
fix('링크 중력·관성·스프링·COP 오프셋을 빼먹은 최종 모터 요구값으로 사용하지 않는다. [E19]',
    '링크 중력·관성·스프링·COP 오프셋을 빼먹은 최종 모터 요구값으로 사용하지 않는다. [E19]\n\n'
    '정적 평형에서는 지면반력의 작용점(COP)이 MTP가 아니라 전신 COM 바로 아래에 있어야 한다. 이 조건과 링크 중력을 넣어 다시 풀면 '
    f'hip pitch 요구는 {f(nom["hip_pitch_req"], 4)} Nm, 무릎 구동부(연동 포함)가 스프링과 함께 내야 하는 일반화 힘은 {f(nom["knee_load"], 4)} Nm다. '
    '무릎 값의 크기는 위 17.392329 Nm에서 링크 중력 0.854829 Nm를 뺀 것이고, 부호는 폄 방향이다. hip 값이 2.35 Nm에서 크게 줄어드는 것은 작용점이 달라졌기 때문이다. '
    '상세는 「무릎 정적 유지 토크」 절에 있다. [F1]', 'R17a')

# ---------------------------------------------------------------- four-bar text (R18)
b30 = bands['both30'][0]
fix('전달각 30° 이상의 예각 기준을 더하면 약 0.5156–2.3481 rad다. 이 숫자는 표본 그리드상의 경계이며 스토퍼 승인값이 아니다.',
    '출력측 전달각(커플러와 정강이 레버 사이 예각) 30° 이상 기준을 더하면 약 0.5156–2.3481 rad다. 그러나 이 기준은 폐루프 끝에서 생기는 입력측 사점, '
    f'곧 크랭크와 커플러가 일직선이 되는 자세를 걸러내지 못한다. 입력측 각도에도 30°를 요구하면 {f(b30[0], 4)}–{f(b30[1], 4)} rad로 줄어든다. '
    '이 숫자들은 계산상의 경계이며 스토퍼 승인값이 아니다. 다음 절에서 닫힌 식으로 다시 유도한다.', 'R18a')
fix('전달각 30° 조건까지 넣은 표본 범위는 약 0.5156–2.3481 rad이지만, 이 숫자도 바로 하드 스토퍼 치수가 되지 않는다.',
    f'출력측 전달각 30° 조건만 넣은 범위는 약 0.5156–2.3481 rad, 입력측 각도까지 넣은 범위는 {f(b30[0], 4)}–{f(b30[1], 4)} rad이지만, 이 숫자들도 바로 하드 스토퍼 치수가 되지 않는다.', 'R18b')
fix('현재 가장 먼저 동결을 막는 항목은 무릎 목표 2.5 rad가 4절 링크 폐루프 끝 약 2.3481 rad를 넘는다는 점이다.',
    '현재 가장 먼저 동결을 막는 항목은 두 가지다. 무릎 목표 2.5 rad가 4절 링크 폐루프 끝 약 2.3481 rad를 넘고, 걸쇠 스프링이 연결된 명목 자세를 구동기가 정적으로 유지할 수 없다.', 'R18c')

# ---------------------------------------------------------------- spring preload pointer (R17)
fix('에너지는 저장된 탄성 위치에너지이며 스프링이 새로운 순에너지를 만드는 것은 아니다. [E1,E19]',
    '에너지는 저장된 탄성 위치에너지이며 스프링이 새로운 순에너지를 만드는 것은 아니다. [E1,E19]\n\n'
    'RunEnv는 리셋 때 양발을 loaded=True로 두므로 첫 제어 주기부터 스프링이 연결된다. 이때 명목 무릎각을 정적으로 유지하는 데 필요한 모터 토크는 구동기 한계를 넘는다. '
    '수치는 「무릎 정적 유지 토크」 절에 있다. [F1]', 'R17b')

# ---------------------------------------------------------------- headwind wording (R22)
fix('로봇 전진에 필요한 추가 기계동력은 19.845 W다. 여기서 D×9는',
    '이 항력을 이기며 지면 속도 4 m/s로 전진하는 데 드는 공력 기계동력은 D×4 = 19.845 W다. 같은 CdA의 무풍 조건 3.92 W보다 15.925 W 많다. 여기서 D×9는', 'R22')

# ---------------------------------------------------------------- new section: four-bar closed form
T = [tb[k] for k in (0.25, 0.5156, 0.716, 1.5, 2.0, 2.2454, 2.3, 2.34, 2.348)]
fb_rows = '\n'.join(f'| {r["q"]:.4f} | {r["N"]:+.5f} | {r["mu_out_deg"]:.2f} | {r["mu_in_deg"]:.2f} | {43.2*r["N_abs"]:.2f} |' for r in T)
insert_page_after('4절 링크: 가능한 범위 재계산', f'''
# 4절 링크 닫힌 해와 양쪽 사점

![그림 F1. 닫힌 식으로 계산한 전달비 크기와 양쪽 예각. 초록은 두 각이 모두 30° 이상인 구간, 빨강은 폐루프 불성립 구간.](assets/fourbar-both-angles.png)

```text
폐루프 식: g(phi,q) = ||Q(q) − P(phi)||² − b² = 0
N = d phi/d q = −(∂g/∂q)/(∂g/∂phi)
  = c [l_f cos(q+beta) − a sin(q+beta−phi)]
    / ( a [l_f cos(phi) − c sin(q+beta−phi)] )
|N| = (c/a) × sin(mu_out) / sin(mu_in)
mu_out: 커플러–정강이 레버 사이 예각 (출력측 전달각)
mu_in : 크랭크–커플러 사이 예각 (입력측)
```

위 식은 SymPy로 폐루프 식을 음함수 미분해 얻었고, 각도 표현과의 차이는 기호 계산에서 정확히 0이다. 기존 코드는 phi를 수치 미분하므로 사점 근처에서 값이 조금 다르다. 표의 값은 닫힌 식이다. [F1,F3]

| 무릎 q rad | N (부호 포함) | mu_out ° | mu_in ° | 모델 관절 피크 Nm |
| --- | --- | --- | --- | --- |
{fb_rows}

전달비는 두 가지 방식으로 무너진다. mu_out이 0이 되면 N이 0이 되어 모터가 무릎에 토크를 전달하지 못한다. 이 출력측 사점은 q = {f(fb["output_dead_point_rad"], 4)} rad에 있다. mu_in이 0이 되면 N이 무한대로 발산한다. 이 입력측 사점이 폐루프 끝 q = {f(fb["closure_upper_rad"], 6)} rad다. 출력측 전달각만 보는 기준은 두 번째를 놓친다. [F1]

| 기준 | 무릎 유효 구간 rad |
| --- | --- |
| 폐루프 성립 (모델 하한 0.2부터) | 0.2000–{f(fb["closure_upper_rad"], 6)} |
| 출력측 30° 이상 | {f(bands["out30"][0][0], 4)}–{f(bands["out30"][0][1], 4)} |
| 양쪽 30° 이상 | {f(b30[0], 4)}–{f(b30[1], 4)} |
| 양쪽 40° 이상 | {f(bands["both40"][0][0], 4)}–{f(bands["both40"][0][1], 4)} |
| 양쪽 45° 이상 | {f(bands["both45"][0][0], 4)}–{f(bands["both45"][0][1], 4)} |

양쪽 30° 구간 안에서 |N|은 {f(lim["N_range_in_both30"][0], 4)}–{f(lim["N_range_in_both30"][1], 4)}이고 mu_out의 최솟값은 {f(lim["mu_out_min_in_both30"], 2)}°다. 30°는 흔히 쓰는 설계 관례이며 물리 법칙이 정한 값이 아니다. 실제 한계는 공차·마찰·핀 유격과 함께 정해야 한다.

현재 모델은 |N|을 0.5–5로 자르고 불성립 구간에서는 1로 둔다. 그 결과 2.30–2.348 rad에서 모델의 무릎 관절 피크가 103.8 Nm에서 216 Nm까지 부풀고, 같은 구간의 관절 속도 한계는 6.6 rad/s까지 줄어든다. 2.348 rad를 넘으면 피크가 43.2 Nm로 갑자기 떨어진다. 정책 목표 범위 0.5–2.5 rad는 이 구간들을 모두 포함한다. 기존 정책이 이 구간을 얼마나 사용했는지는 이번에 계측하지 않았다. [F1,E1]
''', 'N01')

# ---------------------------------------------------------------- new section: knee static hold
qs = (0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)
hold_rows = '\n'.join(f'| {q:.2f} | {row(q)["knee_motor_spring_on"]:+.2f} | {row(q)["knee_motor_spring_off"]:+.2f} | {row(q)["knee_peak"]:.2f} | {row(q)["knee_cont"]:.2f} | {row(q)["com_x_minus_mtp"]*1000:+.1f} |' for q in qs)
hr = sm['holdable_knee_ranges_rad']; e148 = sm['evidence148_crosscheck']
K1 = 2*lim['Fz']*.38-(lim['Fz']*.38-lim['K_load'])
insert_page_after('스프링 연결·해제 에너지', f'''
# 무릎 정적 유지 토크: 닫힌 식과 한계

![그림 F2. 수평 몸통·평평한 발 조건에서 무릎각별 정적 유지에 필요한 무릎 구동 토크와 가정 구동 한계. 빨강은 스프링 연결, 점선은 해제.](assets/knee-static-hold.png)

```text
정적 양발 지지, 수직 지면반력 F_z = W/2 = {lim["Fz"]:.6f} N/발
연동 q_a = c − q_k 를 가상일에 넣으면 무릎 구동 좌표의 일반화 힘은
tau_kd = Q_k − Q_a = −K_load sin(theta_2W)
K_load = F_z l_2 − g (m_shin r_shin + m_distal l_2)
       = {lim["Fz"]:.6f}×0.38 − 9.81×({lim["m_shin"]:.5f}×0.19 + {lim["m_distal"]:.5f}×0.38)
       = {lim["K_load"]:.6f} Nm
theta_2W = 몸통 pitch + q_h + q_k  (세계 기준 정강이 각)
tau_spring = −k (q_k − q0),  k = {sm["constants"]["k_spring"]:.6f},  q0 = {sm["constants"]["q0"]}
tau_motor = tau_kd − tau_spring = k (q_k − q0) − K_load sin(theta_2W)
```

m_distal은 부척골·발·발가락 6마디의 합이다. 연동 때문에 무릎이 움직여도 부척골과 발은 회전하지 않고 발목점과 함께 평행이동한다. 그래서 그 무게와 지면반력은 무릎 좌표에서 발목점에 작용하는 것과 같고, 식에 발의 COP 위치가 들어가지 않는다. 가정은 정지 상태, 좌우 동일 하중, 수직 지면반력이다. [F1]

명목 자세에서는 theta_2W = 0.9727 rad이므로 tau_kd = {f(nom["knee_load"], 4)} Nm, tau_spring = {f(nom["spring"], 4)} Nm, tau_motor = {f(nom["knee_motor_spring_on"], 4)} Nm다. 닫힌 식과 Pinocchio 전신 정역학 계산은 {abs(lim["closed_form_nominal"]-lim["pinocchio_nominal"]):.1e} Nm 이내로 같다. 같은 자세의 관절 피크는 0.9×48×{f(nom["N_abs"], 6)} = {f(nom["knee_peak"], 4)} Nm, 가정 연속 토크는 0.9×19.2×{f(nom["N_abs"], 6)} = {f(nom["knee_cont"], 4)} Nm다. 스프링을 해제하면 필요한 모터 토크는 {f(abs(nom["knee_motor_spring_off"]), 4)} Nm로 연속 토크의 {abs(nom["knee_motor_spring_off"])/nom["knee_cont"]*100:.1f}%다. [F1]

| 무릎 q_k rad | 연결 시 모터 Nm | 해제 시 모터 Nm | 관절 피크 Nm | 가정 연속 Nm | COM−MTP mm |
| --- | --- | --- | --- | --- | --- |
{hold_rows}

표는 몸통 수평, 발이 평평해지는 hip pitch {f(sm["constants"]["flat_foot_hip_pitch"], 4)} rad 조건이다. 이 조건에서 스프링이 연결된 채 연속 토크 안에 드는 무릎각은 {f(hr["spring_on_cont"][0][0], 3)}–{f(hr["spring_on_cont"][0][1], 3)} rad, 피크 안에 드는 무릎각은 {f(hr["spring_on_peak"][0][0], 3)}–{f(hr["spring_on_peak"][0][1], 3)} rad다. 모터 토크가 0이 되는 무릎각은 {f(sm["spring_only_equilibrium_knee"], 4)} rad다. [F1]

자세와 무관한 상한도 나온다. sin(theta_2W)는 1을 넘지 못하므로 스프링이 연결된 양발 정적 지지에서 연속 토크로 버틸 수 있는 무릎각은 최대 {f(lim["best_case_upper_knee_cont"], 4)} rad, 피크로 버틸 수 있는 무릎각은 최대 {f(lim["best_case_upper_knee_peak"], 4)} rad다. 명목 1.5 rad는 어떤 몸통·고관절 자세에서도 최소 {f(lim["nominal_minimum_motor_any_posture"], 3)} Nm가 필요해 피크 {f(nom["knee_peak"], 3)} Nm를 넘는다. 한 발 지지(F_z = W)로 바꿔도 K_load가 {K1:.4f} Nm가 되어 최소 {sm["constants"]["k_spring"]*.544-K1:.3f} Nm가 필요하므로 결론은 같다. [F1]

보관된 증거148의 정적 평형 자세(몸통 pitch {e148["pitch"]:.5f}, q_h {e148["qh"]:.5f}, q_k {e148["qk"]:.5f} rad)에 이 식을 넣으면 {f(e148["this_knee_torque"], 4)} Nm가 나온다. 당시 MuJoCo 역동역학 기록은 {f(e148["archived_knee_torque"], 4)} Nm로, 발가락 변형을 포함한 계산과 {abs(e148["this_knee_torque"]-e148["archived_knee_torque"]):.4f} Nm 차이다. 그 자세의 무릎각은 위 상한 {f(lim["best_case_upper_knee_cont"], 4)} rad 안에 있다. 증거148은 그 평형에서 시작한 고정 목표 실행이 6.4–6.5초에 전도했다고 기록했다. 정적으로 버틸 수 있는 것과 안정한 것은 다르다. [F1,F5]

이 결과에서 고려할 수 있는 선택지는 세 가지다. 정지 시작 자세를 무릎 {f(lim["best_case_upper_knee_cont"], 3)} rad 이하의 평형군으로 옮기거나, 리셋 때 걸쇠를 해제 상태로 시작하거나, 스프링의 강성과 정지각을 다시 정하는 것이다. 어느 것도 이번에 채택하거나 모델에 적용하지 않았다. 셋 다 모델을 바꾸므로 새 버전으로 만들고 기존 주행 성능의 퇴행을 따로 확인해야 한다.
''', 'N02')

# ---------------------------------------------------------------- new section: foot moment balance
fm, nfm = lim['e148_foot_moment'], lim['nominal_foot_moment']; m2 = sm['map2d']
insert_page_after('무릎 정적 유지 토크: 닫힌 식과 한계', f'''
# 발 모멘트 평형과 명목 자세

```text
발 기울기: phi_f = 몸통 pitch + q_h + c + beta_foot
beta_foot = {sm["constants"]["beta"]:.6f} rad  (foot_link 고정 장착 회전)
평평한 발: phi_f = 0  →  몸통 수평이면 q_h = {sm["constants"]["flat_foot_hip_pitch"]:.6f} rad
MTP 기준 모멘트 평형 (발 하나): F_z × s = 발가락 스프링 토크의 합
s = x_COM − x_MTP  (전방 +)
```

8축 연동에서는 q_h + q_k + q_a가 q_h + c로 고정되므로 발의 기울기를 hip pitch와 몸통 pitch가 정한다. 발목으로 따로 조절할 수 없다. 명목 자세의 phi_f는 0.000101 rad로 사실상 평평하다. [F1]

발이 평평하고 발가락이 변형되지 않았다면 발가락 스프링 토크는 0이다. 이상적인 강체 접촉에서는 이때 발가락이 하중을 받을 수 없고, 하중은 MTP 뒤 5 mm에 있는 패드 구에만 실린다. 그런데 명목 자세의 전신 COM은 MTP보다 {nfm["s"]*1000:.3f} mm 앞에 있다. 발 하나당 {f(nfm["F_times_s"], 4)} Nm(패드 중심 기준으로는 {f(nfm["F_times_s_from_pad"], 4)} Nm)의 모멘트가 남으므로 명목 자세는 이 점에서도 정적 평형이 아니다. 몸이 앞으로 기울어 발가락이 굽어야 균형이 잡힌다. MuJoCo의 부드러운 접촉에서는 약간의 침투로 발가락이 하중을 받을 수 있어 이 설명은 이상화다. [F1]

증거148의 평형 자세가 이를 보여 준다. 그 자세의 발 기울기는 {f(fm["foot_tilt_rad"], 4)} rad(약 {fm["foot_tilt_rad"]*57.29578:.2f}°)이고 III·IV 뿌리 마디가 각각 0.0777, 0.0603 rad 굽어 있다. F_z × s는 {f(fm["F_times_s"], 4)} Nm이고, 뿌리 스프링 토크의 합은 30×0.0777 + 20×0.0603×cos 24° = {f(fm["sum_toe"], 4)} Nm다. 두 값의 차이는 {abs(fm["F_times_s"]-fm["sum_toe"])/fm["F_times_s"]*100:.1f}%다. s는 발가락을 강체로 둔 COM으로 계산했고 패드와 원위 마디의 몫은 따로 나누지 않았으므로 이 비교는 근사적인 정합성 확인이다. [F1,F5]

몸통이 수평이고 발이 평평하면 COM이 발 위(패드 −5 mm부터 III 끝 +100 mm 사이)에 오는 무릎각은 약 {f(sm["level_flat_support_knee_range"][0], 3)} rad 이상이다. 앞 절의 표에서 s는 1.30 rad일 때 {row(1.3)["com_x_minus_mtp"]*1000:+.1f} mm, 1.50 rad일 때 {row(1.5)["com_x_minus_mtp"]*1000:+.1f} mm다. 반면 스프링이 연결된 채 연속 토크로 버틸 수 있는 무릎각은 {f(hr["spring_on_cont"][0][1], 3)} rad 이하다. 두 구간은 겹치지 않는다. 무릎 0.60–2.30 rad(0.02 간격)와 몸통 pitch ±0.30 rad(0.01 간격)의 {m2["points"]:,}개 격자에서, 발이 평평하고 COM이 지지 구간 안쪽 10 mm 이상에 있으며 스프링이 연결된 채 무릎이 연속 토크 안에 드는 점은 하나도 없었다. 같은 격자에서 스프링을 해제하면 {m2["spring_off_within_cont"]["n"]}개 점이 조건을 만족한다. [F1]

따라서 스프링이 연결된 정지 자세는 발을 기울이고 발가락 스프링을 굽힌 자세여야 한다. 증거148의 평형 후보가 그런 자세다. 이 절의 계산은 발가락을 강체로 두었으므로 기울어진 발의 평형 자세를 직접 찾지는 않는다. 그 탐색은 접촉과 발가락 변형을 포함한 기존 증거148의 방법이 맞다.
''', 'N03')

# ---------------------------------------------------------------- new section: independent cross-check + tools
mx = cc['max']
insert_page_after('학습 이전의 설계 동결 절차', f'''
# 시뮬레이션 착수 전 게이트

문서가 확정한 것은 현재 모델의 수치와 그 한계다. 아래 표에서 현재 상태가 실패나 미구현인 항목은 코드 결함이며 문서 수정으로 닫히지 않는다. 이번 작업에서 런타임 코드는 바꾸지 않았다. 표의 통과 기준은 다음 시뮬레이션 버전을 만들 때 그대로 검사 항목으로 쓸 수 있도록 수치로 적었다. [F1,V2,V3]

| 게이트 | 통과 기준 | 현재 상태 |
| --- | --- | --- |
| G1 모델 일치 | 모델을 바꾼 뒤 Pinocchio 대조의 최대 오차가 10^-9 이하 | 통과 (최대 {mx["gravity_bias_abs_err"]:.1e}) |
| G2 접촉 마찰 | 무작위화한 범위 0.5–1.25가 mjData.contact.friction에 실제로 나타남 | 실패 (발 접촉은 0.8 또는 1.2 고정) |
| G3 자기충돌 | 다리–다리, 꼬리–다리, 다리–몸통 접촉 쌍이 활성 (모델링 예: [F8]) | 미구현 (로봇 geom 24개 모두 conaffinity 0) |
| G4 무릎 범위 | 정책 목표 범위가 링크 유효 구간 안. 후보: 양쪽 30° 구간 {f(b30[0], 3)}–{f(b30[1], 3)} rad | 실패 (현재 0.5–2.5 rad) |
| G5 걸쇠 시작 상태 | 정지 시작 자세의 무릎 유지 토크가 가정 연속 토크 이하 | 실패 (명목 {f(nom["knee_motor_spring_on"], 2)} Nm, 연속 {f(nom["knee_cont"], 2)} Nm) |
| G6 스프링 전환 에너지 | 연결·해제 때 새로 생기는 에너지가 0 | 실패 (계측: 주입 {GC["base@4"]["inject_W"]:.1f}–{GC["base@7"]["inject_W"]:.1f} W) |
| G7 시작 높이 | 접지 상태에서 시작 | 낙하 포함 (접촉 0개에서 시작; 영향은 작음) |
| G8 로터 관성 | 구동 8축의 armature에 반사 관성 입력 | 0 (잠정값으로 영향 계측) |
| G9 판정 통계 | 사전에 고정한 독립 시드와 횟수, 실패 포함 전체 보고 | 개발 세트 기록만 있음 |

각 게이트를 기준선 정책으로 측정한 결과와 결론은 다음 절에 있다.

수식·좌표·질량·정적 한계는 이 문서에서 닫혔다. 남은 것은 위 게이트의 코드 수정과, 실제 부품이 정해져야 채울 수 있는 값이다. 게이트를 닫지 않고 학습을 시작하면 그 결과는 지금과 같은 모델 한계를 그대로 갖는다.
''', 'N04')

pm, dk = pc['pinocchio'], pc['drake']
insert_page_after('시뮬레이션 착수 전 게이트', f'''
# 독립 엔진 교차검증

같은 모델을 다른 라이브러리로 다시 계산해 MuJoCo 값과 비교했다. RunEnv가 실행 시점에 바꾸는 질량·골반 위치·연동을 모두 반영하도록, 컴파일된 모델의 몸체 트리·질량·관성·관절축·armature를 내보낸 뒤 Pinocchio 모델을 처음부터 구성했다. 비교한 자세는 명목 자세 하나와 관절 범위 안에서 무작위로 뽑은 5개(몸통 기울기 최대 0.6 rad 포함)다. [F1,F2]

| 비교 항목 | 6개 자세의 최대 차이 |
| --- | --- |
| 전체 body 위치 | {mx["max_body_pos_err_m"]:.2e} m |
| 전체 body 회전행렬 성분 | {mx["max_body_rot_err"]:.2e} |
| 전신 COM | {mx["com_err_m"]:.2e} m |
| 질량행렬 30×30 성분 | {mx["mass_matrix_abs_err"]:.2e} |
| 중력 일반화 힘 30성분 | {mx["gravity_bias_abs_err"]:.2e} Nm |
| 왼발 Jacobian 선속도 / 각속도 | {mx["foot_jac_lin_err"]:.2e} / {mx["foot_jac_ang_err"]:.2e} |

Pinocchio는 CRBA와 RNEA 계열 알고리즘으로 질량행렬과 중력항을 계산한다. 자유 관절의 선속도는 MuJoCo가 세계 좌표, Pinocchio가 몸체 좌표로 표현하므로 회전행렬로 변환한 뒤 비교했다. 전체 질량은 두 쪽 모두 {cc["total_mass"]:.4f} kg이다. 모든 차이가 부동소수점 반올림 수준이므로 1.0의 FK·Jacobian·COM·꼬리 중력 토크 값은 독립 엔진으로 재현된다. 같은 파라미터를 넣은 비교이므로 파라미터 자체가 실물과 맞는지는 검증하지 못한다. [F1,F2]

닫힌 식 검산도 맞았다. 4절 링크 전달비의 음함수 미분식과 각도 표현의 차이는 기호 계산에서 0이고, 표의 값에서는 {fb["max_abs_err_N_vs_angle_formula"]:.1e} 이하다. 무릎 정적 유지 토크의 닫힌 식은 Pinocchio 전신 계산과 {abs(lim["closed_form_nominal"]-lim["pinocchio_nominal"]):.1e} Nm 이내로 같다. [F1,F3]

파서도 시험했다. Pinocchio {pm["version"]}의 MJCF 파서는 기본 모델 파일 sim/raptor_r02.xml을 그대로 읽었고, MuJoCo와의 질량행렬 최대 차이는 {pm["mass_matrix_max_abs_diff_vs_mujoco"]:.1e}였다. Drake {dk["version"]}도 파일을 읽어 질량과 COM이 같았지만, 관절 stiffness·frictionloss·actuatorfrcrange와 general 구동기, 충돌 필터를 지원하지 않는다는 경고를 내고 구동기 {dk["actuators"]}개로 읽었다. 발가락 스프링과 구동기가 빠지므로 Drake를 이 파일의 동역학 대조에 그대로 쓰면 안 된다. 기본 파일의 질량은 {pc["mujoco"]["mass"]:.3f} kg으로, RunEnv가 real_mass로 바꾼 11.2912 kg과 다르다. [F1,F2,F7]

보관된 LQR 행렬도 다시 계산해 1.0의 값과 맞췄다. 전체 A−BK의 스펙트럼 반경 0.9999999996215, 도달 부분공간 0.9910022511, DARE 상대 잔차 2.40248×10^-11이다. [D4,F1]
''', 'N05')

insert_page_after('독립 엔진 교차검증', '''
# 검증 도구 선정

공개 도구 17종을 공식 저장소·PyPI·문서로 조사하고, 설치 가능한 후보를 이 Mac의 Python 3.14.7에 실제로 설치해 import를 시험했다. Pinocchio와 Drake는 모델 로딩까지 시험했다. 조사 보고서와 출처·근거 기록은 별도 폴더에 있다. 아래는 그 결과로 정한 적용 범위다. [F6]

| 도구 | 판정 | 근거 |
| --- | --- | --- |
| Pinocchio 4.1.0 | 적용 | 모델 대조에 사용. MJCF를 직접 읽고 MuJoCo와 10^-14 수준으로 일치 |
| SymPy 1.14.0 | 적용 | 4절 링크 전달비와 정적 토크의 닫힌 식 유도·검산에 사용 |
| MuJoCo 3.14.0 | 유지 | 기존 기준 엔진. 접촉·텐던 스프링·equality를 그대로 사용 |
| Drake 1.57.0 | 보류 | 설치·로딩은 되나 MJCF의 스프링·구동기·충돌 필터를 무시 |
| Crocoddyl 3.2.1 | 다음 단계 후보 | 접촉 순서가 있는 최적 궤적용. 기립 접근 경로 설계에 맞음 |
| TSID 1.10.0 / placo 0.10.1 | 다음 단계 후보 | 작업공간 역동역학 제어. 정지·감속 제어기 후보 |
| mink 1.3.0 | 필요 시 | MuJoCo equality를 쓰는 폐루프 역기구학 |
| MuJoCo Menagerie의 Cassie | 참고 | connect equality와 충돌 class로 링크·자기충돌을 모델링한 예 |
| BAM 1.0.2 | 실물 이후 | 실제 서보의 기록 궤적이 있어야 식별 가능 |
| dairlib | 참고 | Bazel과 Ubuntu 기준 빌드. 이 Mac에서 빌드 시험 안 함 |

Pinocchio README는 pip 설치가 Linux에서만 된다고 적고 있으나, PyPI에는 macOS arm64용 Python 3.14 휠이 있고 이 Mac에서 설치와 실행을 확인했다. PyPI의 bam, aligator, pydrake라는 이름은 이 도구들과 무관한 다른 패키지다. 설치할 때 이름을 그대로 쓰면 안 된다. [F6]

다음 단계 후보는 설치와 import까지만 확인했고 RAPTOR 모델에 적용하지 않았다. 적용하려면 연동 구속·걸쇠 스프링·수동 발가락을 그 도구의 모델로 옮기는 작업이 먼저 필요하다. Crocoddyl과 TSID는 Pinocchio 위에서 동작하므로 이번에 만든 Pinocchio 모델 구성 코드를 출발점으로 쓸 수 있다. [F6]
''', 'N06')

c = lambda k: GC[k]
fl = lambda k: f'{c(k)["falls"]}/{c(k)["n"]}'
sp = lambda k: '—' if c(k)['speed_min'] is None else f'{c(k)["speed_min"]:.3f}'
ht = lambda k: '—' if c(k)['heat_max'] is None else f'{c(k)["heat_max"]:.3f}'
names = [('base', '기준선 (변경 없음)'), ('mu_lo', '발 마찰 ×0.625'), ('mu_hi', '발 마찰 ×1.5625'), ('selfcol', '자기충돌 켬'), ('knee30', '무릎 목표 0.716–2.245'), ('nogap', '접지 상태 시작'),
         ('armature', '로터 반사 관성'), ('latch_nofb', '걸쇠: 대체 연결 제거'), ('latch_clutch', '걸쇠: 현재 각에서 연결'), ('selfcol+knee30+nogap+armature', '걸쇠 외 변경 모두'), ('selfcol+knee30+nogap+armature+mu_lo', '위 + 발 마찰 ×0.625')]
grows = '\n'.join(f'| {n} | {fl(k+"@4")} | {sp(k+"@4")} | {ht(k+"@4")} | {fl(k+"@7")} | {sp(k+"@7")} | {ht(k+"@7")} |' for k, n in names)
b4, b7, za, arm = c('base@4'), c('base@7'), GR['zero_action'], GR['armature']
insert_page_after('시뮬레이션 착수 전 게이트', f"""
# 게이트 측정 결과와 결론

게이트를 닫으려면 측정이 필요해서, 동결된 기준선 정책 scr_a8_h2로 진단 rollout을 실행했다. 조건마다 모델 한 가지만 바꿨고, 4 m/s와 7 m/s 직진 명령을 60초씩, 새 시작 시드 {GR["seeds"][0]}–{GR["seeds"][-1]}의 10개로 돌렸다. 변형은 근거 폴더의 스크립트에서 실행 시점에 적용했고 저장소의 run_env.py는 바꾸지 않았다. 학습은 하지 않았다. [F9]

| 조건 | 4 m/s 전도 | 최소 속도 | 최대 발열 | 7 m/s 전도 | 최소 속도 | 최대 발열 |
| --- | --- | --- | --- | --- | --- | --- |
{grows}

속도는 마지막 40초 평균의 시드별 최솟값(m/s), 발열은 가정 연속 토크 대비 부하제곱의 마지막 40초 평균 가운데 최댓값이다. 전도한 rollout은 속도·발열 집계에서 뺐다. 기준선의 7 m/s 결과(최소 {b7["speed_min"]:.3f} m/s, 발열 {b7["heat_max"]:.3f})는 증거87의 값(6.166858 m/s, 0.963863)을 새 시드로 재현한다. 10회 무전도의 단측 95% 상한은 {GR["statistics"]["10"]*100:.1f}%이므로 무전도 결과는 강건성의 증명이 아니다. 반대로 10회 중 5회, 10회 중 10회 전도는 결론을 내리기에 충분하다. [F9,E2]

## 걸쇠 스프링 (G5, G6)

기준선 주행에서 걸쇠 연결은 초당 {b4["engagements_per_s"]:.2f}–{b7["engagements_per_s"]:.2f}회 일어났고, 20회의 연결 {b4["engagements"]+b7["engagements"]:,}건 가운데 변형이 0.01 rad 이하인 것은 {b4["engagements_zero_deflection"]+b7["engagements_zero_deflection"]}건이었다. 코드의 의도는 유각에서 무릎이 정지각 0.956 rad까지 펴졌을 때 변형 없이 거는 것이다. 그런 연결은 일어나지 않았고, 실제로 동작한 것은 발에 하중이 걸리면 무조건 연결하는 대체 분기였고, 이때 변형은 평균 {b4["engage_defl_mean"]:.3f}–{b7["engage_defl_mean"]:.3f} rad, 최대 {b7["engage_defl_max"]:.3f} rad였다. 강성을 0에서 k로 바꾸는 순간 0.5 k (q − q0)²의 에너지가 새로 생긴다. [F9,E1]

| 항목 | 4 m/s | 7 m/s |
| --- | --- | --- |
| 연결 때 생긴 에너지 | {b4["inject_W"]:.1f} W | {b7["inject_W"]:.1f} W |
| 해제 때 버려진 에너지 | {b4["release_W"]:.1f} W | {b7["release_W"]:.1f} W |
| 모터 양의 기계동력 | {b4["motor_pos_W"]:.1f} W | {b7["motor_pos_W"]:.1f} W |
| 생긴 에너지 / 모터 동력 | {b4["inject_over_motor"]*100:.1f}% | {b7["inject_over_motor"]*100:.1f}% |

에너지가 보존되는 두 변형으로 같은 정책을 돌리면 모두 전도했다. 대체 연결을 없애고 리셋 때 해제 상태로 시작하면 4 m/s에서 {c("latch_nofb@4")["fall_times"][0]}–{c("latch_nofb@4")["fall_times"][-1]}초, 7 m/s에서 {c("latch_nofb@7")["fall_times"][0]}–{c("latch_nofb@7")["fall_times"][-1]}초에 10회 모두 전도했고 스프링은 한 번도 걸리지 않았다. 접지 때 현재 무릎각에서 변형 없이 거는 변형도 {c("latch_clutch@4")["fall_times"][0]}–{c("latch_clutch@4")["fall_times"][-1]}초와 {c("latch_clutch@7")["fall_times"][0]}–{c("latch_clutch@7")["fall_times"][-1]}초에 10회 모두 전도했다. 두 변형의 공통점은 접지 순간의 큰 스프링 토크가 없다는 것이므로, 기준선 정책의 주행은 그 토크에 의존한다고 볼 수 있다. [F9]

정지 유지에서도 같은 결론이 나온다. 명목 자세에서 행동 0으로 3초를 유지시키면, 리셋 때 걸쇠가 연결된 현재 모델은 양쪽 무릎 모터의 부하제곱이 {za["base"]["knee_heat"][0]:.2f}–{za["base"]["knee_heat"][1]:.2f}로 피크에 포화한다(피크는 연속의 2.5배이므로 6.25가 포화값이다). 해제 상태로 시작하면 {za["latch_nofb"]["knee_heat"][0]:.2f}–{za["latch_nofb"]["knee_heat"][1]:.2f}로, 정적 계산의 (16.54/19.16)² = 0.745와 맞는다. 어느 경우든 명목 자세는 0.6–1.1초에 넘어진다. 수동으로 안정한 자세가 아니기 때문이다. [F9,F1]

결론은 다음과 같다. 모델 사양은 대체 연결을 없앤 수동 걸쇠로 정한다. 유각에서 q ≤ q0일 때만 걸고, 발을 뗄 때 풀고, 리셋은 해제 상태로 한다. 시험한 두 가지 에너지 보존 방식 가운데 추가 기구가 필요 없는 쪽이다. 현재 각에서 거는 방식은 임의 각도에서 잠기는 클러치가 따로 필요하므로 채택하지 않는다. 이 사양에서 기존 정책은 쓸 수 없고 다시 학습해야 한다. 정지각 0.956 rad와 강성 168.97 Nm/rad가 적절한지는 정적 근거로 정할 수 없다. 재개할 때 스프링 없음, q0 = 0.956, q0 = 기준선의 평균 접지 무릎각 근처({0.956+b4["engage_defl_mean"]:.2f}–{0.956+b7["engage_defl_mean"]:.2f} rad)의 세 가지를 같은 조건으로 학습해 비교해야 한다. [F9]

## 접촉 마찰 (G2)

발 geom의 마찰에 배율을 곱하면 높은 우선순위를 유지한 채 접촉 마찰이 바뀐다. 배율 0.625와 1.5625는 패드 기준 접촉 마찰 0.5와 1.25에 해당한다. 두 경우 모두 양 속도에서 전도가 없었고, 7 m/s의 최소 속도는 {sp("mu_lo@7")}과 {sp("mu_hi@7")} m/s, 최대 발열은 {ht("mu_lo@7")}과 {ht("mu_hi@7")}였다. 결론: 무작위화를 발 geom 마찰의 배율로 바꾼다. 발끝 1.2와 패드 0.8의 비가 유지되고, 직진 주행에서는 기존 정책에 미치는 영향이 작다. 회전과 급정지에서의 영향은 측정하지 않았다. [F9]

## 자기충돌 (G3)

궤적을 바꾸지 않는 탐지 전용 계측에서 기준선은 좌우 다리가 서로 겹쳤다. 4 m/s에서는 제어 주기의 {b4["overlap_step_frac"]*100:.2f}%(10회 중 {b4["rollouts_with_overlap"]}회, 최대 {b4["overlap_max_penetration_mm"]:.1f} mm), 7 m/s에서는 {b7["overlap_step_frac"]*100:.2f}%(10회 중 {b7["rollouts_with_overlap"]}회, 최대 {b7["overlap_max_penetration_mm"]:.1f} mm)였다. 고관절 roll의 안쪽 최대각은 {b7["hip_roll_in_max"]:.3f} rad였고, 정적 검사에서 양쪽 roll이 안쪽으로 0.2 rad이면 좌우 부척골과 발가락이 겹친다. 좌우 다리, 꼬리–다리, 몸통–다리 원위부 접촉을 켜면 4 m/s는 전도 0회로 결과가 같았지만 7 m/s는 10회 중 {c("selfcol@7")["falls"]}회 전도했다. [F9]

결론: 좌우 다리, 꼬리–다리, 몸통–다리 원위부 쌍을 켠다. 7 m/s의 기존 통과 기록은 다리가 서로 통과한 궤적을 포함하므로 물리적 근거에서 뺀다. 상자 형상 자체가 처음부터 겹치는 쌍이 두 종류 있었다. 대퇴 상단과 몸통 옆면(명목 자세에서 0.13–0.26 mm, roll 0.5 rad에서 16.8 mm), 부척골 끝과 발가락 뿌리(3.5–8.3 mm)다. 이 쌍은 형상을 고친 뒤에 켜야 한다. 같은 다리 안의 접힘 간섭은 무릎 2.3 rad까지 정적 검사에서 나타나지 않았다. 탐지기는 일부러 겹치게 만든 자세로 동작을 확인했다. 처음에는 몸체 단위 충돌 마스크를 갱신하지 않아 아무것도 잡지 못했고, 그것을 고친 뒤의 결과만 사용했다. [F9]

## 무릎 목표 범위 (G4)

기준선이 실제로 쓴 무릎각은 4 m/s에서 {b4["knee_min"]:.3f}–{b4["knee_max"]:.3f} rad, 7 m/s에서 {b7["knee_min"]:.3f}–{b7["knee_max"]:.3f} rad였고, 양쪽 30° 구간 0.716–2.245 rad를 벗어난 시간은 0이었다. 목표 범위를 이 구간으로 자르면 전도 0회, 7 m/s 최소 속도 {sp("knee30@7")} m/s, 발열 {ht("knee30@7")}로 기준선과 같다. 결론: 정책 무릎 목표 범위를 0.716–2.245 rad로 한다. 측정한 직진 주행에서는 잃는 것이 없다. 유각 접힘 목표 65°는 무릎 2.007 rad에 해당해 구간 안이다. 회전, 일어나기, 7 m/s 초과에서 더 넓은 범위가 필요한지는 측정하지 않았다. [F9,F1]

## 시작 높이 (G7)

리셋의 시작 간극은 시드에 따라 {b4["start_gap_mm"][0]:.1f}–{b7["start_gap_mm"][1]:.1f} mm였다. 발가락이 닿은 상태로 시작해도 전도 0회, 7 m/s 최소 속도 {sp("nogap@7")} m/s, 발열 {ht("nogap@7")}로 차이가 작다. 결론: 접지 상태에서 시작한다. 결과에 미치는 영향은 작지만 첫 주기의 하중 판정과 걸쇠 상태가 명확해진다. [F9]

## 로터 관성 (G8)

AK10-9 V3.0의 공개 사양은 로터 관성 1002 g·cm², 내장 감속비 9:1이다. 모듈 출력단 반사 관성은 81 × 1.002×10^-4 = {arm["module_output"]:.6f} kg·m²이고, 관절 기준으로 hip pitch {arm["hip_pitch"]:.5f}, 무릎 {arm["knee_at_N1p4"]:.5f}(|N| = 1.4 대표값), 꼬리 {arm["tail"]:.5f} kg·m²다. hip roll의 A급 모듈은 정해진 부품이 없어 같은 모듈 값에 1.7²을 곱한 {arm["hip_roll_assumed"]:.5f} kg·m²를 가정했다. 이 값을 넣으면 전도는 없지만 모터 양의 기계동력이 4 m/s에서 {(c("armature@4")["motor_pos_W"]/b4["motor_pos_W"]-1)*100:.1f}%, 7 m/s에서 {(c("armature@7")["motor_pos_W"]/b7["motor_pos_W"]-1)*100:.1f}% 늘고, 7 m/s의 최소 속도는 {sp("armature@7")} m/s로 줄며 발열은 {ht("armature@7")}로 한계에 닿는다. 결론: 반사 관성을 잠정 파라미터로 모델에 넣는다. 에너지와 7 m/s 열 여유에 영향이 크다. 무릎은 전달비가 자세에 따라 변하므로 상수 armature는 근사다. [F9,E9]

## 판정 통계 (G9)

60초 rollout 하나는 이 Mac에서 코어당 약 {GR["timing"]["wall_s_per_60s_rollout"]:.1f}초가 걸린다. 결론: 선별은 조건당 59회(무전도일 때 상한 {GR["statistics"]["59"]*100:.2f}%), 채택 판정은 조건당 299회(상한 {GR["statistics"]["299"]*100:.2f}%)로 하고, 시드 목록은 실행 전에 고정해 개발에 쓰지 않은 것으로 둔다. 8코어에서 299회는 약 {299*GR["timing"]["wall_s_per_60s_rollout"]/8/60:.0f}분이다. 이번 측정의 10회는 진단용이며 채택 판정 기준에 못 미친다. [F9]

## 종합

걸쇠를 뺀 변경을 모두 함께 적용하면 4 m/s는 전도 0회, 발열 {ht("selfcol+knee30+nogap+armature@4")}로 유지된다. 7 m/s는 10회 중 {c("selfcol+knee30+nogap+armature@7")["falls"]}회 전도하고 발열이 {ht("selfcol+knee30+nogap+armature@7")}이며, 발 마찰까지 낮추면 발열이 {ht("selfcol+knee30+nogap+armature+mu_lo@7")}로 한계를 넘는다. 걸쇠를 에너지가 보존되게 고치면 두 속도 모두 유지되지 않는다. [F9]

따라서 순서는 이렇게 정한다. 여섯 가지 수정(걸쇠, 마찰, 자기충돌, 무릎 범위, 시작 높이, 로터 관성)을 기본값을 바꾸지 않는 선택 옵션으로 구현해 새 모델 버전으로 묶고, 그 모델에서 스프링 세 가지를 같은 조건으로 다시 학습한 뒤, 고정 시드 299회로 판정한다. 각 수정은 이번에 하나씩 따로 측정했으므로 묶어서 적용해도 원인을 구분할 수 있다. 1.2에서는 이 결론을 코드에 적용하지 않았다. 1.3에서 선택 옵션으로 구현했고, 스프링 비교에 대한 결론도 다음 절에서 바꿨다. 학습 재개는 사용자의 재개 지시가 있어야 한다.
""", 'N07')


o = lambda k: R2['options'][k]; sg = lambda k: R2['springs'][k]; dr = lambda k: R2['drop_rig'][k]; rp = R2['spring_replay']; cap = {r['knee']: r for r in R2['static_knee_capacity']}
nr = R2['rollouts']
insert_page_after('게이트 측정 결과와 결론', f"""
# 모델 개정 2: 구현과 무학습 검증

앞 절의 여섯 가지 결론을 sim/rl/run_env.py에 선택 옵션으로 구현했다. model_rev=2가 여섯 가지를 함께 켜고, 항목마다 따로 켜고 끌 수도 있다. train_run.py에는 --model-rev 인자를 더했다. 기본값 model_rev=1은 이전 동작 그대로다. 수정한 코드로 기준선 rollout {nr["regression"]}개(4·7 m/s, 시드 95001·95007, 60초)를 기본값으로 다시 돌렸을 때 속도, 발열, 주입 에너지, 무릎각 범위, 연결 횟수가 수정 전 기록과 모든 자리에서 같았다. 학습은 하지 않았다. 이 절의 rollout은 동결 정책 {nr["options"]+nr["springs"]}회, 무행동 유지 {nr["stand"]}회, 낙하 시험 {nr["drop_rig"]}회다. [F10]

| 옵션 | 개정 1 (기본) | 개정 2 |
| --- | --- | --- |
| latch_fallback | 접지하면 무조건 연결 | 유각에서 정지각을 신전 방향으로 지날 때만 연결, 리셋은 해제 |
| friction_dr | 바닥 마찰만 무작위 | 발 패드와 바닥에 같은 배율 0.625–1.5625 |
| self_collision | 없음 | 좌우 다리, 꼬리–다리, 몸통–정강이 이하 |
| knee_band | 관절 범위 0.2–2.6 rad | 무릎 목표 0.7159–2.2454 rad |
| start_contact | 패드가 12–13 mm 떠서 시작 | 0.1 mm에서 시작 |
| armature | 0 | 구동 관절의 로터 반사 관성 |

로터 반사 관성은 hip pitch 0.01826, 무릎 0.01495, hip roll 0.02346, 꼬리 0.00812 kg·m²다. 무릎 값은 4절 링크 전달비 제곱의 유효 구간 평균을 쓴 근사이고, hip roll 값은 가정이다. 자기충돌에서 기하 형상 없는 링크로 이어진 이웃(대퇴–몸통, 부척골–발가락, 같은 발의 발가락끼리)은 상자 형상이 처음부터 겹치므로 제외했다. 대퇴–몸통 간섭은 hip roll 범위 ±0.5 rad가 대신하며 실제 간섭 여부는 상세 형상이 정해져야 확인할 수 있다. [F10]

## 옵션별 재측정

구현한 옵션을 하나씩, 그리고 함께 켜서 동결 정책으로 다시 측정했다(조건당 10회, 60초).

| 조건 | 4 m/s 전도 | 7 m/s 전도 | 7 m/s 최소 속도 | 7 m/s 발열 | 주입 에너지 4 / 7 m/s |
| --- | --- | --- | --- | --- | --- |
| 개정 1 | {o("rev1@4")["falls"]}/10 | {o("rev1@7")["falls"]}/10 | {o("rev1@7")["speed_min"]:.3f} | {o("rev1@7")["heat_max"]:.3f} | {o("rev1@4")["inject_W"]:.1f} / {o("rev1@7")["inject_W"]:.1f} W |
| 걸쇠만 | {o("latch@4")["falls"]}/10 | {o("latch@7")["falls"]}/10 | — | — | {o("latch@4")["inject_W"]:.1f} / {o("latch@7")["inject_W"]:.1f} W |
| 자기충돌만 | {o("selfcol@4")["falls"]}/10 | {o("selfcol@7")["falls"]}/10 | {o("selfcol@7")["speed_min"]:.3f} | {o("selfcol@7")["heat_max"]:.3f} | {o("selfcol@4")["inject_W"]:.1f} / {o("selfcol@7")["inject_W"]:.1f} W |
| 마찰만 | {o("friction@4")["falls"]}/10 | {o("friction@7")["falls"]}/10 | {o("friction@7")["speed_min"]:.3f} | {o("friction@7")["heat_max"]:.3f} | {o("friction@4")["inject_W"]:.1f} / {o("friction@7")["inject_W"]:.1f} W |
| 무릎 범위만 | {o("knee@4")["falls"]}/10 | {o("knee@7")["falls"]}/10 | {o("knee@7")["speed_min"]:.3f} | {o("knee@7")["heat_max"]:.3f} | {o("knee@4")["inject_W"]:.1f} / {o("knee@7")["inject_W"]:.1f} W |
| 접지 시작만 | {o("start@4")["falls"]}/10 | {o("start@7")["falls"]}/10 | {o("start@7")["speed_min"]:.3f} | {o("start@7")["heat_max"]:.3f} | {o("start@4")["inject_W"]:.1f} / {o("start@7")["inject_W"]:.1f} W |
| 로터 관성만 | {o("armature@4")["falls"]}/10 | {o("armature@7")["falls"]}/10 | {o("armature@7")["speed_min"]:.3f} | {o("armature@7")["heat_max"]:.3f} | {o("armature@4")["inject_W"]:.1f} / {o("armature@7")["inject_W"]:.1f} W |
| 개정 2에서 대체 연결만 유지 | {o("rev2_keep_fallback@4")["falls"]}/10 | {o("rev2_keep_fallback@7")["falls"]}/10 | {o("rev2_keep_fallback@7")["speed_min"]:.3f} | {o("rev2_keep_fallback@7")["heat_max"]:.3f} | {o("rev2_keep_fallback@4")["inject_W"]:.1f} / {o("rev2_keep_fallback@7")["inject_W"]:.1f} W |
| 개정 2 전체 | {o("rev2_all@4")["falls"]}/10 | {o("rev2_all@7")["falls"]}/10 | — | — | {o("rev2_all@4")["inject_W"]:.1f} / {o("rev2_all@7")["inject_W"]:.1f} W |

실행 시점에 바꿔 넣었던 앞 절의 결과와 같다. 걸쇠를 고치면 10회 모두 {o("latch@7")["fall_t_min"]:.1f}–{o("latch@4")["fall_t_max"]:.1f}초에 전도하고, 자기충돌만 켜면 7 m/s에서 10회 중 5회 전도하며, 나머지 넷은 전도가 없다. 개정 2 전체에서는 주입 에너지가 0이다. 이번에는 꼬리–대퇴 쌍도 켰지만 결과는 달라지지 않았다. [F10,F9]

## 걸쇠 규칙의 두 가지 보완

구현하면서 앞 절의 규칙에서 두 가지를 고쳤다. 첫째, 걸쇠 스프링을 굴곡 방향으로만 힘을 내는 단방향(uni+)으로 쓴다. 양방향 스프링은 제어 주기 20 ms 사이에 무릎이 정지각을 조금 지난 뒤 연결되면 그만큼의 에너지가 생긴다. 단방향이면 정지각 아래에서 힘이 0이므로 연결 때 생기는 에너지가 정확히 0이다. 둘째, 유각에서 무릎이 정지각 위로 굽었다가 다시 펴지며 정지각을 지날 때만 건다. 무릎이 정지각 아래에 있기만 하면 거는 규칙은 정지각이 발을 뗄 때의 무릎각보다 크면 발을 떼자마자 다시 걸려 유각 굴곡을 막는다. [F10]

## 기준선 보행은 수동 스프링과 맞지 않는다

기준선 정책의 무릎이 낸 토크(모터와 스프링의 합)를 1 ms마다 기록하고, 같은 운동을 물리적으로 가능한 걸쇠 스프링으로 다시 낼 때 모터가 맡을 토크를 계산했다. 운동은 고정했으므로 이 계산은 그 보행에 대한 것이다. [F10]

| 항목 | 2 m/s 명령 | 4 m/s | 7 m/s |
| --- | --- | --- | --- |
| 접지 순간 무릎각 평균 | {rp["2"]["gait"]["knee_touchdown_mean"]:.3f} rad | {rp["4"]["gait"]["knee_touchdown_mean"]:.3f} rad | {rp["7"]["gait"]["knee_touchdown_mean"]:.3f} rad |
| 입각 중 무릎각 평균 | {rp["2"]["gait"]["stance_knee_mean"]:.3f} rad | {rp["4"]["gait"]["stance_knee_mean"]:.3f} rad | {rp["7"]["gait"]["stance_knee_mean"]:.3f} rad |
| 무릎 필요 토크 최대 | {rp["2"]["gait"]["req_peak"]:.1f} Nm | {rp["4"]["gait"]["req_peak"]:.1f} Nm | {rp["7"]["gait"]["req_peak"]:.1f} Nm |
| 입각 중 필요 토크 RMS | {rp["2"]["gait"]["req_stance_rms"]:.1f} Nm | {rp["4"]["gait"]["req_stance_rms"]:.1f} Nm | {rp["7"]["gait"]["req_stance_rms"]:.1f} Nm |
| 기록된 무릎 모터 부하제곱 | {rp["2"]["gait"]["actual_motor_heat"]:.2f} | {rp["4"]["gait"]["actual_motor_heat"]:.2f} | {rp["7"]["gait"]["actual_motor_heat"]:.2f} |
| 스프링 없이 같은 운동: 부하제곱 | {rp["2"]["candidates"]["none"]["heat"]:.2f} | {rp["4"]["candidates"]["none"]["heat"]:.2f} | {rp["7"]["candidates"]["none"]["heat"]:.2f} |
| 스프링 없이: 구동 한계를 넘는 시간 | {rp["2"]["candidates"]["none"]["over"]*100:.1f}% | {rp["4"]["candidates"]["none"]["over"]*100:.1f}% | {rp["7"]["candidates"]["none"]["over"]*100:.1f}% |
| q0 0.956 수동 걸쇠가 걸린 시간 | {rp["2"]["candidates"]["k170_q0.956"]["latched"]*100:.1f}% | {rp["4"]["candidates"]["k170_q0.956"]["latched"]*100:.1f}% | {rp["7"]["candidates"]["k170_q0.956"]["latched"]*100:.1f}% |
| q0 1.3 수동 걸쇠가 걸린 시간 | {rp["2"]["candidates"]["k170_q1.3"]["latched"]*100:.1f}% | {rp["4"]["candidates"]["k170_q1.3"]["latched"]*100:.1f}% | {rp["7"]["candidates"]["k170_q1.3"]["latched"]*100:.1f}% |

이 보행은 무릎이 굽은 채 접지한 뒤 입각 동안 평균적으로 더 펴진다. 그동안 무릎에는 모터 한계를 넘는 토크가 필요했고, 그 대부분을 접지 순간에 변형된 채 연결된 스프링이 냈다. 수동 스프링은 입각에서 먼저 눌려야 힘을 내므로 이 운동에서는 걸리지 않거나, 걸려도 힘을 내지 않는다. 강성 0–600 Nm/rad, 정지각 0.80–1.80 rad의 격자 전체에서 4 m/s와 7 m/s의 모터 부하제곱은 스프링이 없을 때보다 낮아지지 않았다. 동결 정책을 개정 2에서 스프링 격자(강성 85·170·255, 정지각 0.956–1.5 rad)로 직접 돌려도 4·7 m/s는 조건마다 10회 모두 전도했고, 정지각 1.2 rad 이하에서는 결과가 스프링 없는 경우와 같았다. 2 m/s 명령에서는 스프링 없이 10회 중 {10-sg("none@2")["falls"]}회가 60초를 버텼지만 모터 부하제곱의 최댓값이 {sg("none@2")["heat_max"]:.2f}로 연속 정격을 넘었다. 따라서 기준선 정책은 미세조정의 출발점으로도 쓸 근거가 없고, 재학습한 정책은 다른 형태의 보행을 찾아야 한다. [F10]

## 스프링 세 가지의 무학습 비교

앞 절은 스프링 없음, q0 0.956, q0 약 1.3의 세 가지를 학습으로 비교하자고 했다. 학습 없이 정할 수 있는 부분을 두 가지 방법으로 확인했다. 첫째는 정적 여유다. 몸통이 수평이고 발가락이 평평하면 hip 각이 고정되어 정강이의 기울기가 무릎각만으로 정해진다. 한 다리의 수직 지면반력을 스프링 없이 무릎 구동만으로 버틸 수 있는 크기는 다음과 같다. [F10,F1]

| 무릎각 | 전달비 | 체중 1배당 무릎 토크 | 연속 정격으로 버티는 힘 | 피크로 버티는 힘 | q0 0.956 스프링 토크 |
| --- | --- | --- | --- | --- | --- |
""" + ''.join(f'| {q:.3f} rad | {cap[q]["N"]:.3f} | {cap[q]["torque_per_BW"]:.1f} Nm | 체중의 {cap[q]["F_cont_BW"]:.2f}배 | 체중의 {cap[q]["F_peak_BW"]:.2f}배 | {cap[q]["spring170_from_0p956"]:.1f} Nm |\n' for q in (.956, 1.1, 1.2, 1.3, 1.5)) + f"""
무릎이 0.956 rad일 때는 한 다리로 체중의 {cap[.956]["F_cont_BW"]:.2f}배를 연속 정격 안에서 버티지만 1.3 rad에서는 {cap[1.3]["F_cont_BW"]:.2f}배로, 한 다리로 체중도 연속으로 버티지 못한다. 정지각이 1.3 rad인 스프링은 그 각에서 힘이 0이고 더 굽어야 힘을 내는데, 그 구간은 부하가 더 크다. 정지각 0.956 rad의 스프링은 1.3 rad에서 {cap[1.3]["spring170_from_0p956"]:.1f} Nm, 체중의 {cap[1.3]["spring170_from_0p956"]/cap[1.3]["torque_per_BW"]:.2f}배를 받친다. [F10]

둘째는 낙하 시험이다. 몸통의 자세와 좌우 위치를 구속 장치로 잡고(높이와 전후는 자유), 양 다리를 같은 자세로 두고 패드를 지면에서 띄워 놓았다가 떨어뜨렸다. 무릎 서보는 접지 무릎각을 목표로 유지하고, 스프링은 그 각에서 변형 없이 걸린 상태로 시작한다. 정책은 쓰지 않았다. 아래는 첫 입각 동안 무릎 모터 부하제곱의 시간 적분(초)과 모터가 피크에 닿은 시간 비율이다.

| 접지 무릎각 | 스프링 | 10 cm | 20 cm | 20 cm 포화 비율 |
| --- | --- | --- | --- | --- |
""" + ''.join(f'| {q} rad | {"없음" if lab == "none" else lab[1:]+" Nm/rad"} | {dr(f"{lab}@{q}@h0.1")["heat_integral_s"]:.3f} | {dr(f"{lab}@{q}@h0.2")["heat_integral_s"]:.3f} | {dr(f"{lab}@{q}@h0.2")["motor_sat_frac"]*100:.0f}% |\n' for q in (.956, 1.3) for lab in ('none', 'k170', 'k340')) + f"""
같은 낙하 높이에서 접지 무릎각 0.956 rad는 1.3 rad보다 무릎 부하가 훨씬 작고, 1.3 rad에 강성 340 Nm/rad 스프링을 달아도 0.956 rad에 스프링이 없는 경우보다 나쁘다. 같은 자세에서는 스프링이 모터 부하를 줄인다. 0.956 rad, 10 cm에서 강성 170 Nm/rad는 적분값을 {(1-dr("k170@0.956@h0.1")["heat_integral_s"]/dr("none@0.956@h0.1")["heat_integral_s"])*100:.0f}% 줄였고, 20 cm에서는 {(1-dr("k170@0.956@h0.2")["heat_integral_s"]/dr("none@0.956@h0.2")["heat_integral_s"])*100:.0f}% 줄였다. 시험한 강성 85–340 Nm/rad에서 강성이 클수록 더 줄었다. 다만 서보가 목표각을 굳게 잡고 있어 무릎이 0.03–0.2 rad밖에 굽지 않았고, 강성 170 Nm/rad 스프링이 저장한 에너지는 낙하 에너지의 {dr("k170@0.956@h0.1")["spring_E_peak_J"]/dr("k170@0.956@h0.1")["drop_energy_J"]*100:.0f}%(0.956 rad, 10 cm)에서 {dr("k170@1.3@h0.2")["spring_E_peak_J"]/dr("k170@1.3@h0.2")["drop_energy_J"]*100:.0f}%(1.3 rad, 20 cm) 사이였다. 되튀는 높이도 낙하 높이의 {max(v["restitution"] for v in R2["drop_rig"].values())*100:.0f}% 이하였다. 스프링이 에너지를 되돌려 주려면 제어가 입각에서 무릎이 굽도록 두어야 하고, 그것은 정책이 정할 일이다. [F10]

![낙하 시험: 접지 무릎각 0.956 rad, 강성 170 Nm/rad, 10 cm (80 ms 간격)](assets/drop-rig-k170-q0956.png)

## 결론

여섯 가지 수정은 개정 2로 구현을 마쳤고 기본값의 재현성을 확인했다. 스프링에 대해서는 세 가지를 따로 학습할 필요가 없다. 정지각 약 1.3 rad는 정적 여유와 낙하 시험에서 모두 불리하므로 후보에서 뺀다. 정지각 0.956 rad의 수동 걸쇠는 무릎을 그 각까지 펴지 않으면 걸리지 않으므로, 그 모델은 스프링 없는 경우를 포함한다. 동결 정책 측정에서 두 경우의 결과가 실제로 같았다. 따라서 재개할 때는 개정 2와 무릎 스프링 [170, 0.956, uni+, latch] 한 가지로 학습하고, 학습된 보행에서 입각 중 걸쇠가 걸린 비율과 무릎 부하제곱을 보고 스프링의 필요를 판단한다. 걸림 비율이 0에 가까우면 스프링과 걸쇠는 빼도 되는 부품이다. 강성 170 Nm/rad는 잠정값으로 둔다. [F10]

확인하지 못한 것은 다음과 같다. 개정 2에서 달릴 수 있는 정책이 있는지, 그 속도와 발열이 어떤지는 학습 없이는 알 수 없다. 걸쇠 기구의 질량과 손실은 모델에 없다. 낙하 시험은 양발 수직 낙하이고 서보가 목표각을 유지하는 한 가지 제어만 썼다. 명목 자세는 개정 2에서도 무행동으로 {R2["stand"]["rev2_k170_q0.956@0"]["fall_t_min"]:.1f}–{R2["stand"]["rev2_k170_q0.956@0"]["fall_t_max"]:.1f}초에 앞으로 넘어지며, 이때 무릎 모터 부하제곱은 {R2["stand"]["rev2_k170_q0.956@0"]["heat_knee_median"]:.2f}이다(개정 1은 {R2["stand"]["rev1@0"]["heat_knee_median"]:.2f}). 서 있기에는 균형 제어가 필요하다. [F10]
""", 'N08')



ta = lambda c: RT['accept'][f'accept@{c}']; td_ = lambda c: RT['dr'][f'dr@{c}']; tb = lambda c: RT['dr_baseline_rev1'][f'dr@{c}']; tf = lambda c: RT['ft_accept59'][f'accept@{c}']
tg = RT['gait']; tr = RT['replay']; ftg = RT['ft_gait']
insert_page_after('모델 개정 2: 구현과 무학습 검증', f"""
# 모델 개정 2: 학습과 판정

## 실행한 것

사용자 지시(2026-10-12)로 개정 2에서 학습 1회를 실행했다. 인자는 기준선 scr_a8_h2와 같고 model_rev 2와 무릎 스프링 [170, 0.956, uni+, latch]만 다르다. 처음부터 학습했고 실행 이름은 rev2_a8_h2, 길이는 {RT["train"]["steps"]:,} 스텝이다. 판정 시드 96001–96299는 결과가 나오기 전에 고정했고 이전 진단(95001–95010)에 쓰지 않은 것이다. 판정 rollout은 {RT["rollouts"]["accept"]}회, 무작위화 조건은 {RT["rollouts"]["dr"]+RT["rollouts"]["dr_baseline"]}회다. [F13]

## 학습 경과

결정적 평가를 처음 통과한 시점은 {RT["train"]["first_pass_step"]:,} 스텝으로 기준선({RT["baseline_train"]["first_pass_step"]:,})과 같다. 커리큘럼의 명령 속도 상한은 {RT["train"]["final_cmd_max"]:.2f} m/s에서 끝났고 기준선은 {RT["baseline_train"]["final_cmd_max"]:.2f} m/s였다. 마지막 네 번의 평가 속도는 {min(RT["train"]["last_eval_speeds"]):.2f}–{max(RT["train"]["last_eval_speeds"]):.2f} m/s로, 상한의 80%인 통과 기준에 못 미쳤다. [F13]

## 고정 시드 판정

무작위화와 외란 없이 60초씩 달린 결과다. 속도는 마지막 40초 평균, 무릎 부하제곱은 연속 정격을 1로 둔 값이다.

| 명령 | 전도 | 속도 평균 (최소) | 무릎 부하제곱 평균 (최대) | 걸쇠 연결 | 자기충돌 |
|---|---:|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {ta(c)["falls"]}/{ta(c)["n"]} | {ta(c)["speed"]:.2f} ({ta(c)["speed_min"]:.2f}) m/s | {ta(c)["heat_knee"]:.2f} ({ta(c)["heat_knee_max"]:.2f}) | {ta(c)["engagements"]}회 | {ta(c)["self_contact_step_frac"]*100:.1f}% |\n' for c in ('2', '4', '5.5', '7')) + f"""
전도는 네 조건 모두 0/299다. 299회에서 전도가 없으면 전도 확률의 단측 95% 상한은 {RT["binomial"]["upper95_one_sided"]*100:.3f}%다(1 − 0.05^(1/299)). 걸쇠가 만든 에너지는 0이다. 그러나 가장 뜨거운 관절은 모든 조건에서 무릎이고 부하제곱이 {ta("2")["heat_knee"]:.2f}–{ta("7")["heat_knee"]:.2f}로 연속 정격을 넘는다. 걸쇠는 한 번도 걸리지 않았다. 따라서 이 정책은 전도 기준은 통과하지만 열 기준에서 채택할 수 없다. [F13]

## 무작위화 조건

학습 때의 무작위화(질량, 발 패드 마찰, 시작 상태, 요 충격, 밀기)를 켜고 수준 0에서 59회씩 60초 달렸다. 오른쪽 열은 옛 기준선을 옛 모델(개정 1)에서 같은 방식으로 돌린 것이다. 옛 모델은 걸쇠가 에너지를 만들어 주므로 두 열은 같은 물리 조건의 비교가 아니다.

| 명령 | 개정 2 정책 전도 | 옛 기준선 전도 (개정 1) |
|---|---:|---:|
""" + ''.join(f'| {c} m/s | {td_(c)["falls"]}/{td_(c)["n"]} | {tb(c)["falls"]}/{tb(c)["n"]} |\n' for c in ('2', '4', '7')) + f"""
학습 에피소드는 15초이고 이 시험은 60초다. 개정 2 정책은 4 m/s 이상에서 60초 안에 절반 넘게 넘어진다. 외란에 대한 여유는 확보되지 않았다. [F13]

## 걸쇠가 걸리지 않은 이유

정책의 무릎각을 보행 단계별로 기록했다(시드 96001, 15초, 왼쪽/오른쪽).

| 명령 | 접지 무릎각 | 이지 무릎각 | 입각 중 최대 | 유각 중 최대 |
|---|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {tg[c]["touchdown_mean"][0]:.2f} / {tg[c]["touchdown_mean"][1]:.2f} rad | {tg[c]["liftoff_mean"][0]:.2f} / {tg[c]["liftoff_mean"][1]:.2f} rad | {tg[c]["stance_max"][0]:.2f} / {tg[c]["stance_max"][1]:.2f} rad | {tg[c]["swing_max"][0]:.2f} / {tg[c]["swing_max"][1]:.2f} rad |\n' for c in ('2', '4', '7')) + f"""
이 보행도 기준선처럼 무릎을 굽힌 채 접지해 입각 동안 편다. 걸쇠는 유각에서 무릎이 정지각 0.956 rad 위로 굽었다가 다시 그 각까지 펴질 때 걸리는데, 유각 말기의 무릎은 접지각({tg["2"]["touchdown_mean"][1]:.2f}–{tg["4"]["touchdown_mean"][1]:.2f} rad)까지만 펴진다. 7 m/s에서 무릎이 0.956 rad 아래로 내려가는 것은 발을 떼는 순간과 그 직후이며 걸쇠 조건이 아니다. 굽힌 접지는 옛 걸쇠가 만든 것이 아니라 같은 보상과 명목 자세(무릎 1.5 rad)에서 다시 나온 것이다. [F13]

## 스프링이 걸렸다면

같은 운동에서 무릎이 낸 토크를 1 ms마다 기록하고, 발이 지면에 닿아 있는 동안 스프링(170 Nm/rad, 정지각 0.956 rad, 단방향)이 걸려 있다고 가정해 모터가 맡을 토크를 계산했다. 운동은 고정했고, 접지 전에 스프링을 변형시키는 일은 넣지 않았으므로 이 값은 효과의 상한이다.

| 명령 | 필요 토크 최대 | 모터 부하제곱, 스프링 없음 | 모터 부하제곱, 입각 중 스프링 |
|---|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {tr[c]["req_peak"]:.1f} Nm | {tr[c]["none"][0]:.2f} | {tr[c]["stance_k170_q0956"][0]:.2f} |\n' for c in ('2', '4', '7')) + f"""
스프링이 입각에서 작용하면 무릎 부하제곱은 연속 정격 아래로 내려간다. 이 표의 스프링 없는 값은 1 ms 토크에서 직접 계산한 것이고 판정 표의 발열 지표와 거의 같다. 접지각 {tg["4"]["touchdown_mean"][0]:.2f} rad에서 정지각 0.956 rad의 스프링에 저장되는 에너지는 0.5 × 170 × ({tg["4"]["touchdown_mean"][0]:.3f} − 0.956)² = {0.5*170*(tg["4"]["touchdown_mean"][0]-0.956)**2:.1f} J이고, 실제 기구에서는 이것을 유각에서 무릎 모터가 공급해야 한다. 옛 모델의 걸쇠는 이 일을 대가 없이 만들어 주고 있었다. [F13,F10]

## 후속 미세조정

정책이 걸쇠를 쓰도록, 유각 말기에 무릎을 펴는 기존 보상 항(extend, 목표 발목각 135°는 연동 때문에 무릎 0.785 rad에 해당)의 가중치만 0.3에서 2.0으로 올려 같은 정책에서 {RT["finetune"]["steps"]-RT["train"]["steps"]:,} 스텝을 더 학습했다(rev2_ft_extend2). 시드 59개 선별 결과는 다음과 같다.

| 명령 | 전도 | 속도 | 무릎 부하제곱 | 걸쇠 연결 | 접지 무릎각 |
|---|---:|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {tf(c)["falls"]}/{tf(c)["n"]} | {tf(c)["speed"]:.2f} m/s | {tf(c)["heat_knee"]:.2f} | {tf(c)["engagements"]}회 | ' + (f'{ftg[c]["touchdown_mean"][0]:.2f} / {ftg[c]["touchdown_mean"][1]:.2f} rad' if c in ftg else '기록 안 함') + ' |\n' for c in ('2', '4', '5.5', '7')) + f"""
걸쇠는 여전히 한 번도 걸리지 않았다. 7 m/s의 접지각이 {tg["7"]["touchdown_mean"][0]:.2f} rad에서 {ftg["7"]["touchdown_mean"][0]:.2f} rad로 조금 내려갔을 뿐이다. 같은 정책에서 스프링을 빼거나 강성을 85·340 Nm/rad로 바꿔도 결과가 같다(조건당 10회). 이 가중치와 길이의 미세조정으로는 걸쇠를 쓰는 보행에 이르지 못한다. [F13]

## 결론

개정 2는 학습 가능한 모델이다. 정책은 물리적으로 닫힌 조건에서 299회 모두 넘어지지 않고 7 m/s 명령에서 {ta("7")["speed"]:.2f} m/s로 달린다. 그러나 채택하지 않는다. 무릎이 연속 정격의 {ta("2")["heat_knee"]:.1f}–{ta("7")["heat_knee"]:.1f}배로 과열되고, 무작위화 조건에서 자주 넘어진다.

앞 절은 걸쇠가 걸린 비율이 0에 가까우면 스프링이 필요 없다고 보자고 했다. 이 기준은 맞지 않았다. 걸린 비율은 0인데 무릎은 과열되고, 같은 운동에 스프링이 작용하면 부하가 정격 아래로 내려간다. 스프링은 필요하고, 정책이 걸쇠를 쓰는 동작을 찾지 못한 것이다. 한 번의 학습이 스프링 없는 경우와 있는 경우를 함께 가른다는 앞 절의 논리는 탐색이 그 동작에 닿을 때만 성립한다.

남은 선택은 걸쇠를 쓰는 접지를 정책이 찾게 하는 방법이다. 후보는 세 가지다. 명목 자세의 무릎을 정지각 근처로 옮겨 펴진 접지에서 시작하게 하는 것, 입각 중 걸쇠 연결을 직접 보상하는 것, 기구를 접지 순간의 각에서 거는 클러치로 바꾸는 것이다. 세 번째는 기구 설계의 변경이므로 사용자 결정이 필요하다. 1.4에서는 어느 것도 실행하지 않았고, 첫 번째를 다음 절에서 실행했다.

확인하지 못한 것: 학습 시드는 하나다. 속도 차이가 모델 개정 때문인지 시드 때문인지 가르지 못한다. 걸쇠를 쓰는 보행이 실제로 무릎 발열을 정격 아래로 낮추는지는 그런 정책이 나와야 알 수 있다. 선회와 정지, 외란 수준 0 초과는 시험하지 않았다. [F13]
""", 'N09')

na = lambda c: RN['accept'][f'accept@{c}']; nd = lambda c: RN['dr'][f'dr@{c}']; ng = RN['gait']; nr = RN['replay']
insert_page_after('모델 개정 2: 학습과 판정', f"""
# 모델 개정 2: 명목 무릎각을 옮긴 재학습

## 실행한 것

사용자가 앞 절의 세 후보 중 첫 번째를 골랐다(2026-10-12). 명목 자세의 무릎만 1.5 rad에서 1.0 rad로 옮겼다. 연동 상수가 그대로이므로 명목 발목은 −0.9997 rad가 된다. 기구, 걸쇠 규칙, 스프링 [170, 0.956, uni+, latch], 보상, 학습 인자는 rev2_a8_h2와 같다. 처음부터 학습했고 실행 이름은 rev2_n10, 길이는 {RN["train"]["steps"]:,} 스텝이다. 판정 시드와 절차는 앞 절과 같고 rollout은 모두 {RN["rollouts"]:,}회다. [F14]

결정적 평가를 처음 통과한 시점은 {RN["train"]["first_pass_step"]:,} 스텝으로 앞의 두 실행({RT["train"]["first_pass_step"]:,})보다 늦다. 커리큘럼의 명령 속도 상한은 {RN["train"]["final_cmd_max"]:.2f} m/s에서 끝나 rev2_a8_h2와 같다. [F14]

## 고정 시드 판정

괄호 안은 rev2_a8_h2의 값이다.

| 명령 | 전도 | 속도 평균 | 무릎 부하제곱 평균 | 걸쇠 연결 (299회 합) | 자기충돌 |
|---|---:|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {na(c)["falls"]}/{na(c)["n"]} ({ta(c)["falls"]}/{ta(c)["n"]}) | {na(c)["speed"]:.2f} ({ta(c)["speed"]:.2f}) m/s | {na(c)["heat_knee"]:.2f} ({ta(c)["heat_knee"]:.2f}) | {na(c)["engagements"]}회 | {na(c)["self_contact_step_frac"]*100:.1f}% |\n' for c in ('2', '4', '5.5', '7')) + f"""

무릎 부하제곱은 네 속도 모두에서 줄었고 2 m/s에서는 연속 정격 아래다. 4 m/s 이상에서는 {min(na(c)["heat_knee"] for c in ("4", "5.5", "7")):.2f}–{max(na(c)["heat_knee"] for c in ("4", "5.5", "7")):.2f}로 여전히 정격을 넘고, 가장 뜨거운 관절은 모든 조건에서 무릎이다. 7 m/s에서 한 번 넘어졌다. 시작 후 {na("7")["fall_t"][0]:.2f}초의 전도이고 시드는 96221이다. 299회 중 1회 전도일 때 전도 확률의 단측 95% 상한은 {RN["binomial_1_of_299_upper95"]*100:.2f}%다(Clopper–Pearson). 걸쇠가 만든 에너지는 0이다. [F14]

## 무작위화 조건

| 명령 | rev2_n10 전도 | rev2_a8_h2 전도 |
|---|---:|---:|
""" + ''.join(f'| {c} m/s | {nd(c)["falls"]}/{nd(c)["n"]} | {td_(c)["falls"]}/{td_(c)["n"]} |\n' for c in ('2', '4', '7')) + f"""

학습 무작위화를 켠 60초 시험이다. 4 m/s에서는 줄었지만 2 m/s와 7 m/s에서는 줄지 않았다. 외란 여유는 여전히 확보되지 않았다. [F14]

## 걸쇠가 걸리지 않는 이유

| 명령 | 접지 무릎각 (좌, 우) | 발 뗄 때 무릎각 (좌, 우) | 입각 중 최대 (좌, 우) |
|---|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {ng[c]["touchdown_mean"][0]:.2f}, {ng[c]["touchdown_mean"][1]:.2f} rad | {ng[c]["liftoff_mean"][0]:.2f}, {ng[c]["liftoff_mean"][1]:.2f} rad | {ng[c]["stance_max"][0]:.2f}, {ng[c]["stance_max"][1]:.2f} rad |\n' for c in ('2', '4', '7')) + f"""

접지 무릎각은 rev2_a8_h2의 {tg["2"]["touchdown_mean"][1]:.2f}–{tg["4"]["touchdown_mean"][1]:.2f} rad에서 {min(min(ng[c]["touchdown_mean"]) for c in ng):.2f}–{max(max(ng[c]["touchdown_mean"]) for c in ng):.2f} rad로 내려왔다. 정지각 0.956 rad까지 남은 차이는 {max(max(ng[c]["touchdown_mean"]) for c in ng)-0.956:.2f} rad 이하다. 그러나 걸쇠는 유각에서 무릎이 0.956 rad까지 펴져야 걸리므로 이 차이만으로도 걸리지 않는다. 7 m/s에서는 오른 다리 접지각이 정지각에 닿아 299회 중 {RN["seeds_with_engagement_7"]}회에서 걸쇠가 걸렸지만 한 회에 최대 {RN["max_engagements_7"]}번이고, 입각 중 걸린 시간 비율은 {na("7")["latched_stance_frac"]*100:.1f}%다. [F14]

같은 정책에서 스프링을 빼거나 강성을 85·340 Nm/rad로 바꿔도 속도와 무릎 부하제곱이 같다(2·4·7 m/s, 조건당 10회). 즉 발열이 줄어든 것은 스프링의 효과가 아니고 무릎을 더 편 접지의 효과다. [F14]

학습 도중의 300만 스텝 체크포인트는 달랐다. 2 m/s, 시드 2개의 60초 시험에서 걸쇠가 회당 {RN["peek_3M"]["as_trained@2"]["engagements"]/2:.0f}번 걸리고 입각의 {RN["peek_3M"]["as_trained@2"]["latched_stance_frac"]*100:.0f}% 동안 걸려 있었으며 무릎 부하제곱은 {RN["peek_3M"]["as_trained@2"]["heat_knee"]:.2f}였다. 스프링을 빼도 {RN["peek_3M"]["no_spring@2"]["heat_knee"]:.2f}로 같았다. 걸쇠는 걸렸지만 스프링이 일을 하지 않았고, 이후 명령 속도가 오르면서 정책은 걸쇠가 걸리는 접지에서 벗어났다. 표본이 2회뿐인 관찰이다. [F14]

## 같은 운동에 스프링이 작용한다면

| 명령 | 스프링 없음 | 입각 중 k 170, 정지각 0.956 | 필요한 무릎 토크 최대 |
|---|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {nr[c]["none"][0]:.2f} | {nr[c]["stance_k170_q0956"][0]:.2f} | {nr[c]["req_peak"]:.1f} Nm |\n' for c in ('2', '4', '7')) + f"""

운동을 고정하고 입각 내내 스프링이 걸려 있다고 가정한 계산이며 상한이다. 이 가정에서는 세 속도 모두 무릎 부하제곱이 정격 아래로 내려간다. 7 m/s의 여유는 작다. 실제 정책은 스프링이 작용하면 운동을 바꾸므로 이 값이 그대로 나온다는 뜻은 아니다. [F14]

## 결론

rev2_n10도 채택하지 않는다. 4 m/s 이상에서 무릎이 연속 정격을 넘고, 7 m/s에서 299회 중 1회 넘어졌으며, 무작위화 조건에서 자주 넘어진다. 명목 무릎각 이동은 접지를 정지각 가까이로 옮기고 무릎 부하제곱을 {min((1-na(c)["heat_knee"]/ta(c)["heat_knee"])*100 for c in ("2", "4", "5.5", "7")):.0f}–{max((1-na(c)["heat_knee"]/ta(c)["heat_knee"])*100 for c in ("2", "4", "5.5", "7")):.0f}% 줄였지만, 걸쇠를 쓰는 보행을 만들지는 못했다.

남은 차이는 접지각과 정지각 사이의 {max(max(ng[c]["touchdown_mean"]) for c in ng)-0.956:.2f} rad 이하다. 이것을 닫는 후보는 앞 절의 나머지 두 가지다. 입각 중 걸쇠 연결을 직접 보상하는 것과, 접지 순간의 각에서 거는 클러치로 기구를 바꾸는 것이다. 어느 것도 실행하지 않았다. 다음 절의 기하 점검은 두 후보의 전망을 다시 평가한다.

확인하지 못한 것: 학습 시드는 하나다. 두 실행의 차이 중 얼마가 명목 자세 때문이고 얼마가 시드 때문인지 가르지 못한다. 7 m/s의 시작 직후 전도 1회가 명목 자세 변경 때문인지도 확인하지 않았다. 걸쇠를 쓰는 보행이 실제로 무릎 발열을 정격 아래로 낮추는지는 여전히 그런 정책이 나와야 알 수 있다. [F14]
""", 'N10')

gfa = {round(r['knee'], 3): r for r in GP['flat_foot']}
gpm = lambda c, grp, k: [GP['speeds'][c][s][grp][k][0] for s in ('left', 'right')]
gte = lambda c: GY[c]['touchdown_events']
insert_page_after('모델 개정 2: 명목 무릎각을 옮긴 재학습', f"""
# 모델 개정 2: 접지 기하 점검

## 무엇을 확인했나

걸쇠가 걸리지 않는 원인이 다리 길이나 각도에 있는지를 학습 없이 확인했다. rev2_n10 정책의 결정적 rollout(시드 96001, 2·4·7 m/s)에서 접지와 발 떼기 순간의 자세를 기록하고, 같은 자세에서 무릎만 정지각 0.956 rad로 바꾼 기구학 계산과, 운동을 고정한 채 스프링 연결 규칙만 바꾼 재계산을 했다. [F15]

## 다리 기하

무릎과 발목이 1:1로 묶여 있어 발의 기울기는 무릎각과 무관하다(몸통 기울기 + hip 각 + {GP["coupling_c"]:.4f} rad). 무릎은 다리 길이와 발의 앞뒤 위치만 바꾼다. 몸통이 수평이고 발이 평평할 때의 값은 다음과 같다.

| 무릎각 | hip 높이 | 발목의 앞뒤 위치 (무게중심 기준, +가 앞) |
|---|---:|---:|
""" + ''.join(f'| {k:.3f} rad | {gfa[k]["hip_height"]:.3f} m | {gfa[k]["foot_x_com"]*100:+.1f} cm |\n' for k in (0.74, 0.956, 1.05, 1.2, 1.288, 1.5)) + f"""

정지각 0.956 rad에서 hip 높이는 {gfa[0.956]["hip_height"]:.3f} m이고, 무릎을 0.74 rad까지 펴도 {gfa[0.74]["hip_height"]:.3f} m다. 정지각은 다리를 거의 다 편 자리에 있고 그보다 더 펴서 얻는 길이는 {(gfa[0.74]["hip_height"]-gfa[0.956]["hip_height"])*100:.1f} cm뿐이다. [F15]

## 접지 순간

| 명령 | 유각 마지막 무릎각 | 접지 첫 무릎각 | 무릎 각속도 | 접지 후 추가 굽힘 | 발 뗄 때 무릎각 |
|---|---:|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {gte(c)["knee_last_swing"]:.2f} rad | {gte(c)["knee_first_stance"]:.2f} rad | {gte(c)["knee_rate"]:.1f} rad/s | {gte(c)["flexion_after_touchdown"]:.3f} rad | {gte(c)["knee_liftoff"]:.2f} rad |\n' for c in ('2', '4', '7')) + f"""

20 ms 제어 주기로 본 값이며 두 다리의 평균이다. 무릎은 초당 {min(abs(gte(c)["knee_rate"]) for c in ("2", "4", "7")):.1f}–{max(abs(gte(c)["knee_rate"]) for c in ("2", "4", "7")):.1f} rad로 펴지는 도중에 발이 땅에 닿고, 입각 내내 계속 펴져 {min(gte(c)["knee_liftoff"] for c in ("2", "4", "7")):.2f}–{max(gte(c)["knee_liftoff"] for c in ("2", "4", "7")):.2f} rad에서 발을 뗀다. 무릎은 정지각을 지나가지만 접지 뒤에 지나간다. 걸쇠는 유각에서만 걸리도록 되어 있으므로 걸리지 않는다. 접지 후 하중을 받아 더 굽는 양은 {min(gte(c)["flexion_after_touchdown"] for c in ("2", "4", "7")):.3f}–{max(gte(c)["flexion_after_touchdown"] for c in ("2", "4", "7")):.3f} rad에 그친다. [F15]

| 명령 | 접지 때 hip 높이 (좌, 우) | 같은 발 위치에서 무릎이 정지각이면 (좌, 우) | 유각 말기에 무릎을 정지각으로 폈을 때 최대 여유 (좌, 우) |
|---|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {gpm(c, "td", "hip_z")[0]:.3f}, {gpm(c, "td", "hip_z")[1]:.3f} m | {gpm(c, "td", "cf_same_x_clear")[0]*100:.1f}, {gpm(c, "td", "cf_same_x_clear")[1]*100:.1f} cm | {gpm(c, "late_swing", "cf_clear_max")[0]*100:.1f}, {gpm(c, "late_swing", "cf_clear_max")[1]*100:.1f} cm |\n' for c in ('2', '4', '7')) + f"""

둘째 열의 음수는 발이 지면 아래로 들어간다는 뜻이다. 접지 때 hip 높이는 {min(min(gpm(c, "td", "hip_z")) for c in ("2", "4", "7")):.3f}–{max(max(gpm(c, "td", "hip_z")) for c in ("2", "4", "7")):.3f} m로 정지각의 다리 길이 {gfa[0.956]["hip_height"]:.3f} m보다 낮다. 같은 자리에 무릎을 정지각으로 편 채 디디려면 몸이 {min(abs(x) for c in ("2", "4", "7") for x in gpm(c, "td", "cf_same_x_clear"))*100:.1f}–{max(abs(x) for c in ("2", "4", "7") for x in gpm(c, "td", "cf_same_x_clear"))*100:.1f} cm 더 높아야 한다. 유각 말기에는 hip 각을 그대로 두고 무릎을 정지각까지 펼 수 있는 순간이 있지만, 2 m/s에서는 지면 여유가 {min(gpm("2", "late_swing", "cf_clear_max"))*100:.1f}–{max(gpm("2", "late_swing", "cf_clear_max"))*100:.1f} cm다. [F15]

## 정지각을 옮기면

같은 운동에서 스프링(k 170)의 연결 규칙과 정지각만 바꿔 무릎 부하제곱을 다시 계산했다. 걸쇠 열의 괄호는 걸쇠가 걸린 입각의 비율이다.

| 명령 | 스프링 없음 | 걸쇠 0.956 | 걸쇠 1.10 | 걸쇠 1.20 | 입각 내내 0.956 (상한) | 접지각 클러치 k 170 | k 340 | k 680 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {GY[c]["none"]:.2f} | {GY[c]["latch"]["0.956"]["heat"]:.2f} ({GY[c]["latch"]["0.956"]["engaged_frac"]*100:.0f}%) | {GY[c]["latch"]["1.1"]["heat"]:.2f} ({GY[c]["latch"]["1.1"]["engaged_frac"]*100:.0f}%) | {GY[c]["latch"]["1.2"]["heat"]:.2f} ({GY[c]["latch"]["1.2"]["engaged_frac"]*100:.0f}%) | {GY[c]["stance"]["0.956"]:.2f} | {GY[c]["touchdown"]["k170"]["heat"]:.2f} | {GY[c]["touchdown"]["k340"]["heat"]:.2f} | {GY[c]["touchdown"]["k680"]["heat"]:.2f} |\n' for c in ('2', '4', '7')) + f"""

정지각을 1.20 rad로 올리면 걸쇠는 입각의 {min(GY[c]["latch"]["1.2"]["engaged_frac"] for c in ("2", "4", "7"))*100:.0f}% 이상에서 걸리지만 발열은 그대로다. 입각의 무릎이 그 각보다 펴져 있어 스프링이 늘어나지 않기 때문이다. 발열을 줄이는 정지각은 입각의 무릎각보다 펴진 쪽이어야 하고, 그런 각은 이 보행의 유각에서 닿지 않는다. 한 정지각이 두 조건을 함께 만족하지 못한다. 접지각 클러치는 항상 걸리지만 접지 후 굽힘이 작아 k 170에서는 효과가 작고, 세 속도 모두 정격 아래로 내려가는 것은 k 680에서다. 모두 운동을 고정한 계산이다. 정책은 스프링이 작용하면 운동을 바꾼다. [F15]

## 결론

다리 길이의 문제라는 근거는 없다. 문제는 각도의 관계다. 이 보행은 무릎을 펴면서 디디고 입각 내내 펴는 보행이라, 굽힘에 저항하는 병렬 무릎 스프링이 일할 굽힘 구간이 없다. 걸쇠가 걸리려면 발이 닿기 전에 다리를 접지 때보다 길게 폈다가 다시 굽혀 스프링을 미리 당겨야 하는데, 지금의 몸 높이에서는 그럴 지면 여유가 저속에서 1 cm 미만이다.

앞 절의 두 후보를 다시 보면 다음과 같다. 걸쇠 연결을 보상하는 방법은 정책이 몸을 {min(abs(x) for c in ("2", "4", "7") for x in gpm(c, "td", "cf_same_x_clear"))*100:.0f}–{max(abs(x) for c in ("2", "4", "7") for x in gpm(c, "td", "cf_same_x_clear"))*100:.0f} cm 높이고 접지 직전에 무릎을 다시 굽히는 보행을 찾아야 하므로 전망이 좋지 않다. 접지각 클러치는 기하의 제약을 받지 않지만 강성을 두 배에서 네 배로 올려야 할 수 있다. 1.6에서는 어느 것도 실행하지 않았고, 접지각 클러치를 다음 절에서 실행했다.

확인하지 못한 것: 한 정책, 한 시드, 20 ms 표본의 관찰이다. 다른 보행(입각에서 무릎이 크게 굽는 보행)이 가능한지, 그 보행에서 스프링이 얼마나 일하는지는 계산하지 않았다. 링크 길이를 바꾼 모델은 만들지 않았다. 스프링 없이 무릎 구동기의 감속비나 등급을 바꾸는 방법도 검토하지 않았다. [F15]
""", 'N11')

ca = lambda c: RC['accept'][f'accept@{c}']; cd = lambda c: RC['dr'][f'dr@{c}']; cb = lambda l, c: RC['ablate'][f'{l}@{c}']; cz = lambda l, c: RC['zero_shot_on_n10'][f'{l}@{c}']; cg = RC['gait']
insert_page_after('모델 개정 2: 접지 기하 점검', f"""
# 모델 개정 2: 접지각 클러치 학습

## 실행한 것

사용자가 접지각 클러치를 골랐다(2026-10-13). 무릎 스프링의 연결 방식만 바꿨다. 발에 하중이 걸리는 제어 주기에 그 순간의 무릎각을 정지각으로 잡아 연결하고, 발을 떼면 푼다. 굽힘 쪽으로만 힘을 내는 단방향 스프링이고 강성은 340 Nm/rad다. run_env.py에 이미 있던 touchdown 연결 방식이며 코드는 바꾸지 않았다. 나머지 인자는 rev2_n10과 같다(명목 무릎 1.0 rad). 처음부터 학습했고 실행 이름은 rev2_td340, 길이는 {RC["train"]["steps"]:,} 스텝이다. 판정 시드와 절차는 앞 절과 같고 rollout은 모두 {RC["rollouts"]:,}회다. [F16]

학습 전에 rev2_n10 정책에 클러치만 걸어 실제 동역학으로 시험했다(조건당 10회). 전도는 없었고 클러치가 만든 에너지는 0이었다. 무릎 부하제곱은 k 340에서 {cz("td340", "2")["heat_knee"]:.2f}·{cz("td340", "4")["heat_knee"]:.2f}·{cz("td340", "7")["heat_knee"]:.2f}(2·4·7 m/s)로, 운동을 고정한 계산값 {GY["2"]["touchdown"]["k340"]["heat"]:.2f}·{GY["4"]["touchdown"]["k340"]["heat"]:.2f}·{GY["7"]["touchdown"]["k340"]["heat"]:.2f}보다 높았다. [F16]

결정적 평가를 처음 통과한 시점은 {RC["train"]["first_pass_step"]:,} 스텝, 커리큘럼의 명령 속도 상한은 {RC["train"]["final_cmd_max"]:.2f} m/s로 rev2_n10과 같다. [F16]

## 고정 시드 판정

괄호 안은 rev2_n10의 값이다.

| 명령 | 전도 | 속도 평균 | 무릎 부하제곱 | 가장 뜨거운 관절 (부하제곱) | 입각 중 클러치 연결 |
|---|---:|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {ca(c)["falls"]}/{ca(c)["n"]} ({na(c)["falls"]}/{na(c)["n"]}) | {ca(c)["speed"]:.2f} ({na(c)["speed"]:.2f}) m/s | {ca(c)["heat_knee"]:.2f} ({na(c)["heat_knee"]:.2f}) | {RC["hottest"][c]["joint"][0].replace("_joint", "").replace("left_", "").replace("right_", "").replace("_", " ")} ({ca(c)["heat_max"]:.2f}) | {ca(c)["latched_stance_frac"]*100:.0f}% |\n' for c in ('2', '4', '5.5', '7')) + f"""

전도는 네 조건 모두 0/299다. 무릎 부하제곱은 네 속도 모두에서 줄었고 2 m/s와 4 m/s에서 연속 정격 아래다. 5.5 m/s와 7 m/s에서는 {ca("5.5")["heat_knee"]:.2f}와 {ca("7")["heat_knee"]:.2f}로 정격을 조금 넘는다. 7 m/s에서는 꼬리 pitch 관절이 {ca("7")["heat_max"]:.2f}로 가장 뜨겁다. 앞의 두 실행에서는 모든 조건에서 무릎이 가장 뜨거웠으므로 꼬리 과열은 이 정책에서 새로 나타난 것이다. 클러치가 만든 에너지는 0이다. [F16]

![그림 T1. 세 실행의 고정 시드 판정 비교. 왼쪽부터 무릎 부하제곱, 실제 속도, 학습 무작위화 조건의 전도 수.](assets/rev2-compare.png)

## 스프링의 기여

같은 정책에서 평가할 때만 스프링을 빼거나 강성을 바꿨다(조건당 10회). 값은 무릎 부하제곱이고 괄호는 전도 수다.

| 명령 | 학습된 그대로 (k 340) | 스프링 없음 | k 170 | k 680 |
|---|---:|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {cb("as_trained", c)["heat_knee"]:.2f} ({cb("as_trained", c)["falls"]}) | {cb("no_spring", c)["heat_knee"]:.2f} ({cb("no_spring", c)["falls"]}) | {cb("td170", c)["heat_knee"]:.2f} ({cb("td170", c)["falls"]}) | {cb("td680", c)["heat_knee"]:.2f} ({cb("td680", c)["falls"]}) |\n' for c in ('2', '4', '7')) + f"""

2 m/s에서는 스프링을 빼면 무릎 부하제곱이 {cb("as_trained", "2")["heat_knee"]:.2f}에서 {cb("no_spring", "2")["heat_knee"]:.2f}로 오른다. 스프링이 무릎 일을 실제로 덜어 주는 첫 정책이다. 4 m/s에서는 차이가 {cb("no_spring", "4")["heat_knee"]-cb("as_trained", "4")["heat_knee"]:.2f}이고 7 m/s에서는 {cb("no_spring", "7")["heat_knee"]-cb("as_trained", "7")["heat_knee"]:.2f}다. 고속에서 rev2_n10보다 발열이 낮은 것은 대부분 스프링이 아니라 보행이 달라진 결과다. [F16]

## 무작위화 조건

| 명령 | rev2_td340 전도 | rev2_n10 전도 | rev2_a8_h2 전도 |
|---|---:|---:|---:|
""" + ''.join(f'| {c} m/s | {cd(c)["falls"]}/{cd(c)["n"]} | {nd(c)["falls"]}/{nd(c)["n"]} | {td_(c)["falls"]}/{td_(c)["n"]} |\n' for c in ('2', '4', '7')) + f"""

학습 무작위화를 켠 60초 시험이다. 2 m/s의 전도는 rev2_n10보다 늘었다. 외란 여유는 세 실행 모두 확보되지 않았다. [F16]

## 보행

접지 무릎각은 {min(min(cg[c]["touchdown_mean"]) for c in cg):.2f}–{max(max(cg[c]["touchdown_mean"]) for c in cg):.2f} rad, 발 뗄 때는 {min(min(cg[c]["liftoff_mean"]) for c in cg):.2f}–{max(max(cg[c]["liftoff_mean"]) for c in cg):.2f} rad다. 무릎을 펴면서 디디고 입각 내내 펴는 형태는 그대로다. 클러치는 접지 직후의 작은 굽힘 구간에서만 힘을 낸다.

![그림 T2. rev2_n10, 명령 4 m/s. 무릎각이 걸쇠 정지각(점선)을 입각(음영)에 들어선 뒤에 지나가고 스프링 연결 표시는 없다.](assets/rev2-n10-knee.png)

![그림 T3. rev2_td340, 명령 4 m/s. 아래쪽 막대는 클러치가 연결된 구간이다.](assets/rev2-td340-knee.png)

![그림 T4. rev2_td340, 명령 4 m/s, 5초 이후 첫 왼발 접지 프레임. 영상 rev2_td340_4mps.mp4의 한 장면이다.](assets/rev2-td340-touchdown.png)

영상과 그림은 판정과 같은 초기 시드 96001의 결정적 10초 rollout이다. 렌더한 rollout의 상태 기록이 렌더 없이 실행한 같은 rollout과 완전히 같은 것을 확인했고, 정책 파일과 각 파일의 SHA-256을 함께 남겼다. [F16]

## 결론

rev2_td340은 세 실행 중 가장 낫지만 채택하지 않는다. 5.5 m/s 이상에서 무릎이 연속 정격을 넘고, 7 m/s에서 꼬리 pitch가 정격의 {ca("7")["heat_max"]:.1f}배이며, 무작위화 조건에서 자주 넘어진다. 4 m/s까지는 전도 0/299, 모든 관절 정격 아래다.

접지각 클러치는 걸쇠와 달리 실제로 연결되고 저속에서 무릎 일을 덜어 준다. 그러나 이 보행에서는 접지 후 굽힘이 작아 고속에서는 기여가 거의 없다. 남은 문제는 세 가지다. 5.5 m/s 이상의 무릎 과열, 7 m/s의 꼬리 과열, 무작위화 조건의 전도다.

확인하지 못한 것: 학습 시드는 하나다. 실행 사이의 차이 중 얼마가 클러치 때문이고 얼마가 시드 때문인지 가르지 못한다. 꼬리 과열의 원인은 조사하지 않았다. 강성 680이나 다른 정지각 규칙으로 학습한 결과는 없다. 선회와 정지, 외란 수준 0 초과는 시험하지 않았다. [F16]
""", 'N12')

tp = lambda run, c: TL['probe'][f'{run}@{c}']; ty = lambda run, c, l='': TL['dynamics'][f'{run}@{c}{l}']; tgh = lambda g, c: TL['gear_heat'][f'{g}@{c}']; tgz = lambda l, c: TL['gear_zero_shot'][f'{l}@{c}']
insert_page_after('모델 개정 2: 접지각 클러치 학습', f"""
# 모델 개정 2: 꼬리 과열의 원인

## 무엇을 확인했나

rev2_td340이 7 m/s에서 꼬리 pitch 모터의 연속 정격을 넘는 이유를 학습 없이 조사했다. 세 정책의 결정적 rollout(시드 96001, 30초 중 뒤 25초)에서 꼬리 pitch의 토크, 각도, 서보 목표를 1 ms 간격으로 기록하고, 꼬리 pitch 축의 운동방정식을 항별로 나눴다. rollout은 모두 {TL["rollouts"]}회다. [F17]

## 발열의 구성

| 정책 | 2 m/s | 4 m/s | 5.5 m/s | 7 m/s |
|---|---:|---:|---:|---:|
""" + ''.join(f'| {run} | ' + ' | '.join(f'{tp(run, c)["tail"]["heat"]:.2f}' for c in ('2', '4', '5.5', '7')) + ' |\n' for run in ('rev2_a8_h2', 'rev2_n10', 'rev2_td340')) + f"""

꼬리 pitch 부하제곱이다. rev2_td340은 모든 속도에서 다른 두 정책보다 높고 5.5 m/s부터 정격을 넘는다. [F17]

7 m/s의 rev2_td340에서 꼬리 pitch 토크는 평균 {tp("rev2_td340", "7")["tail"]["tau_mean"]:.1f} Nm, 표준편차 {tp("rev2_td340", "7")["tail"]["tau_std"]:.1f} Nm다. 평균은 꼬리 무게를 받치는 토크이고 세 정책, 네 속도에서 {min(r["tail"]["tau_mean"] for r in TL["probe"].values()):.1f}–{max(r["tail"]["tau_mean"] for r in TL["probe"].values()):.1f} Nm로 같다. 이 몫의 부하제곱은 (4.6 / (0.9 × 19.2))² = {(4.6/(.9*19.2))**2:.2f}이다. 나머지는 모두 걸음마다 오르내리는 토크다. 토크 변동 에너지의 {tp("rev2_td340", "7")["tail"]["power_stride_to_4x"]*100:.0f}%가 보폭 주파수 {tp("rev2_td340", "7")["tail"]["stride_hz"]:.2f} Hz의 0.5–4.5배 대역에 있고 가장 큰 성분은 {tp("rev2_td340", "7")["tail"]["peak_hz"]:.2f} Hz(한 걸음마다 한 번)다. 20 Hz 위의 몫은 {tp("rev2_td340", "7")["tail"]["power_above_20hz"]*100:.0f}%다. [F17]

관절 한계와 서보 포화는 원인이 아니다. 꼬리각은 {tp("rev2_td340", "7")["tail"]["q_mean"]:.2f} ± {tp("rev2_td340", "7")["tail"]["q_std"]:.2f} rad이고 범위 ±0.8 rad의 끝 0.03 rad 안에 든 시간은 0이며, 서보 목표가 범위를 벗어난 시간도 0이다. 다만 토크는 구동 한계 {ty("rev2_td340", "7")["peak_limit_Nm"]:.1f} Nm에 시간의 {ty("rev2_td340", "7")["at_peak_limit_frac"]*100:.1f}% 동안 닿는다. [F17]

## 토크가 쓰이는 곳

꼬리 pitch 축의 운동방정식에서 모터 토크는 꼬리가 몸통에 대해 흔들리는 관성항, 몸통의 회전과 병진이 꼬리에 거는 관성항, 중력·코리올리항의 합이다. 1 ms마다 계산한 값의 표준편차는 다음과 같다(7 m/s). 항의 합과 모터 토크의 차이는 {max(r["residual_rms"] for r in TL["dynamics"].values()):.0e} Nm 이하다.

| 정책 | 꼬리 자체 흔들림 | 몸통 회전 | 몸통 병진 | 중력·코리올리 | 모터 토크 | 몸통 pitch 각가속도 |
|---|---:|---:|---:|---:|---:|---:|
""" + ''.join(f'| {run} | {ty(run, "7")["std"]["own_swing"]:.1f} Nm | {ty(run, "7")["std"]["trunk_rotation"]:.1f} Nm | {ty(run, "7")["std"]["trunk_translation"]:.1f} Nm | {ty(run, "7")["std"]["gravity_coriolis"]:.1f} Nm | {ty(run, "7")["tau_std"]:.1f} Nm | {ty(run, "7")["trunk_pitch_acc_rms"]:.0f} rad/s² |\n' for run in ('rev2_a8_h2', 'rev2_n10', 'rev2_td340')) + f"""

꼬리의 관절 기준 관성은 {ty("rev2_td340", "7")["inertia_about_joint"]:.2f} kg·m²다. 앞의 두 항은 크기가 크고 서로 상쇄된다. 몸통이 까딱일 때 꼬리가 공간에서 제자리에 머물면 두 항이 지워져 모터 토크가 작다. rev2_n10은 그렇게 움직인다(35.7과 28.1이 13.3으로 준다). rev2_td340은 상쇄가 적어 모터 토크가 {ty("rev2_td340", "7")["tau_std"]:.1f} Nm다. 대신 몸통 pitch 각가속도가 {ty("rev2_n10", "7")["trunk_pitch_acc_rms"]:.0f}에서 {ty("rev2_td340", "7")["trunk_pitch_acc_rms"]:.0f} rad/s²로, 몸통 pitch 각의 표준편차가 {tp("rev2_n10", "7")["trunk_pitch_std"]:.3f}에서 {tp("rev2_td340", "7")["trunk_pitch_std"]:.3f} rad로 작다. 이 정책은 꼬리를 반작용 질량으로 써서 몸통을 더 고르게 잡고, 그 값을 꼬리 모터의 열로 치른다. 원인은 꼬리의 관성을 걸음 주파수로 가감속하는 토크다. [F17]

같은 정책에서 꼬리를 명목 자세에 고정하면(시드 1개) 30초 동안 넘어지지 않지만 속도가 4 m/s 명령에서 {tp("rev2_td340", "4@locked")["speed"]:.2f} m/s, 7 m/s 명령에서 {tp("rev2_td340", "7@locked")["speed"]:.2f} m/s로 떨어지고, 꼬리 pitch 부하제곱은 {tp("rev2_td340", "4@locked")["tail"]["heat"]:.2f}와 {tp("rev2_td340", "7@locked")["tail"]["heat"]:.2f}다. 꼬리를 들고만 있어도 정격의 절반을 넘게 쓴다. 꼬리 없이 학습한 정책이 아니므로 꼬리가 꼭 필요하다는 근거는 아니다. [F17]

![그림 T5. 왼쪽: 세 정책의 꼬리 pitch 부하제곱. 가운데: 7 m/s에서 꼬리 pitch 토크의 성분별 표준편차. 오른쪽: rev2_td340에 꼬리 pitch 감속만 적용했을 때의 부하제곱.](assets/rev2-tail-heat.png)

## 감속을 넣으면

꼬리 pitch는 지금 B급 모듈 직결(1:1)이다. hip pitch에 이미 쓰는 것과 같은 감속을 꼬리 pitch에 넣고 같은 정책을 재학습 없이 실행했다. 감속에 따라 반사 관성과 관절 토크 한계도 함께 바뀐다.

| 감속비 | 꼬리 pitch 부하제곱 (5.5, 7 m/s) | 무릎 부하제곱 (5.5, 7 m/s) | 속도 (5.5, 7 m/s) | 전도 (4·5.5·7 m/s, 각 10회) |
|---|---:|---:|---:|---:|
""" + ''.join(f'| {g}:1 | {tgh(g, "5.5")["tail_pitch"]:.2f}, {tgh(g, "7")["tail_pitch"]:.2f} | {tgh(g, "5.5")["knee"]:.2f}, {tgh(g, "7")["knee"]:.2f} | {tgh(g, "5.5")["speed"]:.2f}, {tgh(g, "7")["speed"]:.2f} m/s | {tgz(l, "4")["falls"]}, {tgz(l, "5.5")["falls"]}, {tgz(l, "7")["falls"]} |\n' for g, l in (('1', 'as_trained'), ('1.5', 'tail_1p5'), ('2', 'tail_2p0'))) + f"""

1.5:1에서 7 m/s의 꼬리 pitch 부하제곱은 {tgh("1", "7")["tail_pitch"]:.2f}에서 {tgh("1.5", "7")["tail_pitch"]:.2f}로 내려간다. 전도는 없고 속도는 조금 오른다. 꼬리 각속도의 최댓값은 {ty("rev2_td340", "7")["qd_absmax"]:.1f} rad/s이므로 1.5:1에서 모터 쪽 {ty("rev2_td340", "7")["qd_absmax"]*1.5:.1f} rad/s이고 B급의 한계 33 rad/s 안이다. 무릎 부하제곱은 바뀌지 않는다. [F17]

## 결론

꼬리 과열은 관절 한계나 서보 문제가 아니고 중력 지지도 아니다. 관성 {ty("rev2_td340", "7")["inertia_about_joint"]:.2f} kg·m²인 꼬리를 걸음마다 가감속하는 토크가 직결 B급 모듈의 연속 정격을 넘는 것이다. rev2_td340이 다른 정책보다 높은 것은 꼬리로 몸통 pitch를 더 강하게 잡기 때문이다.

꼬리 pitch에 1.5:1 감속을 넣으면 재학습 없이도 정격 아래로 내려간다. 이것은 구동계 설계의 변경이므로 사용자 결정이 필요하다. 감속을 넣은 뒤에도 5.5 m/s 이상의 무릎 과열({tgh("1.5", "5.5")["knee"]:.2f}, {tgh("1.5", "7")["knee"]:.2f})과 무작위화 조건의 전도는 남는다.

확인하지 못한 것: 감속 시험은 조건당 10회이고 고정 시드 299회 판정이 아니다. 감속기의 질량, 백래시, 효율 변화는 모델에 없다. 감속을 넣고 학습하면 정책이 꼬리를 더 세게 쓸 수 있다. 꼬리의 질량이나 길이를 바꾼 모델은 만들지 않았다. 꼬리 yaw는 모든 조건에서 {max(r["heat"]["tail_yaw"] for r in TL["probe"].values()):.2f} 이하다. [F17]
""", 'N13')

# ---------------------------------------------------------------- verification table and correction history
fix('| 새 강화학습·보행 시험 | 실행하지 않음; goal paused 유지 |',
    f'| Pinocchio ↔ MuJoCo (6개 자세) | 위치 {mx["max_body_pos_err_m"]:.1e} m, 질량행렬 {mx["mass_matrix_abs_err"]:.1e}, 중력항 {mx["gravity_bias_abs_err"]:.1e} Nm |\n'
    f'| 4절 링크 닫힌 식 ↔ 각도 표현 | 기호 차이 0, 수치 {fb["max_abs_err_N_vs_angle_formula"]:.1e} |\n'
    f'| 무릎 정적 토크 닫힌 식 ↔ 증거148 | {f(e148["this_knee_torque"], 4)} 대 {f(e148["archived_knee_torque"], 4)} Nm |\n'
    f'| 명목 자세 정적 유지 | 불가: {f(nom["knee_motor_spring_on"], 2)} Nm 필요, 피크 {f(nom["knee_peak"], 2)} Nm |\n'
    f'| 게이트 진단 rollout {GR["timing"]["rollouts"]}회 | 기준선은 걸쇠 주입 에너지와 7 m/s 다리 통과에 의존 |\n'
    f'| 모델 개정 2 기본값 회귀 (rollout {R2["rollouts"]["regression"]}개) | 수정 전 기록과 동일 |\n'
    f'| 모델 개정 2 동결 정책 {R2["rollouts"]["options"]+R2["rollouts"]["springs"]}회 | 주입 에너지 0; 4·7 m/s 전도 10/10 |\n'
    f'| 개정 2 학습 {RT["training_steps"]:,} 스텝 (학습 1회 + 미세조정 1회) | 고정 시드 판정 0/299 전도 (4개 속도); 무릎 부하제곱 {RT["accept"]["accept@2"]["heat_knee"]:.2f}–{RT["accept"]["accept@7"]["heat_knee"]:.2f}; 걸쇠 연결 0회; 미채택 |\n'
    f'| 명목 무릎각 1.0 rad 재학습 {RN["train"]["steps"]:,} 스텝 | 전도 0·0·0·1/299; 무릎 부하제곱 {RN["accept"]["accept@2"]["heat_knee"]:.2f}–{RN["accept"]["accept@5.5"]["heat_knee"]:.2f}; 걸쇠는 7 m/s에서 입각의 {RN["accept"]["accept@7"]["latched_stance_frac"]*100:.1f}%만 연결; 미채택 |', 'F10')
fix('| R16 | 양식 출처를 실제 8쪽 양식으로 정정 | 없음(문서 항목) |',
    '| R16 | 양식 출처를 실제 8쪽 양식으로 정정 | 없음(문서 항목) |\n'
    '| R17 (1.1) | 정적 지면반력 작용점을 COM 아래로 정정; 무릎 정적 유지 토크와 스프링 예압을 합산 | 걸쇠 시작 상태 결정(G5) |\n'
    '| R18 (1.1) | 4절 링크 닫힌 식, 입력측 사점과 유효 구간 추가 | 무릎 범위 계약(G4) |\n'
    '| R19 (1.1) | 역기구학 절의 도달 조건 표기 정정: PDF에서 절댓값 기호가 빠졌음 | 없음(문서 항목) |\n'
    '| R20 (1.1) | 연동 때문에 무릎 실효 상한이 2.5003 rad임을 명시 | 관절 범위 계약 |\n'
    '| R21 (1.1) | 명목 높이가 13.0 mm 떠 있는 리셋 값임을 명시 | 시작 높이(G7) |\n'
    '| R22 (1.1) | 맞바람 예시의 공력 동력 문구 정정 | 없음(문서 항목) |\n'
    '| R23 (1.2) | 게이트 G2–G9를 기준선 정책으로 측정하고 결론 추가; 기존 주행 기록의 물리적 유효성 한계 명시 | 선택 옵션 구현, 재학습, 299회 판정 |\n'
    '| R24 (1.3) | 여섯 수정을 run_env.py의 model_rev 2로 구현; 걸쇠를 단방향 스프링과 신전 통과 연결로 보완; 스프링 후보를 q0 0.956 한 가지로 축소 | 개정 2 학습 1회, 299회 판정 |\n'
    '| R25 (1.4) | 개정 2 학습과 299회 판정 결과 추가; "걸쇠가 걸린 비율로 스프링 필요를 판단"한다는 1.3의 기준을 철회 | 걸쇠를 쓰는 접지를 찾게 하는 방법 결정 |\n'
    '| R26 (1.5) | 명목 무릎각 1.0 rad 재학습과 판정 결과 추가 | 걸쇠 연결 보상 또는 접지각 클러치의 결정 |\n'
    '| R27 (1.6) | 접지 기하 점검 추가: 걸쇠가 걸리는 정지각과 발열을 줄이는 정지각이 겹치지 않음 | 접지각 클러치(강성 포함) 또는 다른 방법의 결정 |\n'
    '| R28 (1.7) | 접지각 클러치 학습과 판정 결과, 영상·차트 근거 추가 | 고속 무릎 과열, 꼬리 과열, 무작위화 조건 전도 |\n'
    '| R29 (1.8) | 꼬리 과열 원인 조사 추가: 관성 가감속 토크가 원인, 1.5:1 감속에서 정격 아래 | 꼬리 pitch 감속 적용 여부 결정 |', 'F11')
fix('문서 작업 기준 코드: 6c30fd6.', '최종판 1.1의 계산 근거는 evidence/final 폴더의 스크립트와 JSON이다. 문서 작업 기준 코드: 6c30fd6.', 'F12')

# ---------------------------------------------------------------- references
s = s.rstrip()+'''

---PAGE---

# 참고문헌·원자료 9

[F1] 최종판 정적 재계산 (2026-10-12)

model_dump, crosscheck_pinocchio, fourbar_closed_form, static_stand_map, knee_static_limits, parser_check; mj_forward와 정역학만 사용

evidence/final 폴더

[F2] Pinocchio 4.1.0

README: 지원 형식(URDF·SDF·MJCF), 폐루프 기구 지원; 이 Mac에서 설치·MJCF 로딩·수치 대조 실행

https://github.com/stack-of-tasks/pinocchio

[F3] SymPy Mechanics 1.14.0

Kane·Lagrange 방법과 선형화 설명; 이번에는 음함수 미분과 식 정리에 사용

https://docs.sympy.org/latest/explanation/modules/physics/mechanics/index.html

[F4] MuJoCo 문서: XML Reference, Modeling

tendon springlength의 dead-band, equality/joint, geom priority와 friction 조합 규칙 원문 확인

https://mujoco.readthedocs.io/en/stable/XMLreference.html

[F5] 증거148 정적 평형 검색

margin-search.jsonl의 평형 자세·요구 토크·발 수직력; README의 전도 시각

https://github.com/djfksjd/ROS_RAPTER/tree/6c30fd6/docs/evidence/148-static-equilibrium

[F6] RAPTOR 검증 도구 조사 보고서 (2026-10-12)

17개 도구의 공식 저장소·PyPI 메타데이터·README 인용과 로컬 설치 시험 기록

~/Documents/RAPTOR_Tooling_Research_20261012/report.md

[F7] Drake 1.57.0 MJCF 파서

로컬 로딩 시 출력된 미지원 경고와 공식 파싱 문서

https://drake.mit.edu/doxygen_cxx/group__multibody__parsing.html

[F9] 게이트 진단 rollout (2026-10-12)

gates.py, gate_rollouts*.jsonl, gate_results.json; 동결 정책 scr_a8_h2, 조건당 4·7 m/s × 시드 10개 × 60초, 학습 없음

evidence/gates 폴더

[F10] 모델 개정 2 구현과 무학습 검증 (2026-10-12)

run_env_model_rev2.diff, regress.py, rev2_eval.py, spring_id.py, drop_rig.py, static_knee_capacity.py, rev2_results.json; 동결 정책 scr_a8_h2와 무정책 시험, 학습 없음

evidence/rev2 폴더

[F13] 모델 개정 2 학습과 판정 (2026-10-12)

rev2_accept.py, accept.jsonl, dr.jsonl, dr_baseline_rev1.jsonl, rev1.jsonl, latch_probe.py, spring_id_rev2.py, ft_extend2_*.jsonl, aggregate_train.py, train_results.json; 학습 실행 sim/rl/runs/rev2_a8_h2, rev2_ft_extend2

evidence/rev2/train 폴더

[F14] 명목 무릎각 1.0 rad 재학습과 판정 (2026-10-12)

n10_accept.jsonl, n10_dr.jsonl, n10_ablate.jsonl, n10_3M_peek.jsonl, n10_latch_probe.py, n10_spring_id.py, aggregate_n10.py, n10_results.json; 학습 실행 sim/rl/runs/rev2_n10

evidence/rev2/train 폴더

[F15] 접지 기하 점검 (2026-10-13)

geometry_probe.py, geometry_probe_n10.json, geometry_replay.py, geometry_replay_n10.json; 대상 정책 sim/rl/runs/rev2_n10

evidence/rev2/train 폴더

[F16] 접지각 클러치 학습과 판정, 시각 자료 (2026-10-13)

td340_accept.jsonl, td340_dr.jsonl, td340_ablate.jsonl, n10_clutch_zero_shot.jsonl, td340_latch_probe.py, aggregate_td340.py, td340_results.json, media/make_media.py, media/*.mp4, media/*.gif, media/*.png; 학습 실행 sim/rl/runs/rev2_td340

evidence/rev2/train 폴더

[F17] 꼬리 과열 원인 조사 (2026-10-13)

tail_probe.py, tail_probe.json, tail_dynamics.py, tail_dynamics.json, td340_tailgear.jsonl, td340_tailgear_heat.json, aggregate_tail.py, tail_results.json, media/tail_heat.png; 대상 정책 rev2_a8_h2, rev2_n10, rev2_td340

evidence/rev2/train 폴더

[F8] MuJoCo Menagerie, Agility Cassie

cassie.xml: equality connect 4개, 스프링 관절 stiffness, 좌우 충돌 class; 구조 참고이며 RAPTOR 수치 근거 아님

https://github.com/google-deepmind/mujoco_menagerie/tree/main/agility_cassie
'''

(HERE/'report.ko.md').write_text(s+'\n')
manifest = dict(date='2026-10-12', version='final 1.8', base='integrated 1.0 (2026-10-11)', commit='6c30fd6fd091082ba261f49d55c3c45174ca997a',
                scope='document correction, static recalculation, diagnostic rollouts of the frozen baseline policy, and opt-in model_rev 2 options in sim/rl/run_env.py (defaults unchanged, uncommitted); three training runs and one fine-tune on model_rev 2 with fixed-seed acceptance',
                source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(), report_sha256=hashlib.sha256((HERE/'report.ko.md').read_bytes()).hexdigest(),
                evidence_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(EV.iterdir()) if p.is_file()},
                tools=dict(mujoco=cc['mujoco'], pinocchio=cc['pinocchio'], sympy='1.14.0', drake=dk['version']), new_dynamic_rollouts=GR['timing']['rollouts'], new_training_steps=RT['training_steps']+RN['train']['steps']+RC['train']['steps'], runtime_fixes_applied='opt-in model_rev 2 (default model_rev 1 reproduces earlier records)', rev2_rollouts=R2['rollouts'], changes=changes)
(HERE/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
print('report.ko.md written:', len(s.splitlines()), 'lines,', len(changes), 'changes,', s.count('---PAGE---')+1, 'pages blocks')
