"""Verify the final report: citations, corrections, arithmetic re-derivations against evidence/final/*.json, and the rendered
PDF (page text, margins, the |D| expression). Run from the project root with a Python that has pdfplumber and pypdf."""
from pathlib import Path
import re, json, hashlib, math
import pdfplumber
from pypdf import PdfReader

R = Path(__file__).resolve().parent
pdf = R/'RAPTOR_최종설계검증보고서_20261012.pdf'
s = (R/'report.ko.md').read_text()
J = lambda n: json.loads((R/'evidence'/'final'/n).read_text())
cc, fb, sm, kl, pc, lq = J('crosscheck_pinocchio.json'), J('fourbar_closed_form.json'), J('static_stand_map.json'), J('knee_static_limits.json'), J('parser_check.json'), J('lqr_recheck.json')
checks = {}

# --- citations
refs = {x['id'] for x in map(json.loads, (R/'sources.jsonl').read_text().splitlines())}
cited = set()
for m in re.findall(r'\[((?:[EDVNF]\d+)[^\]\n]*)\]', s):
    cited.update(re.findall(r'\b[EDVNF]\d+\b', m))
assert not (cited-refs), cited-refs
checks['citations'] = dict(cited=len(cited), registered=len(refs), missing=[])
for code in re.findall(r'```[\s\S]*?```', s):
    assert '| ---' not in code, code[:80]

# --- no prose paragraph may start with a pipe (the 1.0 defect that dropped the absolute-value bars)
incode = False; lines = s.splitlines()
for i, l in enumerate(lines):
    if l.startswith('```'):
        incode = not incode; continue
    if not incode and l.lstrip().startswith('|'):
        assert l.rstrip().endswith('|'), (i+1, l[:60])
assert '|D| ≤ 1' in s and '|D|≤1은' not in s

# --- inherited correction assertions from 1.0
assert s.count('성공 0/4, 전도 4/4') >= 3
assert 'a_i,W = R_W,parent × R_parent,jointMount × a_i,joint' in s
assert '| IV | toe_1 | toe_3 |' in s and '| II | toe_3 | toe_1 |' in s and '| III | toe_2 | toe_2 |' in s
assert 'Q_act,required = M v_dot + h_rigid − J_c^T F' in s

# --- arithmetic re-derivations (independent of the scripts that produced the JSON)
g, M = 9.81, 2.55+2.57+2*3.0856
close = lambda a, b, tol=1e-6: math.isclose(a, b, rel_tol=0, abs_tol=tol)
assert close(M, 11.2912, 1e-12) and close(cc['total_mass'], M, 1e-9)
Fz = M*g/2
assert close(Fz, 55.383336, 1e-9) and close(kl['Fz'], Fz, 1e-9)
m_shin, m_dist = .228*.58+.08, .192*.58+.04*.6+6*.01*.6
K_load = Fz*.38-g*(m_shin*.19+m_dist*.38)
assert close(K_load, kl['K_load'], 1e-9) and close(K_load, 20.011278, 1e-6)
k = 170*M/11.36
assert close(k, sm['constants']['k_spring'], 1e-9)
th2 = -.5273+1.5
motor_nom = k*(1.5-.956)-K_load*math.sin(th2)
assert close(motor_nom, sm['nominal_pose']['knee_motor_spring_on'], 1e-9), motor_nom
a_, b_, c_, beta, lf = .083, .194, .102, math.radians(69), .22


def fourbar(q):
    Qx, Qz = c_*math.cos(q+beta), -lf+c_*math.sin(q+beta); d = math.hypot(Qx, Qz)
    phi = math.atan2(Qz, Qx)+math.acos((a_*a_+d*d-b_*b_)/(2*a_*d))
    return c_*(lf*math.cos(q+beta)-a_*math.sin(q+beta-phi))/(a_*(lf*math.cos(phi)-c_*math.sin(q+beta-phi)))


N15 = fourbar(1.5)
assert close(N15, -1.108553, 1e-6), N15
peak, cont = .9*48*abs(N15), .9*19.2*abs(N15)
assert close(peak, sm['nominal_pose']['knee_peak'], 1e-9) and close(cont, sm['nominal_pose']['knee_cont'], 1e-9)
assert motor_nom > peak > cont
e = sm['evidence148_crosscheck']
m148 = k*(e['qk']-.956)-K_load*math.sin(e['pitch']+e['qh']+e['qk'])
assert close(m148, e['this_knee_torque'], 1e-9) and abs(m148-e['archived_knee_torque']) < .01
assert close(math.hypot(c_*math.cos(fb['closure_upper_rad']+beta), -lf+c_*math.sin(fb['closure_upper_rad']+beta)), a_+b_, 1e-9)
for t in (f'{motor_nom:.4f}', f'{peak:.4f}', f'{cont:.4f}', f'{K_load:.6f}', f'{kl["best_case_upper_knee_cont"]:.4f}', f'{kl["best_case_upper_knee_peak"]:.4f}',
          f'{fb["closure_upper_rad"]:.6f}', f'{e["this_knee_torque"]:.4f}', f'{e["archived_knee_torque"]:.4f}', '0.7159–2.2454', '2.5003', '15.925', '12.997'):
    assert t in s, t
assert close(.5*1.225*.1*81*4, 19.845) and close(19.845-.5*1.225*.1*16*4, 15.925)
assert max(cc['max'].values()) < 1e-9
assert close(lq['full_closed_loop_radius'], .9999999996215, 1e-12) and close(lq['reachable_radius'], .9910022511, 1e-9) and lq['eigs_above_one'] == 0
assert pc['pinocchio']['loaded'] and pc['pinocchio']['mass_matrix_max_abs_diff_vs_mujoco'] < 1e-9 and pc['drake']['actuators'] == 0
checks['arithmetic'] = dict(total_mass=M, Fz=Fz, K_load=K_load, k_spring=k, knee_motor_nominal=motor_nom, knee_peak=peak, knee_cont=cont, N_at_1p5=N15, evidence148_closed_form=m148,
                            pinocchio_vs_mujoco_max=max(cc['max'].values()), status='passed')

# --- gate rollouts: recompute the headline numbers from the raw rows
GD = R/'evidence'/'gates'; rd = lambda n: [json.loads(l) for l in (GD/n).read_text().splitlines()]
raw = [r for r in rd('gate_rollouts.jsonl') if r['mode'] == 'policy' and r['kind'] not in ('base', 'selfcol')]+rd('gate_rollouts_selfcol.jsonl')+rd('gate_rollouts_combo.jsonl')
sel = lambda k, c: [r for r in raw if r['kind'] == k and r['cmd'] == c]
falls = lambda k, c: sum(r['fell'] for r in sel(k, c))
assert all(len(sel(k, c)) == 10 for k in {r['kind'] for r in raw} for c in (4., 7.))
assert (falls('base', 4.), falls('base', 7.), falls('selfcol', 4.), falls('selfcol', 7.)) == (0, 0, 0, 5)
assert all(falls(k, c) == 10 for k in ('latch_nofb', 'latch_clutch') for c in (4., 7.))
assert all(falls(k, c) == 0 for k in ('mu_lo', 'mu_hi', 'knee30', 'nogap', 'armature') for c in (4., 7.))
inj = {c: sum(r['energy_injected_J'] for r in sel('base', c))/sum(r['t'] for r in sel('base', c)) for c in (4., 7.)}
mot = {c: sum(r['motor_positive_work_J'] for r in sel('base', c))/sum(r['t'] for r in sel('base', c)) for c in (4., 7.)}
gr = json.loads((GD/'gate_results.json').read_text())['conditions']
assert close(inj[4.], gr['base@4']['inject_W'], 1e-9) and close(inj[7.], gr['base@7']['inject_W'], 1e-9) and close(mot[7.], gr['base@7']['motor_pos_W'], 1e-9)
assert sum(r['engagements'] for c in (4., 7.) for r in sel('base', c)) == sum(r['engage_nonzero'] for c in (4., 7.) for r in sel('base', c)) == 5756
for t in (f'{inj[4.]:.1f} W', f'{inj[7.]:.1f} W', f'{mot[4.]:.1f} W', f'{mot[7.]:.1f} W', '5,756건', '10회 중 5회'):
    assert t in s, t
assert close(1-.05**(1/59), .0495076, 1e-6) and close(1-.05**(1/299), .0099691, 1e-6) and close(81*1.002e-4*1.5**2, .0182614, 1e-6)
checks['gates'] = dict(rows=len(raw), base_inject_W=inj, base_motor_positive_W=mot, selfcol_falls_7=falls('selfcol', 7.), latch_variant_falls=40, status='passed')

# --- model_rev 2: recompute the headline numbers from the raw rows
RD = R/'evidence'/'rev2'; r2 = lambda n: [json.loads(l) for l in (RD/n).read_text().splitlines()]
opt, spr = r2('options.jsonl'), r2('springs.jsonl'); reg = json.loads((RD/'regression.json').read_text())
of = lambda lab, c: [r for r in opt if r['label'] == lab and r['cmd'] == c]
assert reg['passed'] and all(x['identical'] for x in reg['rows']) and len(reg['rows']) == 4
assert all(len(of(lab, c)) == 10 for lab in {r['label'] for r in opt} for c in (4., 7.))
assert [sum(r['fell'] for r in of(lab, c)) for lab in ('rev1', 'latch', 'selfcol', 'rev2_all', 'rev2_keep_fallback') for c in (4., 7.)] == [0, 0, 10, 10, 0, 5, 10, 10, 0, 0]
assert all(sum(r['fell'] for r in of(lab, c)) == 0 for lab in ('friction', 'knee', 'start', 'armature') for c in (4., 7.))
assert sum(r['energy_injected_J'] for r in of('rev2_all', 4.)+of('rev2_all', 7.)+spr) == 0. and sum(r['engage_nonzero'] for r in spr) == 0
assert all(r['fell'] for r in spr if r['cmd'] in (4., 7.)) and len(spr) == 570
assert all(abs(r['pad_clearance_start_m']-1e-4) < 1e-9 for r in of('rev2_all', 4.)) and all(r['rig_tilt_max'] < .01 for r in json.loads((RD/'drop_rig.json').read_text()))
cap = {x['knee']: x for x in json.loads((RD/'static_knee_capacity.json').read_text())}
for q in (.956, 1.3):
    N = abs(fourbar(q)); sn = math.sin(q-.5273); gr_ = g*(m_shin*.19+m_dist*.38)
    assert close(cap[q]['F_cont_BW'], (.9*19.2*N/sn+gr_)/.38/(M*g), 1e-9) and close(cap[q]['N'], N, 1e-9)
assert close(81*1.002e-4*1.7**2, .0234551, 1e-6)
for t in ('체중의 1.73배', '체중의 0.78배', '0.01495'):
    assert t in s, t
checks['model_rev2'] = dict(regression_identical=True, option_rows=len(opt), spring_rows=len(spr), injected_energy_J_rev2=0.0, falls_rev2_all=20, F_cont_BW={q: cap[q]['F_cont_BW'] for q in (.956, 1.3)}, status='passed')

# --- model_rev 2 training: recompute the headline numbers from the raw rows
TD = RD/'train'; tr_ = lambda n: [json.loads(l) for l in (TD/n).read_text().splitlines()]
acc, drn, drb, fta = tr_('accept.jsonl'), tr_('dr.jsonl'), tr_('dr_baseline_rev1.jsonl'), tr_('ft_extend2_accept59.jsonl'); RT = json.loads((TD/'train_results.json').read_text())
assert len(acc) == 4*299 and sorted({r['init_seed'] for r in acc}) == list(range(96001, 96300)) and all(sum(r['cmd'] == c for r in acc) == 299 for c in (2., 4., 5.5, 7.))
assert sum(r['fell'] for r in acc) == 0 and sum(r['engagements'] for r in acc+fta) == 0 and sum(r['energy_injected_J'] for r in acc+drn+fta) == 0.
assert all(r['heat_knee'] > 1. and r['heat_argmax'].endswith('knee_pitch_joint') for r in acc) and sum(r['fell'] for r in fta) == 0 and all(r['heat_knee'] > 1. for r in fta)
assert [sum(r['fell'] for r in rows_ if r['cmd'] == c) for rows_ in (drn, drb) for c in (2., 4., 7.)] == [RT[k][f'dr@{c}']['falls'] for k in ('dr', 'dr_baseline_rev1') for c in ('2', '4', '7')]
assert close(RT['binomial']['upper95_one_sided'], 1-.05**(1/299), 1e-12) and .99**299 < .05 < .99**298
for c in ('2', '4', '5.5', '7'):
    g = [r for r in acc if r['cmd'] == float(c)]; assert close(RT['accept'][f'accept@{c}']['heat_knee'], sum(r['heat_knee'] for r in g)/299, 1e-9) and close(RT['accept'][f'accept@{c}']['speed'], sum(r['speed_tail'] for r in g)/299, 1e-9)
    assert f'{RT["accept"][f"accept@{c}"]["heat_knee"]:.2f}' in s and f'{RT["accept"][f"accept@{c}"]["speed"]:.2f}' in s
for t in ('0/299', '0.997%', '모델 개정 2: 학습과 판정'):
    assert t in s, t
checks['model_rev2_training'] = dict(accept_rows=len(acc), falls=0, latch_engagements=0, knee_heat={c: RT['accept'][f'accept@{c}']['heat_knee'] for c in ('2', '4', '5.5', '7')},
                                     dr_falls={c: RT['dr'][f'dr@{c}']['falls'] for c in ('2', '4', '7')}, training_steps=RT['training_steps'], status='passed')

nacc = [json.loads(l) for l in (Path(__file__).resolve().parent/'evidence'/'rev2'/'train'/'n10_accept.jsonl').read_text().splitlines()]
RN = json.loads((Path(__file__).resolve().parent/'evidence'/'rev2'/'train'/'n10_results.json').read_text())
assert len(nacc) == 1196 and sum(r['fell'] for r in nacc) == 1 and sum(r['energy_injected_J'] for r in nacc) == 0
for c in ('2', '4', '5.5', '7'):
    g = [r for r in nacc if r['cmd'] == float(c) and not r['fell']]; a = RN['accept'][f'accept@{c}']
    assert close(a['heat_knee'], sum(r['heat_knee'] for r in g)/len(g), 1e-9) and close(a['speed'], sum(r['speed_tail'] for r in g)/len(g), 1e-9) and a['falls'] == 299-len(g)
    assert f'{a["heat_knee"]:.2f} ({RT["accept"][f"accept@{c}"]["heat_knee"]:.2f})' in s, c
p1 = RN['binomial_1_of_299_upper95']; assert abs((1-p1)**299+299*p1*(1-p1)**298-.05) < 1e-9
for t in ('1/299', f'{p1*100:.2f}%', '모델 개정 2: 명목 무릎각을 옮긴 재학습', '[F14]'):
    assert t in s, t
checks['model_rev2_nominal_knee_run'] = dict(accept_rows=len(nacc), falls={c: RN['accept'][f'accept@{c}']['falls'] for c in ('2', '4', '5.5', '7')}, knee_heat={c: RN['accept'][f'accept@{c}']['heat_knee'] for c in ('2', '4', '5.5', '7')},
                                             dr_falls={c: RN['dr'][f'dr@{c}']['falls'] for c in ('2', '4', '7')}, latched_stance_frac_7=RN['accept']['accept@7']['latched_stance_frac'], training_steps=RN['train']['steps'], status='passed')

T_ = Path(__file__).resolve().parent/'evidence'/'rev2'/'train'; GP = json.loads((T_/'geometry_probe_n10.json').read_text()); GY = json.loads((T_/'geometry_replay_n10.json').read_text())
fa = {round(r['knee'], 3): r for r in GP['flat_foot']}
assert all(abs(r['foot_pitch']) < 1e-3 for r in GP['flat_foot']) and fa[0.74]['hip_height'] > fa[0.956]['hip_height'] > fa[1.5]['hip_height']
for c in ('2', '4', '7'):
    assert close(GY[c]['none'], json.loads((T_/'n10_spring_id.json').read_text())[c]['none'][0], 1e-9) and GY[c]['latch']['1.2']['engaged_frac'] > .8 and close(GY[c]['latch']['1.2']['heat'], GY[c]['none'], 1e-2)
    assert GY[c]['latch']['0.956']['engaged_frac'] < .05 and GY[c]['touchdown']['k680']['heat'] < 1 and all(GP['speeds'][c][s]['td']['cf_same_x_clear'][0] < 0 for s in ('left', 'right'))
for t in ('모델 개정 2: 접지 기하 점검', '[F15]', f'{fa[0.956]["hip_height"]:.3f} m'):
    assert t in s, t
checks['geometry_check'] = dict(hip_height_at_q0=fa[0.956]['hip_height'], hip_height_at_0p74=fa[0.74]['hip_height'], heat_none={c: GY[c]['none'] for c in ('2', '4', '7')},
                                latch_q0_1p2={c: GY[c]['latch']['1.2'] for c in ('2', '4', '7')}, touchdown_clutch={c: {k: v['heat'] for k, v in GY[c]['touchdown'].items()} for c in ('2', '4', '7')}, status='passed')

RC = json.loads((T_/'td340_results.json').read_text()); cacc = [json.loads(l) for l in (T_/'td340_accept.jsonl').read_text().splitlines()]
assert len(cacc) == 1196 and sum(r['fell'] for r in cacc) == 0 and sum(r['energy_injected_J'] for r in cacc) == 0
for c in ('2', '4', '5.5', '7'):
    g = [r for r in cacc if r['cmd'] == float(c)]; a = RC['accept'][f'accept@{c}']
    assert close(a['heat_knee'], sum(r['heat_knee'] for r in g)/299, 1e-9) and close(a['speed'], sum(r['speed_tail'] for r in g)/299, 1e-9) and close(a['heat_max'], sum(r['heat_max'] for r in g)/299, 1e-9)
    assert f'{a["heat_knee"]:.2f} ({RN["accept"][f"accept@{c}"]["heat_knee"]:.2f})' in s, c
for k, v in RC['media'].items():
    assert v['render_equals_plain_rollout'] and not v['fell']
    for n, m_ in v['files'].items():
        assert hashlib.sha256((T_/'media'/n).read_bytes()).hexdigest() == m_['sha256'], n
for t in ('모델 개정 2: 접지각 클러치 학습', '[F16]', 'assets/rev2-compare.png'):
    assert t in s, t
checks['touchdown_clutch_run'] = dict(accept_rows=len(cacc), falls=0, knee_heat={c: RC['accept'][f'accept@{c}']['heat_knee'] for c in ('2', '4', '5.5', '7')}, heat_max={c: RC['accept'][f'accept@{c}']['heat_max'] for c in ('2', '4', '5.5', '7')},
                                      dr_falls={c: RC['dr'][f'dr@{c}']['falls'] for c in ('2', '4', '7')}, media_runs=sorted(RC['media']), training_steps=RC['train']['steps'], status='passed')

TL = json.loads((T_/'tail_results.json').read_text()); d7 = TL['dynamics']['rev2_td340@7']; p7 = TL['probe']['rev2_td340@7']['tail']
assert max(r['residual_rms'] for r in TL['dynamics'].values()) < 1e-4 and close(p7['heat'], RC['accept']['accept@7']['heat_max'], 2e-2)
assert close((p7['tau_mean']**2+p7['tau_std']**2)/(.9*p7['cont_Nm'])**2, p7['heat'], 2e-2) and p7['near_limit_frac'] == 0 and p7['target_beyond_range_frac'] == 0
assert TL['gear_heat']['1.5@7']['tail_pitch'] < 1 < TL['gear_heat']['1@7']['tail_pitch'] and all(v['falls'] == 0 for v in TL['gear_zero_shot'].values())
for t in ('모델 개정 2: 꼬리 과열의 원인', '[F17]', 'assets/rev2-tail-heat.png'):
    assert t in s, t
checks['tail_heat_cause'] = dict(tail_heat_7=p7['heat'], tau_mean=p7['tau_mean'], tau_std=p7['tau_std'], inertia=d7['inertia_about_joint'], std_terms=d7['std'], at_peak_limit_frac=d7['at_peak_limit_frac'],
                                 geared={k: v['tail_pitch'] for k, v in TL['gear_heat'].items()}, status='passed')

# --- images and PDF
images = re.findall(r'!\[.*?\]\((.*?)\)', s)
assert all((R/x).is_file() for x in images)
issues, pagemap = [], []
with pdfplumber.open(pdf) as f:
    npages = len(f.pages); texts = []
    for i, p in enumerate(f.pages, 1):
        t = p.extract_text() or ''; texts.append(t); assert len(t) > 100, i
        pagemap.append({'page': i, 'chars': len(t), 'heading': t.splitlines()[1]})
        for c in p.chars:
            if c['x0'] < 47 or c['x1'] > p.width-47 or c['top'] < 15 or c['bottom'] > p.height-18:
                issues.append([i, c['text']])
assert not issues, issues[:10]
flat = re.sub(r'\s+', '', '\n'.join(texts))
assert '|D|≤1' in flat
for n in set(re.findall(r'\d+\.\d{4,}', s)):
    assert n in flat, n
incode = False
for l in lines:
    if l.startswith('```'):
        incode = not incode; continue
    if l.startswith('|') and not incode:
        continue
    for m in re.finditer(r'\|[^|\s][^|]{0,24}\|', l):
        assert re.sub(r'\s+', '', m.group(0)) in flat, m.group(0)
reader = PdfReader(pdf); headings = len(re.findall(r'^# ', s, flags=re.M))
assert len(reader.outline) == headings, (len(reader.outline), headings)
man = json.loads((R/'run_manifest.json').read_text())
assert man['report_sha256'] == hashlib.sha256((R/'report.ko.md').read_bytes()).hexdigest()
out = dict(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), pages=npages, outline_entries=len(reader.outline), figures=len(images), source_records=len(refs),
           text_bounds_errors=issues, long_numbers_checked=len(set(re.findall(r'\d+\.\d{4,}', s))), runtime_code_modified='opt-in model_rev 2 in sim/rl/run_env.py and --model-rev in train_run.py; defaults reproduce earlier records', new_training=True, diagnostic_rollouts_of_frozen_policy=json.loads((GD/'gate_results.json').read_text())['timing']['rollouts'], checks=checks)
(R/'verification.json').write_text(json.dumps(out, ensure_ascii=False, indent=1)); (R/'page-map.json').write_text(json.dumps(pagemap, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, ensure_ascii=False)); print(json.dumps(checks, ensure_ascii=False))
