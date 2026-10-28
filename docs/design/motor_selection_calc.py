"""Motor selection arithmetic for docs/design/MOTOR_SELECTION.ko.md (2026-10-28). Standard library only, no simulation.

Writes docs/design/motor_candidates.json and prints the tables of the report:
    python3 docs/design/motor_selection_calc.py

Inputs are copied here with their source so the arithmetic can be re-checked by hand:
  REQ8   - 8-axis candidate (model_rev 2, rev2_r3_ck7M): motor RMS from the per-motor load^2 (heat_all) in
           docs/evidence/164-rev2-spring-clutch/rev2/train/r3_ck7M_spring_off_8.jsonl (label as_trained, max of 8
           runs per command); simulated actuator envelope from sim/rl/run_env.py (MOTOR_CLASSES, ARCH8, LINK_ETA,
           rev2_r3 override hip_pitch/tail_pitch B 1.5:1); SLIP budget from docs/design/full_budget_speed_scan.json.
  MODULES - datasheet values. verified=True only where an official document was read (RobStride manuals on GitHub).
Everything marked 'assumed' is an assumption, not a measurement.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent/'motor_candidates.json'
ETA = .9            # external stage / linkage efficiency (run_env.LINK_ETA); 1.0 is used for a module at 1:1 without a stage
BELT_KG = .10       # assumed mass of one belt / cable reduction stage (no design exists)
SIM_CLASS = {'A': (22., 40., 500.), 'B': (48., 33., 900.)}  # run_env.MOTOR_CLASSES (peak N·m, no-load rad/s, W)
SIM_CONT = {'A': 8.8, 'B': 19.2}                              # 0.4 x peak, run_env heat model
SIM_MASS = {'A': .5, 'B': .96}                                # run_env.MOTOR_KG
ROTOR_J_OUT_SIM = 81*1.002e-4                                 # run_env.ROTOR_J_OUT (AK10-9 V3 rotor x 9^2)
KNEE_N2_MEAN = 1.84   # mean N^2 of the four-bar over KNEE_BAND (computed from knee_linkage_study.fourbar_N)
KNEE_N_STANCE = 1.5   # four-bar ratio used as nominal (run_env uses 1.5 for the spec; stance knee 0.95-1.25 rad -> N 1.4-1.7)

# rev2_r3_ck7M load^2 = (motor torque / sim continuous torque)^2, max of 8 runs, commands 2/4/5.5/7 m/s
R3_SPEED = [1.93, 4.23, 5.13, 5.72]
R3_HEAT = {'hip_roll': [.424, .523, .555, .579], 'hip_pitch': [.081, .244, .380, .495], 'knee': [.299, .505, .690, .890],
           'tail_yaw': [.036, .099, .132, .159], 'tail_pitch': [.127, .132, .169, .234]}
# simulated drive per joint in rev2_r3: (class, external ratio)
R3_DRIVE = {'hip_roll': ('A', 1.7), 'hip_pitch': ('B', 1.5), 'knee': ('B', 'fourbar'), 'tail_yaw': ('B', 1.), 'tail_pitch': ('B', 1.5)}
# SLIP design-stride budget at 11.36 kg, spring (c) (full_budget_speed_scan.json): v -> (peak N·m, peak rad/s, peak W, RMS N·m)
SLIP = {'hip_pitch': {4: (17.6, 7.7, 96, 7.1), 6: (29.2, 11.7, 210, 11.5), 8: (52.3, 15.2, 340, 20.6), 11.1: (60.2, 20.6, 614, 27.5)},
        'knee': {4: (25.4, 10.5, 134, 13.1), 6: (37.9, 14.7, 227, 19.8), 8: (34.2, 17.5, 329, 20.4), 11.1: (49.2, 21.1, 538, 30.4)},  # knee+ankle coupled
        'knee_indep': {4: (17.7, 10.5, 85, 9.1), 6: (24.3, 14.7, 161, 12.9), 8: (34.2, 17.5, 240, 18.9), 11.1: (49.2, 21.1, 394, 29.0)},
        'ankle_pitch': {4: (14.1, 10.5, 52, 6.8), 6: (15.8, 14.7, 70, 8.0), 8: (12.0, 17.5, 94, 4.2), 11.1: (13.2, 21.1, 154, 4.5)},
        'hip_roll': {4: (31.1, 5., 155.5, 14.0), 6: (32.2, 5., 161, 14.5), 8: (31.8, 5., 159, 14.3), 11.1: (36.0, 5., 180, 16.2)},
        'ankle_roll': {4: (6.2, 5., 31.1, 2.8), 6: (6.4, 5., 32.2, 2.9), 8: (6.4, 5., 31.8, 2.9), 11.1: (7.2, 5., 36, 3.2)},
        'tail_pitch': {v: (36., 12., 400, 10.8) for v in (4, 6, 8, 11.1)}, 'tail_yaw': {v: (36., 12., 400, 10.8) for v in (4, 6, 8, 11.1)}}
# peaks recorded in rev2 rollouts (joint side). None = not recorded.
MEAS_PEAK = {'knee': '63-68 N·m (rev2_a8_h2) / 72-78 N·m (rev2_n10) joint, spring_id_rev2.json / n10_spring_id.json req_peak',
             'tail_pitch': '41.4-43.2 N·m joint at 1:1 (43.2 = sim limit, saturating), tail_probe.json tau_absmax; |qd| 3.1-5.2 rad/s, tail_dynamics.json',
             'hip_pitch': None, 'hip_roll': None, 'tail_yaw': None}

SRC_RS = 'https://github.com/RobStride/Product_Information (commit 384cc56, 2026-10-16)'
# module datasheet values. tau: N·m at module output; w0: rad/s no-load at v_bus; j_out: kg m^2 at module output;
# r_line: ohm line-to-line; l: H (phase/line not stated unless noted); kt_out: N·m/A at module output; i: A.
MODULES = {
    'RS02': dict(vendor='RobStride', ratio=7.75, tau_peak=17., tau_cont=6., w0=410*2*math.pi/60, v_bus=48., kt_out=1.22,
                 r_line=.55, l=486e-6, j_out=4.2e-3, mass=.380, i_peak=23., i_cont=7., t_wind=145., price='CNY 699',
                 comm='CAN 2.0 1 Mbps / CAN FD variant', enc='14 bit', verified=True,
                 url=SRC_RS+' RS02User Manual260713.pdf; RS brochure 2026.010.09',
                 note='rated 6 N·m in manual, 7 N·m in brochure; mass 380/405/418 g across documents; I in A peak (Apk), kt per A rms'),
    'RS03': dict(vendor='RobStride', ratio=9., tau_peak=60., tau_cont=20., w0=195*2*math.pi/60, v_bus=48., kt_out=2.36,
                 r_line=.39, l=.275e-3, j_out=.02, mass=.880, i_peak=43., i_cont=13., t_wind=145., price='CNY 999',
                 comm='CAN 2.0 1 Mbps / CAN FD variant', enc='14 bit', verified=True,
                 url=SRC_RS+' RS03User Manual261013.pdf; RS brochure 2026.010.09',
                 note='rated 20 N·m at 100 rpm with a 215x220 mm heat sink; no-load 195 (brochure) / 200 (manual) rpm, 195 used; mass 880-928 g; winding protection 145 C (manual: actual limit 180 C)'),
    'RS04': dict(vendor='RobStride', ratio=9., tau_peak=120., tau_cont=40., w0=200*2*math.pi/60, v_bus=48., kt_out=2.1,
                 r_line=.16, l=.211e-3, j_out=.04, mass=1.420, i_peak=90., i_cont=27., t_wind=145., price='CNY 1199',
                 comm='CAN 2.0 1 Mbps / CAN FD variant', enc='14 bit', verified=True,
                 url=SRC_RS+' RS04User Manual260713.pdf; RS brochure 2026.010.09',
                 note='rated 40 N·m with a 345x345 mm heat sink (35 N·m with 220x200 mm); mass 1420-1510 g'),
    'RS06': dict(vendor='RobStride', ratio=9., tau_peak=36., tau_cont=11., w0=480*2*math.pi/60, v_bus=48., kt_out=1.09,
                 r_line=.23, l=.165e-3, j_out=.012, mass=.621, i_peak=57., i_cont=14.3, t_wind=145., price='CNY 849',
                 comm='CAN 2.0 1 Mbps / CAN FD variant', enc='14 bit', verified=True,
                 url=SRC_RS+' RS06User Manual260713.pdf; RS brochure 2026.010.09',
                 note='rated 11 N·m at 100 rpm with a 130x160 mm heat sink; kt 1.09 (manual) / 1.1 (brochure)'),
    'AK10-9 V3.0 KV60': dict(vendor='CubeMars', ratio=9., tau_peak=53., tau_cont=18., w0=320*2*math.pi/60, v_bus=48., kt_out=.16*9,
                 r_line=.248, l=2.13e-4, j_out=1002e-7*81, mass=.940, i_peak=31.9, i_cont=10.7, t_wind=None, price='USD 798.90',
                 comm='CAN (not confirmed)', enc='21 bit inner (search summary)', verified=False,
                 url='https://www.cubemars.com/product/ak10-9-v3-0-kv60-robotic-actuator.html (page not opened: search summary only)',
                 note='kt 0.16 N·m/A motor side x 9 (derived); rotor 1002 g cm^2 assumed motor side (x81 = output); the sim class B (48 N·m) is the older AK10-9 rating'),
    'AK80-9 V3.0 KV100': dict(vendor='CubeMars', ratio=9., tau_peak=22., tau_cont=9., w0=None, v_bus=48., kt_out=.095*9,
                 r_line=.160, l=1.16e-4, j_out=None, mass=.490, i_peak=28., i_cont=12., t_wind=None, price='USD 599.90',
                 comm=None, enc=None, verified=False,
                 url='https://www.robotshop.com/products/cubemars-ak80-9-v30-kv100-ak-series-robotic-actuator (search summary only)',
                 note='no-load speed not found (rated 390 rpm); rotor inertia 579 vs 1118 g cm^2 conflict'),
    'GO-M8010-6': dict(vendor='Unitree', ratio=6.33, tau_peak=23.7, tau_cont=None, w0=30., v_bus=24., kt_out=None,
                 r_line=None, l=None, j_out=None, mass=.530, i_peak=40., i_cont=None, t_wind=None, price='USD 369 (search summary)',
                 comm='RS-485 4 Mbps', enc='15 bit rotor', verified=False,
                 url='https://shop.unitree.com/products/go1-motor (search summary only)',
                 note='rated torque not published -> thermal check impossible'),
    'DM-J8009-2EC': dict(vendor='Damiao', ratio=9., tau_peak=40., tau_cont=20., w0=320*2*math.pi/60, v_bus=48., kt_out=None,
                 r_line=.09, l=61e-6, j_out=None, mass=.896, i_peak=50., i_cont=20., t_wind=None, price='USD 470 (retailer)',
                 comm='CAN 1 Mbps', enc='dual, bits not stated', verified=False,
                 url='https://aifitlab.com/products/damiao-dm-j8009-2ec-servo-motor (search summary only)',
                 note='rated voltage 24 V; 320 rpm is the 48 V figure'),
    'DM-J10010L-2EC': dict(vendor='Damiao', ratio=10., tau_peak=120., tau_cont=40., w0=200*2*math.pi/60, v_bus=48., kt_out=None,
                 r_line=.11, l=85e-6, j_out=None, mass=1.372, i_peak=None, i_cont=None, t_wind=None, price='USD 225-230 (retailer)',
                 comm='CAN / UART', enc='dual, bits not stated', verified=False,
                 url='https://aifitlab.com/products/damiao-dm-j10010l-2ec-servo-motor (search summary only)', note=''),
    'RMD-X8-P20-120': dict(vendor='MyActuator', ratio=19.612, tau_peak=120., tau_cont=43., w0=158*2*math.pi/60, v_bus=48., kt_out=2.4,
                 r_line=None, l=.31e-3, j_out=None, mass=1.400, i_peak=43.8, i_cont=17.6, t_wind=None, price='USD 645-921 (retailers)',
                 comm='CAN / EtherCAT', enc='17 bit', verified=False,
                 url='https://wiki.seeedstudio.com/myactuator_series/ (search summary only)', note='R 0.18 ohm, phase/line not stated'),
}

# combinations: (joint, module, external ratio). knee: ratio = belt in front of the four-bar (1 = motor on the crank).
COMBOS = [('hip_pitch', 'RS03', 1.), ('hip_pitch', 'RS03', 1.5), ('hip_pitch', 'RS04', 1.), ('hip_pitch', 'AK10-9 V3.0 KV60', 1.5),
          ('hip_pitch', 'DM-J10010L-2EC', 1.), ('hip_pitch', 'RMD-X8-P20-120', 1.),
          ('hip_roll', 'RS06', 1.), ('hip_roll', 'RS06', 1.5), ('hip_roll', 'RS03', 1.), ('hip_roll', 'AK80-9 V3.0 KV100', 1.7),
          ('hip_roll', 'GO-M8010-6', 1.7),
          ('knee', 'RS03', 1.), ('knee', 'RS04', 1.), ('knee', 'AK10-9 V3.0 KV60', 1.), ('knee', 'DM-J8009-2EC', 1.), ('knee', 'RS06', 2.),
          ('knee', 'DM-J10010L-2EC', 1.),
          ('tail_pitch', 'RS03', 1.), ('tail_pitch', 'RS06', 2.), ('tail_pitch', 'AK10-9 V3.0 KV60', 1.5),
          ('tail_yaw', 'RS06', 1.), ('tail_yaw', 'RS06', 1.5), ('tail_yaw', 'RS03', 1.),
          # 10/12-axis variants (independent knee and ankle pitch, spec budget only)
          ('knee_indep', 'RS03', 1.), ('ankle_pitch', 'RS06', 1.), ('ankle_pitch', 'RS02', 2.), ('ankle_roll', 'RS02', 1.), ('ankle_roll', 'GO-M8010-6', 1.)]


def avail(tau, w0, w):
    """Assumed envelope of run_env.motor_available: full torque to w0/2, linear to zero at w0 (power cap ignored)."""
    a = abs(w)
    return tau if a <= .5*w0 else max(0., tau*(w0-a)/(.5*w0))


def interp(xs, ys, x):
    """Piecewise linear, extrapolated with the end segments."""
    i = 1 if x < xs[1] else len(xs)-1 if x >= xs[-2] else next(k for k in range(1, len(xs)) if x < xs[k])
    return ys[i-1] + (ys[i]-ys[i-1])*(x-xs[i-1])/(xs[i]-xs[i-1])


def solve_v(f, lo=2., hi=11.1, target=1.):
    """Largest v in [lo, hi] (0.05 m/s grid) with f(v) <= target, or None."""
    best, v = None, lo
    while v <= hi + 1e-9:
        if f(v) <= target:
            best = round(v, 2)
        else:
            break
        v += .05
    return best


PROPOSED = {('hip_pitch', 'RS03', 1.), ('hip_roll', 'RS06', 1.5), ('knee', 'RS03', 1.), ('tail_pitch', 'RS03', 1.), ('tail_yaw', 'RS06', 1.)}


def combo(joint, mod, g):
    m = MODULES[mod]
    knee = joint == 'knee'
    eta = ETA if (g != 1 or knee) else 1.  # no external stage at 1:1 (the module rating already includes its own gear)
    # joint-side envelope (knee: crank side of the four-bar, i.e. motor x belt x linkage efficiency)
    tp, tc = m['tau_peak']*g*eta, (m['tau_cont']*g*eta if m['tau_cont'] else None)
    w0 = m['w0']/g if m['w0'] else None
    jr = m['j_out']*g*g if m['j_out'] else None
    row = dict(name=f'{mod} x {g:g}' + (' + four-bar' if knee else ''), joint=joint, module=mod, vendor=m['vendor'],
               proposed_8axis=(joint, mod, g) in PROPOSED,
               gear_ratio=g, gear_ratio_note=('belt before the knee four-bar (N 1.0-2.0); outputs are at the four-bar crank' if knee else
                                              'external stage after the module; outputs at the joint'),
               module_internal_ratio=m['ratio'], gear_efficiency=eta,
               tau_peak_out_Nm=round(tp, 2), tau_cont_out_Nm=(round(tc, 2) if tc else None),
               omega_noload_out_rad_s=(round(w0, 2) if w0 else None),
               kt_out_Nm_per_Arms=(round(m['kt_out']*g*eta, 3) if m['kt_out'] else None),
               r_line_ohm=m['r_line'], r_phase_ohm_assumed_Y=(round(m['r_line']/2, 4) if m['r_line'] else None),
               l_phase_H=m['l'], l_note='source does not state phase or line-to-line',
               j_rotor_reflected_out_kgm2=(round(jr, 5) if jr else None),
               mass_kg=round(m['mass'] + (BELT_KG if g != 1 else 0.), 3), mass_note=('includes assumed belt stage 0.10 kg' if g != 1 else 'module only'),
               v_bus_V=m['v_bus'], i_peak_A=m['i_peak'], i_cont_A=m['i_cont'], current_note='RobStride: phase current in A peak',
               t_winding_max_C=m['t_wind'], r_th_wh_K_per_W=None, r_th_ha_K_per_W=None, c_th_w_J_per_K=None, c_th_h_J_per_K=None,
               thermal_note='no thermal resistance / capacitance published by any source found (2026-10-28)',
               price=m['price'] if g == 1 else m['price'] + ' + belt stage (not priced)', comm=m['comm'], encoder=m['enc'],
               source_url=m['url'], accessed='2026-10-28', verified=m['verified'], source_note=m['note'])
    # --- check against the 8-axis simulated drive (rev2_r3) ---
    if joint in R3_DRIVE:
        cls, gs = R3_DRIVE[joint]
        gs_num = 1. if gs == 'fourbar' else gs
        sim_peak = SIM_CLASS[cls][0]*gs_num*ETA
        sim_w0 = SIM_CLASS[cls][1]/gs_num
        motor_rms = [math.sqrt(h)*SIM_CONT[cls] for h in R3_HEAT[joint]]        # sim motor side
        joint_rms = [r*gs_num*ETA for r in motor_rms]                            # sim joint side (knee: crank side)
        row['sim_drive'] = f'class {cls} ({SIM_CLASS[cls][0]:g} N·m, {SIM_CLASS[cls][1]:g} rad/s) x {gs}'
        row['req_rms_Nm_at_5p72mps'] = round(joint_rms[-1], 2)
        row['peak_ratio_vs_sim'] = round(tp/sim_peak, 2)
        row['w0_ratio_vs_sim'] = round(w0/sim_w0, 2) if w0 else None
        if tc:
            heat = [(r/tc)**2 for r in joint_rms]
            row['load2_at_5p72mps'] = round(heat[-1], 2)
            row['v_heat_1p0_mps_est'] = solve_v(lambda v: interp(R3_SPEED, heat, v), 1.93, 11.1, 1.)
            row['v_heat_0p8_mps_est'] = solve_v(lambda v: interp(R3_SPEED, heat, v), 1.93, 11.1, .8)
        sim_j = ROTOR_J_OUT_SIM*(KNEE_N2_MEAN if gs == 'fourbar' else gs_num**2)
        if jr:
            row['j_reflected_ratio_vs_sim'] = round(jr*(KNEE_N2_MEAN if knee else 1.)/sim_j, 2)
        row['mass_delta_vs_model_kg'] = round(row['mass_kg'] - SIM_MASS[cls], 3)
    # --- check against the SLIP budget (11.36 kg): torque available at the required speed ---
    b = SLIP[joint]
    n = KNEE_N_STANCE if joint == 'knee' else 1.
    def slip_util(v):
        vs = sorted(b)
        tq = interp(vs, [b[x][0] for x in vs], v)/n
        w = interp(vs, [b[x][1] for x in vs], v)*n
        a = avail(tp, w0, w) if w0 else 0.
        return tq/a if a > 0 else math.inf
    def slip_heat(v):
        vs = sorted(b)
        r = interp(vs, [b[x][3] for x in vs], v)/n
        return (r/tc)**2 if tc else math.inf
    u6 = slip_util(6.) if w0 else None
    row['slip_peak_util_at_6mps'] = (round(u6, 2) if u6 is not None and math.isfinite(u6) else ('speed above no-load' if w0 else None))
    row['slip_v_max_peak_mps_est'] = solve_v(slip_util) if w0 else None
    row['slip_v_max_rms_mps_est'] = solve_v(slip_heat) if tc else None
    return row


def main():
    rows = [combo(*c) for c in COMBOS]
    units = {'tau_*_out_Nm': 'N·m at the joint (knee: at the four-bar crank)', 'omega_noload_out_rad_s': 'rad/s at v_bus_V, joint side',
             'j_rotor_reflected_out_kgm2': 'module output-side equivalent inertia x external ratio^2 (kg m^2)',
             'load2': '(RMS torque / continuous torque)^2, 1.0 = rated', 'v_*_mps_est': 'm/s, estimate (extrapolated)',
             'peak_ratio_vs_sim': 'combo peak / simulated actuator peak at the same joint (>= 1 reproduces the sim limit)'}
    doc = dict(generated_by='docs/design/motor_selection_calc.py', date='2026-10-28',
               assumptions=dict(gear_efficiency=ETA, belt_stage_kg=BELT_KG, knee_fourbar_N_nominal=KNEE_N_STANCE,
                                torque_speed_shape='full torque to half no-load speed, linear to zero (run_env assumption, not datasheet)',
                                rms_source='rev2_r3_ck7M 8-run max load^2 (sim continuous = 0.4 x class peak)',
                                rated_torque_caveat='RobStride rated torque is measured with a specified heat sink; mounted in a leg it can be lower'),
               units=units, modules={k: {kk: vv for kk, vv in v.items()} for k, v in MODULES.items()}, candidates=rows)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    keys = ['name', 'joint', 'tau_peak_out_Nm', 'tau_cont_out_Nm', 'omega_noload_out_rad_s', 'peak_ratio_vs_sim', 'w0_ratio_vs_sim',
            'load2_at_5p72mps', 'v_heat_1p0_mps_est', 'j_reflected_ratio_vs_sim', 'mass_kg', 'mass_delta_vs_model_kg',
            'slip_peak_util_at_6mps', 'slip_v_max_peak_mps_est', 'slip_v_max_rms_mps_est', 'verified']
    print('| ' + ' | '.join(keys) + ' |')
    for r in rows:
        print('| ' + ' | '.join(str(r.get(k, '')) for k in keys) + ' |')


if __name__ == '__main__':
    main()
