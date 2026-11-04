"""Same-model detailed comparison and spring/landing sizing; not a new criterion."""
import json
import numpy as np
from study import *


def landing_case(d,bw,q1):
    s=1/(bw-1);z1=height(q1,d);z0=z1-s;grid=np.linspace(q1,2.6,3000)
    if z0<height(2.6,d):return None
    q0=float(np.interp(z0,height(grid,d)[::-1],grid[::-1]))
    return evaluate(d,path(d,q0,q1),1+z1-height(1.,d),landing=True,store=True)


def main():
    original=json.loads((E/'eval/requirements.json').read_text())
    d=Design(mass=original['metadata']['mass_kg'])
    combo=Design(name='1m_leg_13kg_belts',mass=13.,length_scale=1.25,hip_ratio=.4,knee_ratio=.7)
    launch=evaluate(combo,path(combo,2.6,1.),1.,store=True)
    land=landing_case(combo,3.,1.)
    assert launch['feasible'] and land['feasible']
    cases=[]
    sweep=json.loads((E/'eval/sweep.json').read_text())
    for row in sweep['candidates']:
        if row['label'].startswith(('offset','frozen_mass','belts_mass')):continue
        des=Design(**row['design'])
        attempts=[landing_case(des,3.,q) for q in (.65,.8,1.,1.2)]
        cases.append(dict(name=des.name,landing_1m_3bw=[{k:v for k,v in r.items() if k!='samples'} for r in attempts if r]))
    motor_combos=[]
    for row in sweep['actuator_screen']:
        template=Design(**row['winning_design']);lo,hi=4.,16.
        for _ in range(16):
            mid=(lo+hi)/2;trial=replace(template,mass=mid)
            if any(evaluate(trial,path(trial,a,b),1.)['feasible'] for a,b in PATHS):lo=mid
            else:hi=mid
        motor_combos.append(dict(module=row['module'],max_mass_1m_kg=lo,design=asdict(replace(template,mass=lo))))
    # Ideal linear catapult, motors mechanically disconnected during release. Energy is never injected for free.
    sp=json.loads((E/'eval/spring.json').read_text())
    m=sp['total_mass_kg'];s=sp['assumptions']['stroke_m'];eta=sp['assumptions']['release_efficiency']
    k=sp['stiffness_total_n_m'];x=np.linspace(0,s,1001)
    f=eta*k*(s-x);work=eta*k*(s*x-x*x/2)-m*G*x
    v=np.sqrt(np.maximum(2*work/m,0.))
    # Integrable sqrt singularity at t=0; analytic first interval of constant initial acceleration.
    dt0=np.sqrt(2*x[1]/(f[0]/m-G));duration=dt0+np.trapezoid(1/np.maximum(v[1:],1e-10),x[1:])
    sp.update(energy_check_apex_m=float(v[-1]**2/(2*G)),release_duration_s=float(duration),
       delivered_peak_force_n=float(f[0]),delivered_peak_bw=float(f[0]/(m*G)),
       release_peak_power_w=float((f*v).max()),motor_charge_dc_bus_a_assumed_80pct=200/.8/48,
       landing_3bw_absorb_j=float(m*G*1.5),spring_release_j=float(eta*sp['stored_j']))
    # Equivalent knee spring at an assumed 0.20 m cam moment arm, one spring per leg.
    arm=.2
    sp.update(cam_arm_assumed_m=arm,knee_preload_torque_per_leg_nm=sp['preload_force_per_leg_n']*arm,
       knee_stiffness_nm_rad=sp['stiffness_per_leg_n_m']*arm**2,
       proposed_charge_ratio=3.,charge_module_torque_nm=sp['preload_force_per_leg_n']*arm/(3*.9),
       charge_dc_current_a=sp['preload_force_per_leg_n']*arm/(3*.9*MOD['RS04']['kt_out']))
    dump('spring.json',sp)
    for name in ('RS03','RS04','RS06'):
        assert MOD[name]['tn_curve']
    survey=[]
    for name,m in MOD.items():
        c=m.get('tn_curve');survey.append(dict(name=name,verified=m['verified'],mass_kg=m['mass'],
            peak_nm=m['tau_peak'],w0_rad_s=m['w0'],phase_peak_a=m['i_peak'],
            tn_zero_speed_rad_s=c['rpm'][-1]*np.pi/30 if c else None,
            curve_knot_power_w=max(np.array(c['rpm'])*np.pi/30*np.array(c['nm'])) if c else None,
            dc_limit_a=m['tau_peak']/m['kt_out'] if m.get('kt_out') else None,
            used=bool(c and m.get('verified'))))
    dump('details.json',dict(combo_launch=launch,combo_landing=land,landing_other_designs=cases,motor_survey=survey,motor_mass_combinations=motor_combos,
        frozen_cem_range_max=max_height(d,[(a,b) for a,b in PATHS if a<=2.]),
        planar_path_assumption='constant base/CoM offset, no distributed leg/body angular dynamics'))
    # Every actual probe peak is checked against early termination, not called a true apex without descent.
    pr=json.loads((E/'eval/probe_summary.json').read_text())
    result=[]
    for row in pr['variants']:
        b=row['best']
        with np.load(E/'eval'/f"{row['variant']}_best.npz") as a:
            z=a['xyz'][:,2];i=int(np.argmax(z));com=a['com'][:,2];ic=int(np.argmax(com))
            result.append(dict(variant=row['variant'],base_max_sample_at_end=i==len(z)-1,
               base_vertical_speed_at_max=float(a['velocity'][i,2]),com_max_sample_at_end=ic==len(com)-1,
               final_com_vertical_speed_fd=float((com[-1]-com[-2])/.001),
               full_apex_observed=bool(i<len(z)-1 and ic<len(com)-1),
               best_height=b['height'],com_rise_m=b.get('com_rise_m'),fall_s=b['result']['t_fall']))
    dump('probe_peak_audit.json',result)
    print('combo',launch['takeoff_speed_mps'],launch['point_mass_work_total_j'],launch['joint_power_total_peak_w'])
    print('landing',land['joints'],land['dc_bus_ideal_regen_peak_a'])
    print('spring',sp)
    print('peak_audit',result)

if __name__=='__main__':main()
