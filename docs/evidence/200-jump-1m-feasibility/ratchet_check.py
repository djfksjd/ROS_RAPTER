"""Existing passive spring: paid preload and loaded-landing sensitivity, same geometry/limits."""
import hashlib
from study import *
from followup_model import landing_case


def charge(d,rest=1.):
    q=np.linspace(rest,2.6,8001);h=hip_angle(q,d)
    n=np.full(len(q),d.knee_ratio) if d.knee_ratio else fourbar(q)[0]
    lever=-REV['leg']['tibia']*d.length_scale*np.sin(h+q)
    k=500*d.mass/11.36
    # Quasi-static slow flexion; gravity helps compress, Coulomb friction opposes it.
    need=d.mass*G/2*lever+k*np.maximum(q-rest,0)+RZ['friction']['coulomb']['knee_pitch']
    available=.9*n*MOD[d.knee]['tau_peak']
    bad=np.flatnonzero(need>available);i=int(bad[0])
    last=i-1;charge_motor_work=2*np.trapezoid(need[:last+1],q[:last+1])
    dh=np.gradient(h,q)
    hip_need=-d.mass*G/2*d.offset+RZ['friction']['coulomb']['hip_pitch']*np.sign(dh)
    hip_work=2*np.trapezoid((hip_need*dh)[:last+1],q[:last+1])
    return dict(q_limit_rad=float(q[last]),first_violating_q_rad=float(q[i]),
        required_joint_nm=float(need[i]),available_joint_nm=float(available[i]),
        stored_total_j=float(k*(q[last]-rest)**2),stored_per_leg_j=float(.5*k*(q[last]-rest)**2),
        gravity_help_j=float(d.mass*G*(height(rest,d)-height(q[last],d))),
        knee_motor_work_both_j=float(charge_motor_work),motor_work_total_j=float(charge_motor_work+hip_work),stiffness_per_leg_nm_rad=k,rest_rad=rest)


def main():
    original=json.loads((E/'eval/requirements.json').read_text());d=Design(mass=original['metadata']['mass_kg'])
    ch=charge(d);rows=[]
    # Path selection limited to the quasistatically loadable crouch. No claim this is a global dynamic bound.
    best=None
    for q0 in np.linspace(1.02,ch['q_limit_rad'],9):
        for q1 in (.65,.8,1.):
            if q0<=q1:continue
            p=path(d,float(q0),q1)
            for H in np.arange(.06,.601,.005):
                if H<=p['z'][-1]-p['standing_z']:continue
                r=evaluate(d,p,float(H),spring_rest=1.)
                if r['feasible'] and (best is None or H>best['target_m']):best=r
    qload=ch['q_limit_rad'];r=evaluate(d,path(d,qload,.8),1.,spring_rest=1.,store=True)
    deep=evaluate(d,path(d),1.,spring_rest=1.,store=True)
    combo=Design(name='1m_leg_13kg_belts',mass=13.,length_scale=1.25,hip_ratio=.4,knee_ratio=.7)
    land=landing_case(combo,3.,1.)
    cp=path(combo,land['q_start'],land['q_takeoff'])
    loaded_land=evaluate(combo,cp,land['target_m'],landing=True,spring_rest=1.,store=True)
    combo_launch=evaluate(combo,path(combo,2.6,1.),1.,spring_rest=1.,store=True)
    args=ROOT/'sim/rl/runs/l80_r1_6M/args.json'
    dump('ratchet.json',dict(args_sha256=hashlib.sha256(args.read_bytes()).hexdigest(),
        config=json.loads(args.read_text())['springs_override'],charge=ch,
        frozen_loaded_1m=r,frozen_deep_preload_1m=deep,best_quasistatic_preloaded_family=best,
        combo_charge=charge(combo),combo_loaded_launch=combo_launch,combo_loaded_landing=loaded_land,
        qualification='static preload bound only; dynamic counter-movement/unloading can differ; all CEM retained actual ratchet'))
    print('charge',ch,'loaded family height',best['target_m'] if best else None)
    for label,x in [('loaded 1m',r),('deep hypothetical prepaid',deep),('combo loaded launch',combo_launch),('combo loaded landing',loaded_land)]:
        print(label,'feasible',x['feasible'],'springJ',x['spring_energy_start_j'],x['spring_energy_end_j'],'knee',x['joints'][1])

if __name__=='__main__':main()
