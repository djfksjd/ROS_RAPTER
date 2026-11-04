"""Evidence-only planar jump requirements. No policy, environment or frozen file edits.
Heights are relative to the standing pose. Feasible means this path family only.
"""
import hashlib
import json
import sys
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'modeling'), str(ROOT/'sim/rl')]
import knee_linkage_study as K
from realism import current_window

MOD = json.loads((ROOT/'docs/design/motor_candidates.json').read_text())['modules']
REV = json.loads((ROOT/'sim/rl/rev3_l80.json').read_text())
RZ = json.loads((ROOT/'sim/rl/realism_robstride_l80.json').read_text())
G = 9.81
INPUTS = ['sim/rl/rev3_l80.json', 'sim/rl/realism_robstride_l80.json',
          'docs/design/motor_candidates.json', 'sim/rl/run_env.py', 'sim/rl/realism.py',
          'docs/design/freeze_contract.json', 'docs/DESIGN_FREEZE.ko.md']


def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}


def dump(name, value):
    (E/'eval'/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@dataclass(frozen=True)
class Design:
    name: str = 'frozen'
    mass: float = 14.9224
    length_scale: float = 1.
    offset: float = .05  # ASSUMPTION: hip versus vertical GRF line; do not set hip torque to zero silently
    hip: str = 'RS04'
    knee: str = 'RS04'
    hip_ratio: float = .7
    knee_ratio: float = 0.  # 0 means original four-bar


def height(q, d):
    a=(REV['leg']['femur']+REV['leg']['meta'])*d.length_scale
    b=REV['leg']['tibia']*d.length_scale
    return np.sqrt(a*a+b*b+2*a*b*np.cos(q)-d.offset**2)


def hip_angle(q,d):
    a=(REV['leg']['femur']+REV['leg']['meta'])*d.length_scale
    b=REV['leg']['tibia']*d.length_scale
    z=height(q,d)
    return np.arctan2(d.offset,z)-np.arctan2(b*np.sin(q),a+b*np.cos(q))


@lru_cache(maxsize=1)
def fourbar_table():
    from run_env import fourbar_band
    K.FEMUR=REV['leg']['femur']
    band=fourbar_band(REV['fourbar'],hi=REV['band_search_hi'])
    grid=np.linspace(.1,max(2.7,band[1]+.05),300)
    f=REV['fourbar']
    _,n,_,ok=K.fourbar_N(grid,f['a'],f['b'],f['c'],np.deg2rad(f['beta_deg']),f['branch'])
    n=np.clip(np.where(ok,np.abs(n),1.),.2,5.)
    mask=(grid>=band[0])&(grid<=band[1])
    return grid,n,float(np.mean(n[mask]**2))


def fourbar(q):
    grid,n,n2=fourbar_table()
    return np.interp(q,grid,n),n2


def path(d, q0=2.6, q1=.8, n=241):
    assert q0>q1
    q=np.linspace(q0,q1,2001); z=height(q,d); h=hip_angle(q,d)
    zs=np.linspace(z[0],z[-1],n)
    qs=CubicSpline(z,q); hs=CubicSpline(z,h)
    angles=np.array([hs(zs),qs(zs)])
    first=np.array([hs(zs,1),qs(zs,1)]);second=np.array([hs(zs,2),qs(zs,2)])
    ratio_k,n2=fourbar(angles[1])
    if d.knee_ratio: ratio_k=np.full(n,d.knee_ratio);n2=d.knee_ratio**2
    ratios=np.array([np.full(n,d.hip_ratio),ratio_k])
    # Force on hip-pitch and knee-minus-ankle coordinates. Both motors remain explicit.
    lever=np.array([np.full(n,-d.offset),-REV['leg']['tibia']*d.length_scale*np.sin(angles.sum(axis=0))])
    return dict(z=zs,q=angles,dq_dz=first,d2q_dz2=second,ratio=ratios,n2=np.array([d.hip_ratio**2,n2]),
                lever=lever,stroke=float(z[-1]-z[0]),standing_z=float(height(1.,d)),q0=q0,q1=q1)


def limits(mod, speed, torque):
    m=MOD[mod];c=m['tn_curve'];w=np.asarray(speed)
    tn=np.interp(np.abs(w),np.array(c['rpm'])*np.pi/30,c['nm'],right=0.)
    i_limit=m['tau_peak']/m['kt_out']  # exact DC equivalent used by realism.survey_module_spec
    il,ih,forced=current_window(m['kt_out']*w,48.,m['r_line'],i_limit)
    lo=il*m['kt_out'];hi=ih*m['kt_out']
    hi=np.where(w>=0,np.minimum(hi,tn),hi)
    lo=np.where(w<0,np.maximum(lo,-tn),lo)
    ok=(torque>=lo-1e-8)&(torque<=hi+1e-8)&~forced
    motoring=torque*w>=0
    return dict(ok=ok,lo=lo,hi=hi,tn=tn,i=torque/m['kt_out'],i_limit=i_limit,
                voltage=m['kt_out']*w+torque/m['kt_out']*m['r_line'],
                tn_fail=motoring&(np.abs(torque)>tn+1e-8),
                speed_fail=motoring&(np.abs(w)>c['rpm'][-1]*np.pi/30),
                current_fail=np.abs(torque/m['kt_out'])>i_limit+1e-8,
                voltage_fail=np.abs(m['kt_out']*w+torque/m['kt_out']*m['r_line'])>48+1e-8)


def evaluate(d,p,target=1.,landing=False,force_bw=None,store=False,spring_rest=None):
    s=p['stroke'];delta=p['z'][-1]-p['standing_z'];flight_height=target-delta
    if flight_height<=0: raise ValueError('target must exceed extension rise')
    v0=np.sqrt(2*G*flight_height);acc=v0*v0/(2*s)
    # Same posture path reversed on landing; acceleration is positive while speed is negative.
    u=(p['z']-p['z'][0])/s;v=np.sqrt(2*acc*s*u)
    if landing:v=-v
    qd=p['dq_dz']*v; qdd=p['d2q_dz2']*v*v+p['dq_dz']*acc
    F=d.mass*(G+acc)
    force_tau=p['lever']*F/2
    inertia=np.array([MOD[d.hip]['j_out'],MOD[d.knee]['j_out']])*p['n2']
    inert_tau=inertia[:,None]*qdd
    # Stored frozen friction; ratio and longer-leg redesign friction is uncertain, kept unchanged deliberately.
    b=np.array([RZ['friction']['viscous'][j] for j in ('hip_pitch','knee_pitch')])
    c=np.array([RZ['friction']['coulomb'][j] for j in ('hip_pitch','knee_pitch')])
    friction=b[:,None]*qd+c[:,None]*np.sign(qd)
    tau=force_tau+inert_tau+friction
    spring_tau=np.zeros_like(tau)
    if spring_rest is not None:
        # Existing l80 ratchet: 500 scaled by actual mass / DESIGN_MASS, armed at zero deflection.
        spring_tau[1]=-500*d.mass/11.36*np.maximum(p['q'][1]-spring_rest,0.)
        tau=tau-spring_tau  # motor demand after passive assistance, including possible active braking
    w=p['ratio']*qd
    motoring=tau*qd>=0
    tm=np.where(motoring,tau/(.9*p['ratio']),tau*.9/p['ratio'])
    lim=[limits(m,w[i],tm[i]) for i,m in enumerate((d.hip,d.knee))]
    valid=all(x['ok'].all() for x in lim)
    t=v0/acc*np.sqrt(u)
    if landing:t=t[-1]-t
    def peak(x):return float(np.max(np.abs(x)))
    joint=[]
    for i,m in enumerate((d.hip,d.knee)):
        l=lim[i];bad=np.flatnonzero(~l['ok']);first=int(bad[-1] if landing else bad[0]) if len(bad) else None
        row=dict(axis=('hip_pitch','knee_pitch')[i],module=m,ratio_range=[float(p['ratio'][i].min()),float(p['ratio'][i].max())],
                 joint_speed_rad_s=peak(qd[i]),module_speed_rad_s=peak(w[i]),joint_torque_nm=peak(tau[i]),
                 force_torque_nm=peak(force_tau[i]),inertia_torque_nm=peak(inert_tau[i]),
                 module_torque_nm=peak(tm[i]),joint_power_w=peak(tau[i]*qd[i]),module_power_w=peak(tm[i]*w[i]),
                 current_dc_a=peak(l['i']),current_limit_dc_a=l['i_limit'],current_ratio=peak(l['i'])/l['i_limit'],
                 terminal_voltage_v=peak(l['voltage']),motor_curve_peak_w=float(np.max(np.array(MOD[m]['tn_curve']['rpm'])*np.pi/30*np.array(MOD[m]['tn_curve']['nm']))),
                 first_violation=None)
        if first is not None:
            row['first_violation']=dict(time_s=float(t[first]),stroke_m=float(p['z'][first]-p['z'][0]),knee_rad=float(p['q'][1,first]),
                joint_tau_nm=float(tau[i,first]),joint_speed_rad_s=float(qd[i,first]),module_speed_rad_s=float(w[i,first]),
                module_tau_nm=float(tm[i,first]),allowed_module_nm=[float(l['lo'][first]),float(l['hi'][first])],
                active=[k for k in ('tn_fail','speed_fail','current_fail','voltage_fail') if l[k][first]])
        row['fail_sample_counts']={k:int(l[k].sum()) for k in ('tn_fail','speed_fail','current_fail','voltage_fail')}
        joint.append(row)
    order=np.argsort(t)
    joint_power=2*np.sum(tau*qd,axis=0)
    module_power=2*np.sum(tm*w,axis=0)
    copper=2*sum(1.5*l['i']**2*MOD[m]['r_line'] for l,m in zip(lim,(d.hip,d.knee)))
    integrate=lambda power:float(np.trapezoid(power[order],t[order]))
    out=dict(design=asdict(d),q_start=p['q0'],q_takeoff=p['q1'],stroke_m=s,extension_rise_vs_standing_m=delta,
             target_m=target,flight_rise_m=flight_height,takeoff_speed_mps=float(v0),duration_s=float(v0/acc),
             acceleration_mps2=float(acc),force_total_n=F,force_per_leg_n=F/2,force_bw=F/(d.mass*G),
             point_mass_work_total_j=float(F*s),point_mass_work_per_leg_j=float(F*s/2),
             kinetic_energy_j=float(.5*d.mass*v0*v0),point_mass_peak_power_w=F*v0,
             feasible=bool(valid),joints=joint,landing=landing,
             spring_rest_rad=spring_rest,spring_state='released' if spring_rest is None else 'loaded ratchet',
             spring_energy_start_j=float(500*d.mass/11.36*max(p['q'][1,-1 if landing else 0]-(spring_rest or 100.),0.)**2),
             spring_energy_end_j=float(500*d.mass/11.36*max(p['q'][1,0 if landing else -1]-(spring_rest or 100.),0.)**2),
             joint_power_total_peak_w=peak(2*np.sum(tau*qd,axis=0)),
             joint_work_total_j=integrate(joint_power),joint_work_per_leg_j=integrate(joint_power)/2,
             module_work_total_j=integrate(module_power),copper_energy_j=integrate(copper),
             dc_bus_regen_with_copper_peak_a=peak(np.minimum(module_power+copper,0))/48,
             dc_bus_ideal_regen_peak_a=peak(np.minimum(2*np.sum(tm*w,axis=0),0))/48)
    if store:
        out['samples']=dict(time_s=t.tolist(),z=p['z'].tolist(),q=p['q'].tolist(),qd=qd.tolist(),qdd=qdd.tolist(),
              joint_tau=tau.tolist(),module_tau=tm.tolist(),module_speed=w.tolist(),module_cap=[l['tn'].tolist() for l in lim],
              module_current=[l['i'].tolist() for l in lim])
    return out


PATHS=[(a,b) for a in (1.4,1.8,2.,2.2,2.6) for b in (.65,.8,1.,1.2,1.4) if a>b+.1]


def max_height(d,paths=PATHS):
    best=None
    for a,b in paths:
        p=path(d,a,b)
        # Feasible prefix checked on a coarse grid; then refine its highest boundary.
        lower=max(.001,p['z'][-1]-p['standing_z']+.001)
        grid=np.linspace(lower,2.,18);valid=[h for h in grid if evaluate(d,p,h)['feasible']]
        if not valid:continue
        lo=max(valid);hi=min(2.,lo+(2.-lower)/17)
        for _ in range(14):
            mid=(lo+hi)/2
            if evaluate(d,p,mid)['feasible']:lo=mid
            else:hi=mid
        r=evaluate(d,p,lo)
        if best is None or lo>best['target_m']:best=r
    return best


def main():
    start=time.process_time();before=hashes();dump('inputs_before.json',before)
    # RunEnv supplies actual total mass, module inventory and actual joint band, read-only.
    sys.path.insert(0,str(ROOT/'docs/evidence/199-l80-jump'))
    import jump_probe as jp
    env=jp.make();env.reset(seed=0)
    mass=float(env.model.body_mass.sum())
    meta=dict(mass_kg=mass,knee_band=list(env.knee_band),nominal_angles=dict(zip(env.active,env.q0.tolist())),
              realism_enabled=env.rz.on,drive_mass_kg=sum(v.get('drive_kg',0)*(1 if k.startswith('tail') else 2) for k,v in REV['axes'].items()),module_mass_kg=sum(MOD[v['module']]['mass']*(1 if k.startswith('tail') else 2) for k,v in REV['axes'].items()))
    env.close();d=Design(mass=mass)
    required=evaluate(d,path(d),store=True)
    old=evaluate(d,path(d,2.,.8),store=True)
    dump('requirements.json',dict(metadata=meta,deep=required,current_cem_range=old,
        geometry=dict(max_geometric_stroke_m=float(height(.61627,d)-height(2.6,d)),
            current_cem_stroke_m=float(height(.62,d)-height(2.,d))),
        point_mass_15kg=dict(speed_mps=float(np.sqrt(2*G)),energy_j=15*G)))
    candidates=[]
    def add(label,x,paths=PATHS):
        r=max_height(x,paths);candidates.append(dict(label=label,design=asdict(x),best=r));return r
    add('frozen',d)
    for offset in (0.,.1):add(f'offset_{offset}',replace(d,name=f'offset_{offset}',offset=offset))
    # Finite ratio screen, same path family. Independent hip/knee choices, explicit cap of CPU.
    ratios=[]
    for gh in (.4,.7,1.):
        for gk in np.arange(.3,1.601,.1):
            x=replace(d,name='belt_screen',hip_ratio=gh,knee_ratio=round(float(gk),2))
            r=max_height(x)
            ratios.append(dict(hip_ratio=gh,knee_ratio=x.knee_ratio,height_m=r['target_m'] if r else None,best=r))
            if time.process_time()-start>900:raise RuntimeError('analytic CPU guard')
    best_ratio=max((r for r in ratios if r['best']),key=lambda r:r['height_m'])
    tuned=replace(d,name='tuned_belts',hip_ratio=best_ratio['hip_ratio'],knee_ratio=best_ratio['knee_ratio'])
    add('tuned_belts',tuned)
    actuator_screen=[]
    for mod in ('RS03','RS06'):
        # Four sagittal modules replaced. Smaller module mass reflected; structure unchanged assumption.
        x=replace(tuned,name=f'{mod}_both',hip=mod,knee=mod,mass=mass+4*(MOD[mod]['mass']-MOD['RS04']['mass']))
        options=[]
        for gh in (.4,.7,1.):
            for gk in (.3,.5,.7,.9,1.2,1.6,2.,2.5,3.):
                trial=replace(x,hip_ratio=gh,knee_ratio=gk)
                r=max_height(trial)
                if r: options.append((r['target_m'],trial,r))
        _,winner,r=max(options,key=lambda row:row[0])
        add(x.name,winner)
        actuator_screen.append(dict(module=mod,winning_design=asdict(winner),best=r))
    # Fixed frozen four-bar, and tuned belts, both with assumed total masses.
    for m in (4.,6.,8.,10.,12.):
        add(f'frozen_mass_{m}',replace(d,name=f'frozen_mass_{m}',mass=m))
        add(f'belts_mass_{m}',replace(tuned,name=f'belts_mass_{m}',mass=m))
    for scale in (1.25,1.5):
        # ASSUMED +0.8 kg for each +0.2 m leg length incl structure/linkage; motor masses fixed.
        m=mass+.8*(scale-1)/.25
        add(f'long_{scale}',replace(d,name=f'long_{scale}',length_scale=scale,mass=m))
        add(f'long_belts_{scale}',replace(tuned,name=f'long_belts_{scale}',length_scale=scale,mass=m))
    # Maximum mass in this path family at the unchanged 1 m target, with each transmission/length.
    mass_thresholds=[]
    for template in (d,tuned,replace(tuned,name='long_belts_1.25',length_scale=1.25),replace(tuned,name='long_belts_1.5',length_scale=1.5)):
        lo,hi=4.,20.
        for _ in range(16):
            mid=(lo+hi)/2;candidate=replace(template,mass=mid)
            fits=any(evaluate(candidate,path(candidate,a,b),1.)['feasible'] for a,b in PATHS)
            if fits:lo=mid
            else:hi=mid
        design=replace(template,mass=lo)
        mass_thresholds.append(dict(design=asdict(design),max_mass_1m_kg=lo,
            below_module_and_drive_mass=lo<meta['module_mass_kg']+meta['drive_mass_kg']))
    # Landing force calculations use a reversed posture trajectory with constant GRF.
    landing=[]
    for peak in (3.,4.,5.):
        # 1 m drop above touchdown, flat-force ideal is a minimum stopping distance.
        stop=1/(peak-1);z1=float(height(.8,d));z0=z1-stop
        qgrid=np.linspace(.8,2.6,2000);zg=height(qgrid,d)
        reachable=bool(z0>=zg[-1]);q0=float(np.interp(z0,zg[::-1],qgrid[::-1]))
        result=evaluate(d,path(d,q0,.8),1+(z1-height(1.,d)),landing=True,store=True) if reachable else None
        landing.append(dict(peak_bw=peak,minimum_stop_m=stop,triangular_force_stop_m=1/(peak/2-1),
                            force_peak_n=mass*G*peak,absorb_j=mass*G*(1+stop),geometry_reachable=reachable,result=result))
    dump('landing.json',landing)
    dump('sweep.json',dict(candidates=candidates,ratio_screen=ratios,best_ratio=best_ratio,actuator_screen=actuator_screen,mass_thresholds=mass_thresholds))
    # Isolated catapult provides all lift energy, motors disengaged during release.
    # Spring total compression stroke 0.30 m; 80% recovery, 2 kg mechanism mass assumed.
    spring_mass=mass+2.;compression=.30;eta=.8;rise=1.;energy=spring_mass*G*(rise+compression)/eta
    spring=dict(assumptions=dict(added_mass_kg=2.,stroke_m=compression,release_efficiency=eta,charge_mechanical_power_w=200.),
        total_mass_kg=spring_mass,stored_j=energy,stored_per_leg_j=energy/2,
        stiffness_total_n_m=2*energy/compression**2,stiffness_per_leg_n_m=energy/compression**2,
        preload_force_total_n=2*energy/compression,preload_force_per_leg_n=energy/compression,
        charge_time_s=energy/200,nominal_apex_m=rise,
        warning='energy-only sizing, needs latch/clutch and release mechanism; no full-dynamic pass')
    dump('spring.json',spring)
    after=hashes();assert before==after
    dump('audit.json',dict(inputs_sha256=before,hashes_unchanged=True,cpu_s=time.process_time()-start))
    print('mass',mass,'first',required['joints'])
    print([(r['label'],r['best']['target_m'] if r['best'] else None) for r in candidates])


if __name__=='__main__':main()
