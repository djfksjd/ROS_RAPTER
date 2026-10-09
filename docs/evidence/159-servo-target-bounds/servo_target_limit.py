"""Snapshot PD target bounds for experimental stand control, not hardware safety."""
import numpy as np


def limit_servo_action(action, *, q, qd, q0, scale, kp, kd,
                       torque_lower, torque_upper, action_lower, action_upper):
    values=[np.asarray(v,dtype=float) for v in [action,q,qd,q0,scale,kp,kd,
                                               torque_lower,torque_upper,action_lower,action_upper]]
    if any(v.shape!=(8,) or not np.isfinite(v).all() for v in values):
        raise ValueError('finite eight-motor vectors required')
    action,q,qd,q0,scale,kp,kd,low,high,physical_low,physical_high=values
    if (np.any(scale<=0) or np.any(kp<=0) or np.any(kd<0)
            or np.any(low>high) or np.any(physical_low>physical_high)
            or np.any(physical_low<-1) or np.any(physical_high>1)):
        raise ValueError('valid gains and motor/action intervals required')
    with np.errstate(over='raise',invalid='raise'):
        try:
            target_low=(q+(low+kd*qd)/kp-q0)/scale
            target_high=(q+(high+kd*qd)/kp-q0)/scale
            lower=np.maximum(physical_low,target_low)
            upper=np.minimum(physical_high,target_high)
            infeasible=lower>upper
            result=np.where(infeasible,
                            np.clip(.5*(target_low+target_high),physical_low,physical_high),
                            np.clip(action,lower,upper))
        except FloatingPointError as error:
            raise ValueError('finite target calculation required') from error
    return result,infeasible
