"""Experimental observation-conditioned, bounded startup trajectory.

Uses only the initial raw observation and fixed mechanical parameters. This is
not a validated recovery, STOP, or hardware controller.
"""
import numpy as np


class BrakingApproach:
    def __init__(self, observation, *, q0, kp, scale, policy_indices,
                 spring_stiffness, spring_rest, reference_action, lower, upper,
                 symmetric_pairs=()):
        observation=np.asarray(observation,dtype=float)
        q0=np.asarray(q0,dtype=float);kp=np.asarray(kp,dtype=float)
        scale=np.asarray(scale,dtype=float);indices=np.asarray(policy_indices,dtype=int)
        stiffness=np.asarray(spring_stiffness,dtype=float)
        rest=np.asarray(spring_rest,dtype=float)
        self.reference=np.array(reference_action,dtype=float,copy=True)
        self.lower=np.array(lower,dtype=float,copy=True)
        self.upper=np.array(upper,dtype=float,copy=True)
        if (observation.shape!=(46,) or q0.shape!=(12,) or kp.shape!=(12,)
                or scale.shape!=(12,) or indices.shape!=(8,)
                or stiffness.shape!=(8,) or rest.shape!=(8,)
                or self.reference.shape!=(8,) or self.lower.shape!=(8,)
                or self.upper.shape!=(8,) or np.any(indices<0) or np.any(indices>=12)
                or len(np.unique(indices))!=8):
            raise ValueError('original observation and eight-motor design required')
        if not all(np.isfinite(x).all() for x in [observation,q0,kp,scale,stiffness,rest,self.reference,self.lower,self.upper]):
            raise ValueError('finite initial observation and design required')
        if (np.any(observation[6:9]!=0) or np.any(kp[indices]<=0) or np.any(scale[indices]<=0)
                or np.any(stiffness<0) or np.any(self.lower<-1)
                or np.any(self.upper>1) or np.any(self.lower>=self.upper)
                or np.any(self.reference<self.lower) or np.any(self.reference>self.upper)):
            raise ValueError('zero commands and valid mechanical/action bounds required')
        q=q0[indices]+observation[12:24][indices]
        spring_torque=-stiffness*(q-rest)
        self.initial=(q-q0[indices]-spring_torque/kp[indices])/scale[indices]
        for left,right in symmetric_pairs:
            if not (0<=left<8 and 0<=right<8 and left!=right):
                raise ValueError('valid symmetric motor pair required')
            self.initial[[left,right]]=np.mean(self.initial[[left,right]])
        self.initial=np.clip(self.initial,self.lower,self.upper)

    def act(self, seconds):
        if not np.isfinite(seconds) or seconds<0:
            raise ValueError('finite nonnegative trajectory time required')
        if seconds>=.2:
            return self.reference.copy()
        fraction=float(seconds)/.2
        return self.initial+(self.reference-self.initial)*fraction
