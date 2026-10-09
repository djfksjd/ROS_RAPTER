"""Experimental stand controller fed exclusively by the original observation.

Passive joint state is estimated from a frozen local model, never supplied to
act(). Hardware sensors and operation outside the reference region are untested.
"""
import numpy as np


class StandObserverController:
    def __init__(self, design):
        self.A=np.array(design['Ac'],dtype=float,copy=True)
        self.B=np.array(design['Bc'],dtype=float,copy=True)
        self.C=np.array(design['Cc'],dtype=float,copy=True)
        self.L=np.array(design['L'],dtype=float,copy=True)
        self.K=np.array(design['Kc'],dtype=float,copy=True)
        self.indices=np.array(design['indices'],dtype=int,copy=True)
        self.reference=np.array(design['reference'],dtype=float,copy=True)
        self.u_reference=np.array(design['u_reference'],dtype=float,copy=True)
        self.affine=np.array(design['affine'],dtype=float,copy=True)
        self.action_lower=np.array(design['action_lower'] if 'action_lower' in design
                                   else np.full(8,-1.),dtype=float,copy=True)
        self.action_upper=np.array(design['action_upper'] if 'action_upper' in design
                                   else np.full(8,1.),dtype=float,copy=True)
        if self.A.ndim!=2:
            raise ValueError('square observer dynamics required')
        n=self.A.shape[0]
        if (self.A.shape!=(n,n) or self.B.shape!=(n,8)
                or self.C.shape!=(len(self.indices),n)
                or self.L.shape!=(n,len(self.indices)) or self.K.shape!=(8,n)
                or self.reference.shape!=(46,) or self.u_reference.shape!=(8,)
                or self.affine.shape!=(n,) or self.action_lower.shape!=(8,)
                or self.action_upper.shape!=(8,) or np.any(self.indices<0)
                or np.any(self.indices>=46)):
            raise ValueError('invalid observer design dimensions')
        if not all(np.isfinite(x).all() for x in [self.A,self.B,self.C,self.L,self.K,
                                                 self.reference,self.u_reference,self.affine,
                                                 self.action_lower,self.action_upper]):
            raise ValueError('finite observer design required')
        if (np.any(self.action_lower<-1.) or np.any(self.action_upper>1.)
                or np.any(self.action_lower>=self.action_upper)
                or np.any(self.u_reference<self.action_lower)
                or np.any(self.u_reference>self.action_upper)):
            raise ValueError('normalized physical action bounds required')
        self.reset()

    def reset(self):
        self.prior=np.zeros(len(self.A))
        self.posterior=self.prior.copy()

    def act(self, observation):
        observation=np.asarray(observation,dtype=float)
        if observation.shape!=(46,) or not np.isfinite(observation).all():
            raise ValueError('finite original 46-dimensional observation required')
        if np.any(observation[6:9]!=0.):
            raise ValueError('stand observer accepts only zero commands')
        measured=(observation-self.reference)[self.indices]
        with np.errstate(over='raise',invalid='raise'):
            try:
                posterior=self.prior+self.L@(measured-self.C@self.prior)
                action=np.clip(self.u_reference-self.K@posterior,
                               self.action_lower,self.action_upper)
                prior=self.A@posterior+self.B@(action-self.u_reference)+self.affine
            except FloatingPointError as error:
                raise ValueError('observer arithmetic exceeded finite range') from error
        if not np.isfinite(prior).all() or not np.isfinite(action).all():
            raise ValueError('finite observer state and action required')
        self.posterior=posterior
        self.prior=prior
        return action
