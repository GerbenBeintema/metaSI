import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import distributions

class Distrubutions():
    def __truediv__(self, other):
        assert not isinstance(other, Distrubutions)
        return self*(1/other)
    def __rtruediv__(self, other):
        assert False
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self*other
    def __sub__(self, other):
        return self+(-other)
    def __rsub__(self, other):
        return +other+(-self)
    def __neg__(self):
        return (-1)*self
    def prob(self, other):
        return torch.exp(self.log_prob(other))
    def log_prob(self, other):
        return self.dist.log_prob(other)
    def sample(self, sample_shape=()):
        if isinstance(sample_shape,int):
            sample_shape = (sample_shape,)
        return self.dist.sample(sample_shape)
    def stack(self, list_of_distributions, dim=0):
        raise NotImplementedError('nope, stack should be implemented in subclass')

def stack_distributions(list_of_distributions, dim=0):
    assert len(list_of_distributions)>0
    l = list_of_distributions[0]
    return l.stack(list_of_distributions, dim=dim)


class Multimodal_distrubutions(Distrubutions):
    def __init__(self, weights): #also other parameters?
        if not torch.allclose(torch.sum(weights,dim=-1), torch.tensor(1.)):
            # wsum = torch.sum(weights,dim=-1)
            # for i,(wsumi, w) in enumerate(zip(wsum,weights)):
            #     print(i, wsumi-1, w)
            raise ValueError
        assert torch.all(weights>=0)
        self.weights = weights

    def log_prob_per_weighted(self, other):
        s = (...,None) + (slice(None,None,None),)*len(self.event_shape)
        log_probs = super(Multimodal_distrubutions, self).log_prob(other[s]) # (Nb, Nw) #add the weights dimention
        return torch.log(self.weights) + log_probs #(Nb, n_weights)
    def log_prob(self, other):
        Z = self.log_prob_per_weighted(other)
        Zmax = torch.max(Z,dim=-1).values
        return Zmax + torch.log(torch.sum(torch.exp(Z-Zmax[...,None]),dim=-1))
    def prob_per_weighted(self, other):
        return torch.exp(self.log_prob_per_weighted(other))

    @property
    def event_shape(self):
        return self.dist.event_shape
    @property
    def batch_shape(self):
        return self.dist.batch_shape[:-1] #skip weights
    @property
    def n_weights(self):
        return self.dist.batch_shape[-1]
    
    def sample(self, sample_shape=(), use_vectorize=False):
        sample_shape = (sample_shape,) if isinstance(sample_shape,int) else sample_shape
        samples = self.dist.sample(sample_shape) #.shape = (sample_shape, batch_shape, n_weights, event_shape)
        wbroad = torch.broadcast_to(self.weights.detach(), sample_shape + self.batch_shape + (self.n_weights,))
        
        samples = torch.reshape(samples, (-1, self.n_weights) + self.event_shape) #to samples_shape.prod*batch_shape.prod, n_weights, event_shape
        wbroad = torch.reshape(wbroad, (-1, self.n_weights)) #to samples_shape.prod*batch_shape.prod, n_weights
        
        wbroad_acc = np.add.accumulate(wbroad,axis=1)
        randnow = np.random.rand(wbroad.shape[0])