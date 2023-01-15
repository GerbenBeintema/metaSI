import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import distributions

class Distrubution:
    def __truediv__(self, other):
        assert not isinstance(other, Distrubution)
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
    def __repr__(self) -> str:
        try:
            return f'{str(self.__class__).split(".")[-1][:-2]} of batch_shape={self.batch_shape} event_shape={self.event_shape}'
        except:
            return super().__repr__()
    @property
    def event_shape(self):
        return self.dist.event_shape
    @property
    def batch_shape(self):
        return self.dist.batch_shape
    @property
    def stddev(self):
        return self.variance**0.5
    
def stack_distributions(list_of_distributions, dim=0): #potentially add some list type of distribution?
    assert len(list_of_distributions)>0
    l = list_of_distributions[0]
    return l.stack(list_of_distributions, dim=dim)

#todo:
#add covariance?
#crossing of distributions (concat)
class Mixture(Distrubution):
    def __init__(self, dists : Distrubution, weights=None, log_weights=None) -> None:
        if log_weights==None:
            log_weights = torch.log(weights)
        elif weights==None:
            weights = torch.exp(log_weights)
        if not torch.allclose(torch.sum(weights,dim=-1), torch.tensor(1.)):
            raise ValueError('The weights of the Multimodal distribution do not sum to 1')
        assert torch.all(weights>=0)
        self.weights = weights
        self.log_weights = log_weights
        self.dists = dists

    @property
    def event_shape(self):
        return self.dists.event_shape
    @property
    def batch_shape(self):
        return self.dists.batch_shape[:-1]
    @property
    def n_components(self):
        return self.dists.batch_shape[-1]

    def log_prob_per_weighted(self, other):
        #example shapes:
        #self.batch_shape [b1, b2]
        #self.dists.batch_shape [b1, b2, nw]
        #other [a1, a2, b1, b2, ny1, ny2]
        #res: other [a1, a2, b1, b2, nw, ny1, ny2]
        s = (...,None) + (slice(None,None,None),)*len(self.event_shape)
        log_probs = self.dists.log_prob(other[s])
        return self.log_weights + log_probs #(Nb, n_components)
    def log_prob(self, other): #this is numerically stable.
        r = self.log_prob_per_weighted(other) #[a1, a2, b1, b2, n_components]
        rmax = torch.max(r,dim=-1,keepdim=True).values ##[a1, a2, b1, b2, 1]
        return rmax[..., 0] + torch.log(torch.sum(torch.exp(r-rmax),dim=-1))
    def prob_per_weighted(self, other): #this is not very numerically stable
        return torch.exp(self.log_prob_per_weighted(other))

    def sample(self, sample_shape=()):
        sample_shape = (sample_shape,) if isinstance(sample_shape,int) else sample_shape
        samples = self.dists.sample(sample_shape) #.shape = (sample_shape, batch_shape, n_components, event_shape)
        wbroad = torch.broadcast_to(self.weights.detach(), sample_shape + self.batch_shape + (self.n_components,))
        
        samples = torch.reshape(samples, (-1, self.n_components) + self.event_shape) #to samples_shape.prod*batch_shape.prod, n_components, event_shape
        wbroad = torch.reshape(wbroad, (-1, self.n_components)) #to samples_shape.prod*batch_shape.prod, n_components
        
        wbroad_acc = np.add.accumulate(wbroad,axis=1)
        randnow = np.random.rand(wbroad.shape[0])
        ids = np.array([np.digitize(r, w_acc) for r, w_acc in zip(randnow, wbroad_acc)]) #is done is a for loop, can be improved
        arange = np.arange(samples.shape[0])
        samples = samples[arange, ids] #samples*batch, event_shape
        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)

    def stack(self, list_of_distributions, dim=0 ):
        if dim<0:
            dim += len(self.batch_shape) + 1 #-1 , [b1, b2, b3] -> 
        dists = stack_distributions([l.dists for l in list_of_distributions], dim = dim)
        weights = torch.stack([l.weights for l in list_of_distributions],dim=dim)
        log_weights = torch.stack([l.log_weights for l in list_of_distributions],dim=dim)
        return self.__class__(dists, weights, log_weights) #__new__?
    
    @property
    def mean(self):
        #loc.shape = Batch_shape + (n_components,) + (event_shape[0],)
        #weights.shape = (batch_shape, n_components)
        if hasattr(self.dists, 'mixture_mean'):
            return self.dists.mixture_mean(self)
        means = self.dists.mean # (batch_shape, n_components) + event_shape
        weights = self.weights[(...,) +(None,)*len(self.event_shape)]
        return torch.sum(means*weights,dim=-1)
    
    @property
    def variance(self):
        #variance(x) = sum_components weight*(var_component(x) + mean_component(x)**2 )  - self.mean**2
        if hasattr(self.dists, 'mixture_variance'):
            return self.dists.mixture_variance(self)
        mean_mixture = self.mean #shape: batch_shape + event_shape
        variance_components = self.dists.variance
        mean_components = self.dists.mean
        weights = self.weights[(...,) +(None,)*len(self.event_shape)]
        term1 = torch.sum(weights*(variance_components + mean_components**2), dim=-1)
        return term1 - mean_mixture**2
    
    def __getitem__(self, x): #this does not work for event shapes
        #this might not work entirely correctly
        x = x  if isinstance(x, tuple) else (x,)
        dists = self.dists[x+(...,slice(None,None,None))]
        weights = self.weights[x+(...,slice(None,None,None))]
        log_weights = self.log_weights[x+(...,slice(None,None,None))]
        return self.__class__(dists, weights, log_weights)
    
    ### Transforms ###
    def __add__(self, other):
        assert not isinstance(other, Distrubution)
        return self.__class__(self.dists + other, self.weights, self.log_weights)
    def __mul__(self, other):
        assert not isinstance(other, Distrubution)
        return self.__class__(self.dists*other, self.weights, self.log_weights)
    def __matmul__(self, other): #self@other
        assert not isinstance(other, Distrubution)
        return self.__class__(self.dists@other, self.weights, self.log_weights)
    def __rmatmul__(self, other): #other@self
        assert not isinstance(other, Distrubution)
        return self.__class__(other@self.dists, self.weights, self.log_weights)