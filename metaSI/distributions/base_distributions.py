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
        term1 = torch.sum(weights*(variance_components + mean_components**2), dim=-1-len(self.event_shape)) 
        return term1 - mean_mixture**2
    
    def __getitem__(self, x): #this does not work for event shapes
        #this might not work entirely correctly
        x = x  if isinstance(x, tuple) else (x,)
        S = slice(None,None,None)
        dists = self.dists[x+(...,S)]
        weights = self.weights[x+(...,S)]
        log_weights = self.log_weights[x+(...,S)]
        return self.__class__(dists, weights, log_weights)
    
    ### Transforms ###
    def __add__(self, other): #other has a shape which casts to [batch_shape] + [event_shape] and is probably torch
        other = torch.unsqueeze(torch.broadcast_to(torch.as_tensor(other), self.batch_shape + self.event_shape), dim=-1-len(self.event_shape))
        assert not isinstance(other, Distrubution)
        return self.__class__(self.dists + other, self.weights, self.log_weights)
    def __mul__(self, other):
        other = torch.unsqueeze(torch.broadcast_to(torch.as_tensor(other), self.batch_shape + self.event_shape), dim=-1-len(self.event_shape))
        assert not isinstance(other, Distrubution)
        return self.__class__(self.dists*other, self.weights, self.log_weights)
    def __matmul__(self, other): #self@other
        other = torch.unsqueeze(torch.broadcast_to(torch.as_tensor(other), self.batch_shape + (self.event_shape[0],self.event_shape[0])), dim=-1-2*len(self.event_shape))
        assert not isinstance(other, Distrubution)
        return self.__class__(self.dists@other, self.weights, self.log_weights)
    def __rmatmul__(self, other): #other@self
        other = torch.unsqueeze(torch.broadcast_to(torch.as_tensor(other), self.batch_shape + (self.event_shape[0],self.event_shape[0])), dim=-1-2*len(self.event_shape))
        assert not isinstance(other, Distrubution)
        return self.__class__(other@self.dists, self.weights, self.log_weights)

    def log_integrate_multiply(self, other):
        assert isinstance(other, Mixture)
        dists_1 = self.dists[..., None] # batch_shape + (n_components, 1)
        dists_2 = self.dists[..., None, :] # batch_shape + (1, n_components)
        logw1 = self.log_weights[..., None] # batch_shape + (n_components, 1)
        logw2 = other.log_weights[..., None, :] # batch_shape + (1, n_components)

        gij = dists_1.log_integrate_multiply(dists_2) # batch_shape + (n_components, n_components)
        comb = (logw1 + logw2 + gij).flatten(start_dim=len(self.batch_shape))# batch_shape + (n_components * n_components)
        max_comb = comb.max(dim=-1,keepdim=True).values # batch_shape + (1,)
        return max_comb[...,0] + torch.log(torch.sum(torch.exp(comb-max_comb),dim=-1))

class Crossed_distribution(Distrubution):
    def __init__(self, *list_of_distributions) -> None:
        assert len(list_of_distributions)>0
        assert all(list_of_distributions[0].batch_shape==pdf.batch_shape for pdf in list_of_distributions), f'all pdfs need to have the {list_of_distributions}'
        self.list_dists = list_of_distributions
        self.event_shapes = [dist.event_shape for dist in self.list_dists]
        self.event_nys = [np.prod(d,dtype=int) for d in self.event_shapes]
        self.acc_event_nys = [sum(self.event_nys[:i]) for i in range(len(self.event_nys)+1)] #[0, ny1, ny2+ny1, ny3+ny2+ny1]
        self._event_shape = (sum(self.event_nys),)
        self._batch_shape = self.list_dists[0].batch_shape

    @property
    def event_shape(self):
        return self._event_shape
    @property
    def batch_shape(self):
        return self._batch_shape

    def log_prob(self, other):
        #other has shape batch_size + self.event_shape
        log_probs = []
        assert other.shape[-1]==self._event_shape[0]
        for dist, ny_left, ny_right in zip(self.list_dists, self.acc_event_nys[:-1], self.acc_event_nys[1:]):
            other_sub = other[...,ny_left] if dist.event_shape==tuple() else other[...,ny_left:ny_right] #reshape this one to event_shape
            log_probs.append(dist.log_prob(other_sub))
        log_probs = torch.stack(log_probs, dim=-1)
        return log_probs.sum(-1)
    
    def __repr__(self) -> str:
        return super().__repr__() + ' with crossed distributions:\n' + '\n'.join(f' - {i}: {str(l)}' for i,l in enumerate(self.list_dists))

    
    ### Transforms ###
    def __add__(self, other): #other has a shape which casts to [batch_shape] + [event_shape] and is probably torch
        other = torch.as_tensor(other,dtype=float)
        dists = []
        assert other.shape[-1]==self._event_shape[0]
        for dist, ny_left, ny_right in zip(self.list_dists, self.acc_event_nys[:-1], self.acc_event_nys[1:]):
            other_sub = other[...,ny_left] if dist.event_shape==tuple() else other[...,ny_left:ny_right] #reshape this one to event_shape
            dists.append(dist + other_sub)
        return Crossed_distribution(*dists)
    def __mul__(self, other):
        other = torch.as_tensor(other,dtype=float)
        dists = []
        assert other.shape[-1]==self._event_shape[0]
        for dist, ny_left, ny_right in zip(self.list_dists, self.acc_event_nys[:-1], self.acc_event_nys[1:]):
            other_sub = other[...,ny_left] if dist.event_shape==tuple() else other[...,ny_left:ny_right] #reshape this one to event_shape
            dists.append(dist * other_sub)
        return Crossed_distribution(*dists)

    def __getitem__(self, x): #this does not work for event shapes
        #this might not work entirely correctly
        return Crossed_distribution(*[l[x] for l in self.list_dists])

    @property
    def mean(self):
        out = []
        for dist in self.list_dists:
            m = dist.mean
            out.append(m[...,None] if dist.event_shape==() else m)
        return torch.cat(out, dim=-1)
    @property
    def variance(self):
        out = []
        for dist in self.list_dists:
            m = dist.variance
            out.append(m[...,None] if dist.event_shape==() else m)
        return torch.cat(out, dim=-1)

    def stack(self, list_of_distributions, dim=0 ):
        #list_of_distributions : [crossed1, crossed2]
        return Crossed_distribution(*[D[0].stack(D, dim=dim) for D in zip(*[l.list_dists for l in  list_of_distributions])])

    def sample(self, sample_shape=()):
        samples = []
        for dist in self.list_dists:
            sample = dist.sample(sample_shape)
            if dist.event_shape==():
                samples.append(sample[...,None])
            else:
                samples.append(sample)
        return torch.cat(samples,dim=-1)

if __name__=='__main__':
    from metaSI.distributions.normals import Normal
    batch_shape = []
    dist1 = Normal(torch.randn(1)*0, torch.exp(torch.randn(1)))[0]
    dist2 = Normal(torch.randn(1)*0, torch.exp(torch.randn(1)))[0]
    pdf_crossed = Crossed_distribution(dist1, dist2)
    print(pdf_crossed)
    print(pdf_crossed.batch_shape)
    print(pdf_crossed.event_shape)
    y1 = torch.linspace(-5,5,100)
    y2 = torch.linspace(-5,5,100)
    Y = torch.stack(torch.meshgrid(y1,y2,indexing='xy'),dim=-1)
    logp = pdf_crossed.prob(Y)
    print(logp.shape)
    print(torch.mean(logp)*10**2)

    print(pdf_crossed.mean)
    print(pdf_crossed.variance)
    print([l.scale**2 for l in pdf_crossed.list_dists])

    pdf_crossed2 = Crossed_distribution(dist1, dist2)

    pdf = stack_distributions([pdf_crossed, pdf_crossed2])

    pdf2 = Crossed_distribution(pdf,pdf) #this is cursed but it works
    print(stack_distributions([pdf2,pdf2,pdf2]))

    print('pdf2', pdf2)
    print(pdf2.mean, pdf2.variance)
    K = pdf2*[2,2,1,1]
    print(K.mean, K.variance)
    K = pdf2 + [2,2,1,1]
    print(K.mean, K.variance)

    print(pdf2.sample())