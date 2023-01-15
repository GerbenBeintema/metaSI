from warnings import warn
from torch import distributions
import numpy as np
import torch


from metaSI.distributions.base_distributions import Distrubution, Mixture

class VonMises(Distrubution):

    def __init__(self, loc, k):
        self.k = torch.abs(k)
        self.loc = ((loc+torch.pi*(k<0))+torch.pi)%(2*torch.pi)-torch.pi
        self.dist = torch.distributions.VonMises(loc, k)
    
    @property
    def event_shape(self):
        return tuple()
    @property
    def batch_shape(self):
        return self.k.shape
    def stack(self, list_of_distributions, dim=0):
        k = torch.stack([l.k for l in list_of_distributions], dim=dim)
        loc = torch.stack([l.loc for l in list_of_distributions], dim=dim)
        return VonMises(loc, k)
    def __getitem__(self, x):
        return VonMises(self.loc[x], self.k[x])
    
    _mean_integration_num0 = 300
    _max_mean_integration_num = 20_000
    _tol_mean_integration = 0.05
    def mixture_mean(self, other):
        E = slice(None,None,None)
        mean_integration_num = self._mean_integration_num0
        while True:
            th_test_0 = torch.linspace(-torch.pi, torch.pi, mean_integration_num)[(E,) + (None,)*len(other.batch_shape)]
            pth = other.prob(th_test_0)
            if torch.all(abs(torch.mean(pth)*2*torch.pi - 1)<self._tol_mean_integration):
                break
            mean_integration_num = mean_integration_num*2
            assert mean_integration_num<self._max_mean_integration_num, "Failed to couldn't compute mean due to integration errors int(p(th) dth)!=1"
        real_part = torch.mean(torch.cos(th_test_0) * pth, axis=0)*2*torch.pi
        imaginary_part = torch.mean(torch.sin(th_test_0) * pth, axis=0)*2*torch.pi
        return torch.atan2(imaginary_part, real_part)
    def mixture_variance(self, other):
        E = slice(None,None,None)
        th_test_0 = torch.linspace(-torch.pi,torch.pi,self._mean_integration_num0)[(E,) + (None,)*len(other.batch_shape)]
        delta = other.mean
        th_test = th_test_0+delta[(None,)+(E,)*len(other.batch_shape)]
        pth = other.prob(th_test).detach()
        return (torch.mean(pth*(th_test_0)**2,dim=0)*torch.pi*2)**0.5

def Mixture_VonMises(k, loc, weights=None, log_weights=None):
    dists = VonMises(loc, k)
    return Mixture(dists, weights, log_weights)


if __name__=='__main__':

    pth = VonMises(torch.as_tensor([2.]), torch.as_tensor([2.]))[0]

    th_test = torch.linspace(-2*torch.pi, 2*torch.pi, 500)
    print(pth)
    prob = pth.prob(th_test)
    from matplotlib import pyplot as plt
    plt.plot(th_test.numpy(), prob.numpy())
    N = 100_000
    th_samples = pth.sample(N)
    plt.hist(th_samples.numpy(), bins=int(N**0.5), density=True)
    plt.show()

    pth2 = VonMises(torch.as_tensor([2.]), torch.as_tensor([2.]))[0]



    pth = Mixture_VonMises(torch.as_tensor([1., 2.]), torch.as_tensor([1., 2.]), weights=torch.as_tensor([0.5, 0.5]))

    th_test = torch.linspace(-2*torch.pi, 2*torch.pi, 500)
    print(pth)
    prob = pth.prob(th_test)
    prob_per_weighted = pth.prob_per_weighted(th_test)
    from matplotlib import pyplot as plt
    plt.plot(th_test.numpy(), prob.numpy(), 'k')
    plt.plot(th_test.numpy(), prob_per_weighted.numpy(), 'r')
    N = 100_000
    th_samples = pth.sample(N)
    plt.hist(th_samples.numpy(), bins=int(N**0.5), density=True)
    plt.show()
    print(pth.mean)
    print(pth.stddev)

    