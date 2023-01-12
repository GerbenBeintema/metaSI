from warnings import warn
from torch import distributions
import numpy as np
import torch


from metaSI.distributions.base_distributions import Distrubution, Multimodal_distrubution

class Angle_pdf(Distrubution):
    def __init__(self, c=None, deltath=None):
        self.c = torch.abs(c)
        self.deltath = (deltath+torch.pi*(c<0)+torch.pi)%(2*torch.pi)-torch.pi
    
    @property
    def event_shape(self):
        return tuple()
    @property
    def batch_shape(self):
        return self.c.shape
    
    def log_prob(self, th): #up to 1e6 of c
#         x = 
        #Ai0e(c) == Ai0(c)*torch.exp(-abs(c))
        #Ai0e(c)*torch.exp(abs(c)) == Ai0(c)
#         absc = 
#         2*torch.pi*torch.special.i0e(c)
        return self.c*torch.cos(th-self.deltath) - np.log(2*np.pi) - torch.log(torch.special.i0e(self.c)) - abs(self.c)
    def stack(self, list_of_distributions, dim=0):
        c = torch.stack([l.c for l in list_of_distributions], dim=dim)
        deltath = torch.stack([l.deltath for l in list_of_distributions], dim=dim)
        return Angle_pdf(c, deltath)

    def sample(self, sample_shape): #this is not trivial
        warn('Sampling of Angle_pdf is currently not entirely accurate', stacklevel=2)
        pdf_normal = distributions.normal.Normal(self.deltath+torch.pi*(self.c<0), 1/torch.abs(self.c)**0.5)
        return (pdf_normal.sample(sample_shape)+torch.pi)%(2*torch.pi) - torch.pi
        
class Multimodal_Angle_pdf(Multimodal_distrubution):
    def __init__(self, c, deltath, weights=None, log_weights=None): #add a, b options
        super(Multimodal_Angle_pdf, self).__init__(weights, log_weights)
        assert c.shape==deltath.shape==self.weights.shape
        #loc.shape = scale.shape = weights.shape = batch_shape + (,Nw)
        self.dist = Angle_pdf(c, deltath)

        #some parameters to compute the mean
        self.mean_integration_num0 = 360
        self.tol_mean_integration = 0.10
    
    @property
    def c(self):
        return self.dist.c
    @property
    def deltath(self):
        return self.dist.deltath
    
    ### Transforms ###
    def __add__(self, other):
        if np.all(other==0):
            return self
        else:
            assert False, 'Transforming an angle pdf does not make sense to me'
    def __mul__(self, other):
        if np.all(other==1):
            return self
        else:
            assert False, 'Transforming an angle pdf does not make sense to me'

    def __getitem__(self, x): 
        x = (x,) if not isinstance(x, tuple) else x
        E = slice(None,None,None)
        c = self.c[x+(...,E)]
        deltath = self.deltath[x+(...,E)]
        weights = self.weights[x+(...,E)]
        log_weights = self.log_weights[x+(...,E)]
        return Multimodal_Angle_pdf(c=c, deltath=deltath, weights=weights, log_weights=log_weights)
    #event_shape and batch_shape
    @property
    def mean(self):
        # complex numbers to compute mean
        # 
        # mean = arg(\int exp(i th) p(th))
        #    r = \int cos(th) * p(th) dth
        #    i = \int sin(th) * p(th) dth
        # mean = atan2(i, r)
        E = slice(None,None,None)
        mean_integration_num = self.mean_integration_num0
        while True:
            th_test_0 = torch.linspace(-torch.pi, torch.pi, mean_integration_num)[(E,) + (None,)*len(self.batch_shape)]
            pth = self.prob(th_test_0)
            if torch.all(abs(torch.mean(pth)*2*torch.pi - 1)<self.tol_mean_integration):
                break
            mean_integration_num = mean_integration_num*2
            assert mean_integration_num<10_000, "Failed to couldn't compute mean due to integration errors int(p(th) dth)!=1"
        real_part = torch.mean(torch.cos(th_test_0) * pth, axis=0)*2*torch.pi
        imaginary_part = torch.mean(torch.sin(th_test_0) * pth, axis=0)*2*torch.pi
        return torch.atan2(imaginary_part, real_part)
    
    @property
    def stddev(self):
        E = slice(None,None,None)
        th_test_0 = torch.linspace(-torch.pi,torch.pi,300)[(E,) + (None,)*len(self.batch_shape)]
        delta = self.mean
        th_test = th_test_0+delta[(None,)+(E,)*len(self.batch_shape)]
        pth = self.prob(th_test).detach()
        return (torch.mean(pth*(th_test_0)**2,dim=0)*torch.pi*2)**0.5

    def stack(self, list_of_distributions, dim=0):
        c = torch.stack([l.c for l in list_of_distributions], dim=dim)
        deltath = torch.stack([l.deltath for l in list_of_distributions], dim=dim)
        weights = torch.stack([l.weights for l in list_of_distributions], dim=dim)
        log_weights = torch.stack([l.log_weights for l in list_of_distributions], dim=dim)
        return Multimodal_Angle_pdf(c, deltath, weights, log_weights)