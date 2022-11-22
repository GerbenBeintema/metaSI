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
    
    def log_prob(self, th): #up to 1e6
#         x = 
        #Ai0e(c) == Ai0(c)*torch.exp(-abs(c))
        #Ai0e(c)*torch.exp(abs(c)) == Ai0(c)
#         absc = 
#         2*torch.pi*torch.special.i0e(c)
        return self.c*torch.cos(th-self.deltath) - np.log(2*np.pi) - torch.log(torch.special.i0e(self.c)) - abs(self.c)
    def stack(self, list_of_distributions, dim=0):
        c = torch.stack([l.c for l in list], dim=dim)
        deltath = torch.stack([l.deltath for l in list], dim=dim)
        return Angle_pdf(c, deltath)

    def sample(self, sample_shape): #this is not trivial
        warn('Sampling of Angle_pdf is currently not entirely accurate', stacklevel=2)
        pdf_normal = distributions.normal.Normal(self.deltath+torch.pi*(self.c<0), 1/torch.abs(self.c)**0.5)
        return (pdf_normal.sample(sample_shape)+torch.pi)%(2*torch.pi) - torch.pi
        
class Multimodal_Angle_pdf(Multimodal_distrubution):
    def __init__(self, c, deltath, weights): #add a, b options
        super(Multimodal_Angle_pdf, self).__init__(weights)
        assert c.shape==deltath.shape==weights.shape
        #loc.shape = scale.shape = weights.shape = batch_shape + (,Nw)
        self.dist = Angle_pdf(c, deltath)
    
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
            assert False, 'Transforming an angle pdf does not make sence'
    def __mul__(self, other):
        if np.all(other==1):
            return self
        else:
            assert False, 'Transforming an angle pdf does not make sence'

    def __getitem__(self, x): 
        x = (x,) if not isinstance(x, tuple) else x
        E = slice(None,None,None)
        c = self.c[x+(...,E)]
        deltath = self.deltath[x+(...,E)]
        weights = self.weights[x+(...,E)]
        return Multimodal_Angle_pdf(c=c, deltath=deltath, weights=weights)
    #event_shape and batch_shape
    @property
    def mean(self):
        w = self.weights.view(-1,self.weights.shape[-1]).detach()
        wmax_index = torch.argmax(w,dim=-1)
        deltath = self.deltath.view(-1,self.weights.shape[-1]).detach()
        c = self.c.view(-1,self.weights.shape[-1]).detach()
        delta = deltath[np.arange(deltath.shape[0]),wmax_index] + (c[np.arange(deltath.shape[0]),wmax_index]<0)*torch.pi
        delta = delta.view(self.weights.shape[:-1])
        E = slice(None,None,None)
        th_test_0 = torch.linspace(-torch.pi,torch.pi,300)[(E,) + (None,)*len(self.batch_shape)]
        K1 = 0
        while True:
            K1 += 1
            K2 = 0
            delta_last = delta
            while True:
                K2 += 1 
                th_test = th_test_0+delta[(None,)+(E,)*len(self.batch_shape)]
                pth = self.prob(th_test)

                err = torch.mean(pth*th_test,dim=0)*2*torch.pi - delta
                derr = pth[0]*torch.pi*2 - 1
                delta = delta - err/derr
                delta = (delta + torch.pi)%(2*torch.pi) - torch.pi
                if torch.allclose(delta_last, delta) or K2==20:
                    break
                delta_last = delta
            if torch.all(derr<0):
                break
            print(K1, torch.sum(derr>0))
            delta = delta + torch.pi*(derr>0)*(torch.randn(delta.shape)*2 if K1>1 else 1)
            if K1==20:
                break
        return delta
    
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
        return Multimodal_Angle_pdf(c, deltath, weights)