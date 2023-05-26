
from typing import Union
from torch.nn.modules.module import Module
from metaSI.utils.fitting import nnModule_with_fit
from metaSI.data.norms import Norm
from metaSI.utils.networks import MLP_res_net
from metaSI.distributions.normals import Mixture_multivariate_normals, Mixture_normals

#the target of this file is to get a good way of approximating p_theta (y | z)
#this is done by writting
#p_theta (y | z) = sum_i w_i(z, theta) N(y | mu_i(z, theta) , Sigma_i(z, theta))

#MISO is easy

#MIMO is strange
import numpy as np
import torch
from torch import Tensor, nn
from metaSI.density_networks import Gaussian_mixture_network

class Ensable(nnModule_with_fit):
    singulars = ['make_training_arrays']

    def __init__(self, models):
        super().__init__()
        assert len(models)>0
        self.models = nn.ModuleList(models) #need to convert this to a correct kind of list

    def test(self):
        print('test')

    #multi:
    def __getattr__(self, name: str): #this will never work, maybe make it multi
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name in self.singulars:
            return self.models[0].__getattribute__(name)
        else:
            if callable(self.models[0].__getattribute__(name)):
                return lambda *args, **kwargs: [model.__getattribute__(name)(*args, **kwargs) for model in self.models]
            else:
                return [model.__getattribute__(name) for model in self.models]
    
    def loss(self, *args, **kwargs):
        return torch.sum(torch.stack([model.loss(*args, **kwargs) for model in self.models]))


if __name__=='__main__':
    net = Gaussian_mixture_network(None,None)
    print(net.loss)
    ens = Ensable([Gaussian_mixture_network(None,None) for _ in range(3)])
    # print(ens)
    # print(ens.parameters())
    print(ens.__dict__)
    print(ens.models)
    print(ens.test())
    print(ens.loss)
    print(ens.models[0].loss) #this works
    zytrain = (torch.rand(2000),torch.rand(2000))
    zyval = (torch.rand(2000),torch.rand(2000))
    data = ens.make_training_arrays(zytrain)
    losses = ens.loss(*data)
    print(losses)
    print(ens.fit(zytrain, zyval, iterations=10))
    print(ens.get_dist(zyval[0]))

    




#this is normal distribution or 
# class Gaussian_mixture_network(nnModule_with_fit):
#     def __init__(self, nz, ny, norm: Norm = Norm(), n_components=10, \
#                 weight_net=MLP_res_net, weight_net_kwargs={}, 
#                 loc_net=MLP_res_net, loc_net_kwargs={}, 
#                 logscale_net=MLP_res_net, logscale_net_kwargs={},
#                 logscale_od_net=None, logscale_od_net_kwargs={}):
#         super(Gaussian_mixture_network, self).__init__()
#         self.norm = norm
        
#         self.nz = nz #(None if z.ndim==1 else z.shape[-1]) if isinstance(z,(np.ndarray)) else z
#         self.ny = ny #(None if y.ndim==1 else y.shape[-1]) if isinstance(y,(np.ndarray)) else y
#         self.nz_val = 1 if self.nz==None else self.nz
#         self.ny_val = 1 if self.ny==None else self.ny
#         self.n_components = n_components
        
#         self.weight_net =   weight_net(self.nz_val, n_components, **weight_net_kwargs)
#         self.loc_net =      loc_net(self.nz_val, n_components*self.ny_val, **loc_net_kwargs)
#         self.logscale_net = logscale_net(self.nz_val, n_components*self.ny_val, **logscale_net_kwargs) #This is the diagonal only
#         if logscale_od_net: #off diagonal terms
#             self.logscale_od_net = logscale_od_net(self.nz_val, n_components*self.ny_val*self.ny_val, **logscale_od_net_kwargs)
#         else:
#             self.logscale_od_net = None

#     def get_dist(self, z):
#         znormed = self.norm.input_transform(z)
#         ydist_normed = self.get_dist_normed(znormed)
#         return self.norm.output_inverse_transform(ydist_normed) #)*self.ystd + self.y0
    
#     def get_dist_normed(self, z): #both the input and output are/will be normalized
#         z = z.view(z.shape[0],-1) #to (Nb, nz)
#         logw = self.weight_net(z) #will be (Nb, n_components)
#         logwminmax = logw - torch.max(logw,dim=-1,keepdim=True).values
#         logw = logwminmax - torch.log(torch.sum(torch.exp(logwminmax),dim=-1)[...,None])
        
#         locs = self.loc_net(z) #output is (Nb, n_components)
#         scale = torch.exp(self.logscale_net(z)) #output is (Nb, n_components)
#         if self.ny is None:
#             dist = Mixture_normals(locs, scale, log_weights=logw)
#         else:
#             locs = locs.view(locs.shape[0], self.n_components, self.ny)       #(Nb, n_components, ny)
#             scale = scale.view(scale.shape[0], self.n_components, self.ny) #(Nb, n_components, ny)
#             scale_trils = torch.diag_embed(scale)                        #(Nb, n_components, ny, ny)
#             if self.logscale_od_net:
#                 out = self.logscale_od_net(z).view(locs.shape[0], self.n_components, self.ny, self.ny)
#                 scale_trils = scale_trils + torch.tril(out,diagonal=-1)
#             dist = Mixture_multivariate_normals(locs=locs, scale_trils=scale_trils, log_weights=logw)
#         return dist
    
#     def loss(self, z, y):
#         dist = self.get_dist_normed(z)
#         return torch.mean(- dist.log_prob(y))/self.ny_val + - 1.4189385332046727417803297364056176 #times ny_val?

#     def make_training_arrays(self, zy):
#         z,y = zy
#         ynorm = self.norm.output_transform(y) #(y-self.y0)/self.ystd
#         znorm = self.norm.input_transform(z) #(z-self.z0)/self.zstd
#         return [torch.as_tensor(di, dtype=torch.float32) for di in [znorm, ynorm]]

