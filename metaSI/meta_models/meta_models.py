
from metaSI.data.norms import Norm
from metaSI.utils.fitting import nnModule_with_fit, Get_sample_fun_to_dataset
from metaSI.utils.networks import MLP_res_net
from metaSI.density_networks.normals import Gaussian_mixture_network

import torch
import numpy as np
from metaSI.distributions.base_distributions import stack_distributions
import random   
from metaSI.data.simulation_results import Multi_step_result, Multi_step_result_list
from torch import nn
from torch.utils.data import ConcatDataset
from metaSI.data.system_data import System_data, System_data_list

class Meta_SS_model(nnModule_with_fit):
    def __init__(self, nu, ny, norm: Norm = Norm(), nz=5, \
        meta_state_advance_net = MLP_res_net, meta_state_advance_kwargs = {},
        meta_state_to_output_dist_net = Gaussian_mixture_network, meta_state_to_output_dist_kwargs=dict(n_components=10)):
        super(Meta_SS_model, self).__init__()
        self.nz = nz
        self.ny = ny
        self.nu = nu
        self.ny_val = 1 if self.ny is None else self.ny
        self.nu_val = 1 if self.nu is None else self.nu
        self.norm = norm
        self.meta_state_advance = meta_state_advance_net(n_in=nz + self.nu_val, n_out=nz, \
                                    **meta_state_advance_kwargs)
        self.meta_state_to_output_dist = meta_state_to_output_dist_net(nz, ny, norm=Norm(), \
                                  **meta_state_to_output_dist_kwargs) #no norm here?

    def make_training_arrays(self, system_data, nf=50, parameter_init=False, return_init_z=True,  stride=1, burn_time=0):
        if isinstance(system_data, System_data_list):
            assert parameter_init==False, 'Using multiple datasets in the case of a parameterized start does not work yet'
            ufuture, yfuture = cat_torch_arrays([self.make_training_arrays(sd, nf=nf, parameter_init=parameter_init,return_init_z=return_init_z,strid=stride)  for sd in system_data])
        else:
            assert isinstance(system_data, System_data)
            system_data_normed = self.norm.transform(system_data)
            u, y = system_data_normed.u, system_data_normed.y
            U, Y = [], []
            for i in range(nf,len(u)+1, stride):
                U.append(u[i-nf:i])
                Y.append(y[i-nf:i])
            ufuture = torch.as_tensor(np.array(U), dtype=torch.float32)
            yfuture = torch.as_tensor(np.array(Y), dtype=torch.float32)
        self.init_z = nn.Parameter(torch.randn((len(ufuture), self.nz))) if parameter_init else torch.zeros((len(ufuture), self.nz))
        return (ufuture, yfuture, self.init_z) if return_init_z else (ufuture, yfuture)
    
    def simulate(self, ufuture, yfuture, init_z, return_z=False):
        zt = init_z
        ydist_preds = []
        ufuture = ufuture[:,:,None] if self.nu is None else ufuture
        zvecs = [] #(Ntime, Nb, nz)
        for ut, yt in zip(torch.transpose(ufuture,0,1),torch.transpose(yfuture,0,1)):
            ydist_pred = self.meta_state_to_output_dist.get_dist_normed(zt)
            ydist_preds.append(ydist_pred)
            zvecs.append(zt) 
            zu = torch.cat([zt, ut], dim=1)
            zt = self.meta_state_advance(zu)
        ydist_preds = stack_distributions(ydist_preds,dim=1) #size is (Nbatch, )
        return (ydist_preds, torch.stack(zvecs,dim=1)) if return_z else ydist_preds

    def loss(self, u, y, init_z, burn_time=0, nf=50, parameter_init=False, stride=1):
        ydist_preds = self.simulate(u, y, init_z)
        return torch.mean(-ydist_preds[:,burn_time:].log_prob(y[:,burn_time:]))/self.ny_val + - 1.4189385332046727417803297364056176

    def multi_step(self, system_data, nf=100):
        if isinstance(system_data, System_data_list):
            return Multi_step_result_list([self.multi_step(system_data_i, nf=nf) for system_data_i in system_data])
        assert isinstance(system_data, System_data)
        if nf=='sim':
            nf = len(system_data)
        ufuture, yfuture, init_z = self.make_training_arrays(system_data, nf=nf)
        with torch.no_grad():
            y_dists, zfuture = self.simulate(ufuture, yfuture, init_z, return_z=True)
        I = self.norm.input_inverse_transform
        O = self.norm.output_inverse_transform
        test_norm = Norm(system_data.u, system_data.y)
        return Multi_step_result(O(yfuture), O(y_dists), test_norm, data=system_data, ufuture=I(ufuture), zfuture=zfuture)

class Meta_SS_model_encoder(Meta_SS_model):
    def __init__(self, nu: int, ny: int, norm: Norm = Norm(), nz: int=5, na: int=6, nb: int=6, na_right:int=0, nb_right:int=0,\
                meta_state_advance_net = MLP_res_net, meta_state_advance_kwargs = {},\
                meta_state_to_output_dist_net = Gaussian_mixture_network, meta_state_to_output_dist_kwargs=dict(n_components=10),\
                past_to_meta_state_net = MLP_res_net, past_to_meta_state_kwargs={}):
        super(Meta_SS_model_encoder, self).__init__(nu, ny, norm, nz, meta_state_advance_net=meta_state_advance_net,\
                meta_state_advance_kwargs=meta_state_advance_kwargs, meta_state_to_output_dist_net=meta_state_to_output_dist_net,\
                    meta_state_to_output_dist_kwargs=meta_state_to_output_dist_kwargs)
        self.na, self.nb, self.na_right, self.nb_right = na, nb, na_right, nb_right
        self.past_to_meta_state = past_to_meta_state_net(self.nu_val*(nb+nb_right)+self.ny_val*(na+na_right), nz, **past_to_meta_state_kwargs)

    def get_training_sample_function(self, system_data, nf=50, **kwargs): #already normalized?
        u, y = system_data.u, system_data.y
        k0 = (max(nf, self.na_right, self.nb_right) + max(self.na, self.nb))
        def get_training_sample(k):
            i = k + k0
            ufuture = u[i-nf:i].astype(np.float32)#.flat
            yfuture = y[i-nf:i].astype(np.float32)#.flat
            upast = u[i-nf-self.nb:i-nf+self.nb_right].astype(np.float32)#.flat
            ypast = y[i-nf-self.na:i-nf+self.na_right].astype(np.float32)#.flat
            return [upast, ypast, ufuture, yfuture]
        get_training_sample.length = len(u) + 1 - k0
        return get_training_sample

    def make_training_arrays(self, system_data, stride=1, preconstruct=True, **kwargs):
        if isinstance(system_data, System_data_list):
            datasets = [self.make_training_arrays(sd, stride=stride, preconstruct=preconstruct, **kwargs) for sd in system_data]
            if preconstruct:
                return cat_torch_arrays(datasets)
            else:
                return ConcatDataset(datasets)
        assert isinstance(system_data, System_data)
        system_data_normed = self.norm.transform(system_data)
        get_training_sample = self.get_training_sample_function(system_data_normed, **kwargs)
        if preconstruct:
            arrays = zip(*[get_training_sample(k) for k in range(0, get_training_sample.length, stride)])
            as_tensor = lambda *x: [torch.as_tensor(np.array(xi), dtype=torch.float32) for xi in x]
            return as_tensor(*arrays)
        else:
            return Get_sample_fun_to_dataset(get_training_sample)
    
    def simulate(self, upast, ypast, ufuture, yfuture, return_z=False):
        Nb = upast.shape[0]
        init_z = self.past_to_meta_state(torch.cat([upast.view(Nb,-1), ypast.view(Nb,-1)],dim=1))
        return super().simulate(ufuture, yfuture, init_z, return_z=return_z)
    def loss(self, upast, ypast, ufuture, yfuture, nf=50, stride=1, clip_loss=None, preconstruct=False):
        ydist_preds = self.simulate(upast, ypast, ufuture, yfuture)
        #log(sqrt(2 pi e)) add the mean entropy of a normal distribution.
        losses = torch.mean(-ydist_preds.log_prob(yfuture), dim=0)/self.ny_val + - 1.4189385332046727417803297364056176
        losses = losses[self.na_right:] if self.na_right>0 else losses #this fixes degeneration of the output distribution problem
        # if clip_loss is not None:
        #     for i,loss in enumerate(losses,start=self.na_right):
        #         if loss.item()>clip_loss:
        #             assert i!=self.na_right, 'clipped after 1 step'
        #             return (self.loss(upast, ypast, ufuture[:,:i], yfuture[:,:i])*(i-self.na_right) + clip_loss*(len(losses)-(i-self.na_right)))/len(losses)
        return torch.mean(losses)
    def multi_step(self, system_data, nf=100):
        if isinstance(system_data, System_data_list):
            return Multi_step_result_list([self.multi_step(system_data_i, nf=nf) for system_data_i in system_data])
        assert isinstance(system_data, System_data)
        if nf=='sim':
            nf = len(system_data) - max(self.na, self.nb)
        upast, ypast, ufuture, yfuture = self.make_training_arrays(system_data, nf=nf)
        with torch.no_grad():
            y_dists, zfuture = self.simulate(upast, ypast, ufuture, yfuture, return_z=True)
        I = self.norm.input_inverse_transform
        O = self.norm.output_inverse_transform
        test_norm = Norm(system_data.u, system_data.y)
        return Multi_step_result(O(yfuture), O(y_dists), test_norm, \
            data=system_data, ufuture=I(ufuture), upast=I(upast), ypast=O(ypast), zfuture=zfuture)

class Meta_SS_model_measure_encoder(Meta_SS_model_encoder): #always includes an encoder
    '''
    z_t+1^t = f^t(z_t^m, u_t)
    z_t+1^m = f^m(z_t+1^t, y_t+1)
    p(y_t| z_t^t) is the output (if z_t^m than it model could be a identity and be oke)
    '''
    def __init__(self, nu: int, ny: int, norm: Norm = Norm(), nz: int=5, na: int=6, nb: int=6, na_right:int=0, nb_right:int=0,\
                meta_state_advance_net = MLP_res_net, meta_state_advance_kwargs = {},\
                meta_state_to_output_dist_net = Gaussian_mixture_network, meta_state_to_output_dist_kwargs=dict(n_components=10),\
                past_to_meta_state_net = MLP_res_net, past_to_meta_state_kwargs={}, 
                measure_update_net = MLP_res_net, measure_update_kwargs={}):
        super(Meta_SS_model_measure_encoder, self).__init__(nu, ny, norm, nz, na, nb, na_right,nb_right, meta_state_advance_net=meta_state_advance_net,\
                meta_state_advance_kwargs=meta_state_advance_kwargs, meta_state_to_output_dist_net=meta_state_to_output_dist_net,\
                meta_state_to_output_dist_kwargs=meta_state_to_output_dist_kwargs, past_to_meta_state_net=past_to_meta_state_net,\
                past_to_meta_state_kwargs=past_to_meta_state_kwargs)
        ny_val = 1 if ny is None else ny
        self.measure_update = measure_update_net(n_in=ny_val+nz, n_out=nz, **measure_update_kwargs)
    def loss(self, upast, ypast, ufuture, yfuture, output_filter_p = 1, nf=50, stride=1, preconstruct=False):
        ydist_preds = self.filter(upast, ypast, ufuture, yfuture, output_filter_p=output_filter_p) #(Nbatch, Ntime)
        losses = torch.mean(-ydist_preds.log_prob(yfuture), dim=0)/self.ny_val + - 1.4189385332046727417803297364056176
        losses = losses[self.na_right:] if self.na_right>0 else losses #this fixes degen problems
        # if clip_loss is not None:
        #     for i,loss in enumerate(losses,start=self.na_right):
        #         if loss.item()>clip_loss:
        #             assert i!=self.na_right, 'clipped after 1 step'
        #             return (self.loss(upast, ypast, ufuture[:,:i], yfuture[:,:i])*(i-self.na_right) + clip_loss*(len(losses)-(i-self.na_right)))/len(losses)
        return torch.mean(losses)
    def filter(self, upast, ypast, ufuture, yfuture, output_filter_p = 1, return_z=False, sample_filter_p = 0, output_noise_rescale=None): #a bit of duplicate code but it's fine. 
        assert output_filter_p + sample_filter_p <= 1 and output_filter_p>=0 and sample_filter_p>=0
        Nb = upast.shape[0]
        zt = self.past_to_meta_state(torch.cat([upast.view(Nb,-1), ypast.view(Nb,-1)],dim=1))
        ydist_preds = []
        ufuture = ufuture[:,:,None] if self.nu is None else ufuture
        yfuture = yfuture[:,:,None] if self.ny is None else yfuture
        zvecs = [] #(Ntime, Nb, nz)
        for ut, yt in zip(torch.transpose(ufuture,0,1),torch.transpose(yfuture,0,1)):
            ydist_pred = self.meta_state_to_output_dist.get_dist_normed(zt)
            ydist_preds.append(ydist_pred)
            zvecs.append(zt)
            r = random.random()
            if r<=output_filter_p + sample_filter_p: #should filter
                if r<=output_filter_p: #output filter
                    ysamp = yt
                else: #sample filter
                    if output_noise_rescale is not None: #rescale noise amplitude, might help with extrapolation issue.
                        ydist_pred = (ydist_pred - ydist_pred.mean)/output_noise_rescale + ydist_pred.mean
                    ysamp = ydist_pred.sample()[:,None] if self.ny is None else ydist_pred.sample()
                zy = torch.cat([zt, ysamp], dim=1)
                zm = self.measure_update(zy)
            else:
                zm = zt
            zu = torch.cat([zm, ut], dim=1)
            zt = self.meta_state_advance(zu)
        ydist_preds = stack_distributions(ydist_preds,dim=1) #size is (Nbatch, )
        return (ydist_preds, torch.stack(zvecs,dim=1)) if return_z else ydist_preds
    def multi_step(self, system_data, nf=100, output_filter_p = 1, sample_filter_p = 0, output_noise_rescale=None):
        if isinstance(system_data, System_data_list):
            res = [self.multi_step(system_data_i, nf=nf, output_filter_p=output_filter_p, sample_filter_p=sample_filter_p) for system_data_i in system_data]
            return Multi_step_result_list(res)
        assert isinstance(system_data, System_data)
        if nf=='sim':
            nf = len(system_data) - max(self.na, self.nb)
        upast, ypast, ufuture, yfuture = self.make_training_arrays(system_data, nf=nf)
        with torch.no_grad():
            y_dists, zfuture = self.filter(upast, ypast, ufuture, yfuture, return_z=True, \
                output_filter_p=output_filter_p, sample_filter_p=sample_filter_p, output_noise_rescale=output_noise_rescale)
        I = self.norm.input_inverse_transform
        O = self.norm.output_inverse_transform
        test_norm = Norm(system_data.u, system_data.y)
        return Multi_step_result(O(yfuture), O(y_dists), test_norm, \
            data=system_data, ufuture=I(ufuture), upast=I(upast), ypast=O(ypast), zfuture=zfuture)

def cat_torch_arrays(arrays):
    # [(x1,x2,y1), (x1,x2,y2)] -> [cat([x1,x2]), cat([x2,x2]), cat([y1,y2])]
    return [torch.cat(o,dim=0) for o in  zip(*arrays)] 

if __name__=='__main__':
    u = np.random.randn(3000,3)
    y = np.random.randn(3000,2)
    sys_data = System_data(u,y)
    u = np.random.randn(1_000,3)
    y = np.random.randn(1_000,2)
    sys_data_val = System_data(u,y)

    # sys_data = System_data_list([sys_data, sys_data]) #works
    from metaSI import get_nu_ny_and_auto_norm
    sys = Meta_SS_model_measure_encoder(*get_nu_ny_and_auto_norm(sys_data))
    # print([a.shape for a in sys.make_training_arrays(sys_data, nf=101)])
    dataset = sys.make_training_arrays(sys_data, nf=101)
    
    sys.fit(sys_data, sys_data_val, loss_kwargs={'preconstruct':True})
