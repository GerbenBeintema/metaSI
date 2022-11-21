from metaSI.data.norms import Norm
from metaSI.utils.fitting import nnModule_with_fit
from metaSI.utils.networks import MLP_res_net
import torch
from metaSI.distributions.angle import Multimodal_Angle_pdf

class Dependent_dist_angle_pdf(nnModule_with_fit):
    ## form a p_theta(th|z) = Multimodal_Angle_pdf(th | c(z), th(z), weights(z))
    ## 
    ##
    def __init__(self, nz, nth, norm: Norm = Norm(), n_weights=10, \
                weight_net=MLP_res_net, weight_net_kwargs={}, 
                deltath_net=MLP_res_net, deltath_net_kwargs={}, 
                c_net=MLP_res_net, c_net_kwargs={}):
        super(Dependent_dist_angle_pdf, self).__init__()
        self.norm = norm
        assert self.norm.ymean==0 and self.norm.ystd==1
        
        self.nz = nz #(None if z.ndim==1 else z.shape[-1]) if isinstance(z,(np.ndarray)) else z
        assert nth==None
        self.nth = nth #(None if y.ndim==1 else y.shape[-1]) if isinstance(y,(np.ndarray)) else y
        self.nz_val = 1 if self.nz==None else self.nz
        self.nth_val = 1 if self.nth==None else self.nth
        self.n_weights = n_weights
        
        self.weight_net =   weight_net(self.nz_val, n_weights, **weight_net_kwargs)
        self.deltath_net =  deltath_net(self.nz_val, n_weights*self.nth_val, **deltath_net_kwargs)
        self.c_net =        c_net(self.nz_val, n_weights*self.nth_val, **c_net_kwargs) 

    def get_dist(self, z):
        znormed = self.norm.input_transform(z)
        ydist_normed = self.get_dist_normed(znormed)
        return self.norm.output_inverse_transform(ydist_normed) #)*self.ystd + self.y0
    
    def get_dist_normed(self,z): #both the input and output are/will be normalized
        z = z.view(z.shape[0],-1) #to (Nb, nz)
        logw = self.weight_net(z) #will be (Nb, n_weights)
        
        logwmax = torch.max(logw,dim=-1).values[...,None] #(Nb, 1)
        logwminmax = logw - logwmax #(Nb, n_weights) - (Nb, 1)
        w = torch.exp(logwminmax)
        w = w/torch.sum(w,dim=-1)[...,None]
        
        deltath = self.deltath_net(z)*torch.pi #output is (Nb, n_weights)
        c = torch.exp(self.c_net(z)) #output is (Nb, n_weights)
        return Multimodal_Angle_pdf(c, deltath, weights=w)
    
    def loss(self, z, y):
        dist = self.get_dist_normed(z)
        return torch.mean(- dist.log_prob(y))/self.nth_val + - 1.4189385332046727417803297364056176 #times ny_val?

    def make_training_data(self, zy):
        z,y = zy
        ynorm = self.norm.output_transform(y) #(y-self.y0)/self.ystd
        znorm = self.norm.input_transform(z) #(z-self.z0)/self.zstd
        return [torch.as_tensor(di, dtype=torch.float32) for di in [znorm, ynorm]]