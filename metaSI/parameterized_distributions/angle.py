from metaSI.data.norms import Norm
from metaSI.utils.fitting import nnModule_with_fit
from metaSI.utils.networks import MLP_res_net
import torch
from metaSI.distributions.circular import Mixture_VonMises

class Par_multimodal_angle_pdf(nnModule_with_fit):
    ## form a p_theta(th|z) = Multimodal_Angle_pdf(th | c(z), th(z), weights(z))
    ## 
    ##
    def __init__(self, nz, nth, norm: Norm = Norm(), n_components=10, \
                weight_net=MLP_res_net, weight_net_kwargs={}, 
                loc_net=MLP_res_net, loc_net_kwargs={}, 
                k_net=MLP_res_net, k_net_kwargs={}):
        super(Par_multimodal_angle_pdf, self).__init__()
        self.norm = norm
        assert self.norm.ymean==0 and self.norm.ystd==1
        
        self.nz = nz #(None if z.ndim==1 else z.shape[-1]) if isinstance(z,(np.ndarray)) else z
        assert nth==None
        self.nth = nth #(None if y.ndim==1 else y.shape[-1]) if isinstance(y,(np.ndarray)) else y
        self.nz_val = 1 if self.nz==None else self.nz
        self.nth_val = 1 if self.nth==None else self.nth
        self.n_components = n_components
        
        self.weight_net = weight_net(self.nz_val, n_components, **weight_net_kwargs)
        self.loc_net =    loc_net(self.nz_val, n_components*self.nth_val, **loc_net_kwargs)
        self.k_net =      k_net(self.nz_val, n_components*self.nth_val, **k_net_kwargs) 

    def get_dist(self, z):
        znormed = self.norm.input_transform(z)
        ydist_normed = self.get_dist_normed(znormed)
        return self.norm.output_inverse_transform(ydist_normed) #)*self.ystd + self.y0
    
    def get_dist_normed(self,z): #both the input and output are/will be normalized
        z = z.view(z.shape[0],-1) #to (Nb, nz)
        logw = self.weight_net(z) #will be (Nb, n_components)
        
        logwmax = torch.max(logw,dim=-1).values[...,None] #(Nb, 1)
        logwminmax = logw - logwmax #(Nb, n_components) - (Nb, 1)
        logw = logwminmax - torch.log(torch.sum(torch.exp(logwminmax),dim=-1, keepdim=True))
        
        loc = self.loc_net(z)*torch.pi #output is (Nb, n_components)
        k = torch.exp(self.k_net(z)+3) #output is (Nb, n_components)
        return Mixture_VonMises(loc, k, log_weights=logw)
    
    def loss(self, z, y):
        dist = self.get_dist_normed(z)
        return torch.mean(- dist.log_prob(y))/self.nth_val + - 1.4189385332046727417803297364056176 #times ny_val?

    def make_training_arrays(self, zy):
        z,y = zy
        ynorm = self.norm.output_transform(y) #(y-self.y0)/self.ystd
        znorm = self.norm.input_transform(z) #(z-self.z0)/self.zstd
        return [torch.as_tensor(di, dtype=torch.float32) for di in [znorm, ynorm]]