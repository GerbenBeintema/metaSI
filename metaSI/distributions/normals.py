import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import distributions
from metaSI.distributions.base_distributions import Distrubution, stack_distributions, Mixture

class Normal(Distrubution):
    def __init__(self, loc, scale) -> None:
        assert loc.shape==scale.shape
        self.loc = loc
        self.scale = scale
        self.dist = distributions.normal.Normal(loc, scale)
    ### Transforms ###
    def __add__(self, other):
        assert not isinstance(other, Distrubution)
        return Normal(loc=self.loc + other, scale=self.scale)
    def __mul__(self, other):
        assert not isinstance(other, Distrubution)
        return Normal(loc=self.loc*other, scale=self.scale*abs(other))
    def __getitem__(self, x): 
        return Normal(self.loc[x], self.scale[x])
    @property
    def mean(self):
        return self.loc
    @property
    def variance(self):
        return self.scale**2
    def stack(self, list_of_distributions, dim=0):
        loc = torch.stack([l.loc for l in list_of_distributions], dim=dim)
        scale = torch.stack([l.scale for l in list_of_distributions], dim=dim)
        return Normal(loc, scale)
    def log_integrate_multiply(self, other):
        assert isinstance(other, Normal)
        # int N(x| mu_1, sigma_1)*N(x| mu_2, sigma_2) dx = 
        # exp( - (mu_1 - mu_2)^2/ (2*(sigma_1^2 + sigma_2^2)) / (sqrt(2pi)* sqrt(sigma_1^2 + sigma_2^2))
        # log:
        # - 0.5*(mu_1 - mu_2)^2/(sigma_1^2 + sigma_2^2) - 0.5*log(2 pi) - 0.5*log(sigma_1^2 + sigma_2^2)
        mu_1, sigma_1 = self.loc, self.scale
        mu_2, sigma_2 = other.loc, other.scale
        sigma_12pow2 = sigma_1**2 + sigma_2**2
        return -0.5*(mu_1 - mu_2)**2/sigma_12pow2 - np.log(2*torch.pi)*0.5 - torch.log(sigma_12pow2)*0.5

class Multivariate_Normal(Distrubution):
    def __init__(self, loc, scale_tril):
        self.loc = loc
        self.scale_tril = scale_tril
        self.dist = distributions.multivariate_normal.MultivariateNormal(loc=loc, scale_tril=scale_tril)
    
    ### Transforms ###
    def __add__(self, other):
        assert not isinstance(other, Distrubution)
        return Multivariate_Normal(loc=self.loc + torch.as_tensor(other,dtype=torch.float), scale_tril=self.scale_tril)
    def __mul__(self, other): #other casts to self.batch_shape + self.event_shape
        assert not isinstance(other, Distrubution)
        if not isinstance(other, torch.Tensor):
            other = torch.as_tensor(other, dtype=torch.float)
        loc = self.loc * other
        scale_tril = torch.einsum('...ij,...j->...ij',self.scale_tril,other)
        return Multivariate_Normal(loc=loc, scale_tril=scale_tril)
    def __matmul__(self, other):
        LTA = torch.einsum('...ji,...jk->...ik', self.scale_tril, other)
        SIGMA = torch.einsum('...ji,...jk->...ik', LTA, LTA)
        scale_tril = torch.linalg.cholesky(SIGMA) #might not be correct?
        loc = torch.einsum('...i,...ik->...k', self.loc, other)
        return Multivariate_Normal(loc=loc, scale_tril=scale_tril)
    def __getitem__(self, x): 
        return Multivariate_Normal(self.loc[x], self.scale_tril[x])

    @property
    def mean(self):
        return self.loc
    @property
    def variance(self): #this does not work
        return self.scale_tril@self.scale_tril.T

    def stack(self, list_of_distributions, dim=0):
        loc = torch.stack([l.loc for l in list_of_distributions], dim=dim)
        scale_tril = torch.stack([l.scale_tril for l in list_of_distributions], dim=dim)
        return Multivariate_Normal(loc, scale_tril)

    def log_integrate_multiply(self, other):
        assert isinstance(other, Multivariate_Normal)
        # int N(x| mu_1, Sigma_1)*N(x| mu_2, Sigma_2) dx
        # defs:
        # Sigma_S^-1 = Sigma_1^-1 + Sigma_2^-1
        # mu_s = Sigma_S*(Sigma_1^-1 mu_1 + Sigma_2^-1 mu_2)
        # 
        # sol:
        # exp(c) |Sigma_s|^1/2/((2*pi)^(n_x/2) | Sigma_1|^1/2 |Sigma_2|^1/2)
        # c = -0.5*mu_1^T Sigma_1^-1 mu_1 -0.5*mu_2^T Sigma_2^-1 mu_2 
        #     +0.5*mu_s^T Sigma_s^-1 mu_s


        # log result:
        # c + logdet(Sigma_s)*0.5 - log(2pi)*(n_x/2) - logdet(Sigma_1)*0.5 - logdet(Sigma_1)*0.5
        # c - logdet(Sigma_s_inv)*0.5 - log(2pi)*(n_x/2) - logdet(Sigma_1)*0.5 - logdet(Sigma_1)*0.5

        #this can be simplified a lot if Sigma is a diagonal matrix
        Sigma_1 = self.dist.covariance_matrix #batch_shape + (ny, ny)
        Sigma_1_inv = self.dist.precision_matrix #batch_shape + (ny, ny)
        Sigma_2 = other.dist.covariance_matrix #batch_shape + (ny, ny)
        Sigma_2_inv = other.dist.precision_matrix #batch_shape + (ny, ny)
        mu_1 = self.dist.loc #batch_shape + ny
        mu_2 = other.dist.loc ##batch_shape + ny

        matvecmul = lambda mat, vec: torch.einsum('...ij,...j', mat, vec)
        vecmatvecmul = lambda vec1, mat, vec2: torch.einsum('...j,...ji,...i', vec1, mat, vec2)

        Sigma_s_inv = Sigma_1_inv + Sigma_2_inv
        Sigma_s = torch.linalg.inv(Sigma_s_inv)
        mu_s = matvecmul(Sigma_s,matvecmul(Sigma_1_inv, mu_1) + matvecmul(Sigma_2_inv, mu_2))

        c = 0.5*( - vecmatvecmul(mu_1, Sigma_1_inv, mu_1) - vecmatvecmul(mu_2, Sigma_2_inv, mu_2) +  vecmatvecmul(mu_s, Sigma_s_inv, mu_s))
        return c - torch.logdet(Sigma_s_inv)*0.5 - torch.logdet(Sigma_1)*0.5 \
            - torch.logdet(Sigma_2)*0.5 - np.log(2*np.pi)*mu_1.shape[-1]*0.5


def Mixture_normals(locs, scales, weights=None, log_weights=None):
    dists = Normal(locs, scales)
    return Mixture(dists, weights, log_weights)

def Mixture_multivariate_normals(locs, scale_trils, weights=None, log_weights=None):
    dists = Multivariate_Normal(locs, scale_trils)
    return Mixture(dists, weights, log_weights)

if __name__=='__main__':
    Nb = ()
    torch.manual_seed(0)
        
    Nnormals = 11
    shape_n = Nb + (Nnormals,)
    locs = torch.randn(shape_n)
    scales = torch.exp(torch.randn(shape_n))+0.01
    weights = torch.exp(torch.randn(shape_n)*2)
    weights = weights/torch.sum(weights,dim=-1)[...,None]


    m = Mixture_normals(locs, scales, weights)
    print(m)
    ytest = torch.linspace(-2,4,500)
    pytest = m.prob(ytest)
    plt.plot(ytest,pytest)
    p_per_weight = m.prob_per_weighted(ytest)
    plt.plot(ytest,p_per_weight,'r')
    pytest_from_weighted = torch.sum(p_per_weight,dim=-1)
    # plt.plot(ytest,pytest_from_weighted,'k')
    xlim = plt.xlim()
    N = 10**4
    plt.hist(m.sample(N).numpy(), bins=int(np.sqrt(N)),density=True,alpha=0.2)
    plt.xlim(xlim)
    plt.grid()
    plt.show()
    assert torch.allclose(pytest_from_weighted, pytest)

if __name__=='__main__':
    Nb = ()
    torch.manual_seed(0)
    ny = 2
    Nnormals = 3
    shape_n = Nb + (Nnormals,)
    locs = torch.as_tensor([[0,0],[2,2.], [-2, -2]],dtype=torch.float)#
    scale_tril = torch.randn(shape_n + (ny,ny))
    for i in range(ny):
        for j in range(ny):
            if i==j:
                scale_tril[...,i,j] = abs(scale_tril[...,i,j])
            elif i<j:
                scale_tril[...,i,j] = 0

    weights = torch.exp(torch.randn(shape_n)/10)
    weights = weights/torch.sum(weights,dim=-1)[...,None]
    print(locs)
    print(weights)


    m = Mixture_multivariate_normals(locs, scale_tril, weights)
    
    xx = torch.linspace(-5,5,501)
    yy = torch.linspace(-5,5,500)
    ytest = torch.stack(torch.meshgrid(yy, xx, indexing='xy'),dim=-1)
    print(m.batch_shape, m.event_shape, ytest.shape)
    for _ in range(2):

        pytest = m.prob(ytest)#m.prob(ytest)
        print('sum=',torch.sum(pytest)*(xx[1]-xx[0])*(yy[1]-yy[0]))
        plt.contour(yy.numpy(),xx.numpy(),pytest.numpy())
        plt.plot(*m.dists.loc.numpy().T,'or')
        plt.grid()
        plt.show()
        M = m.prob_per_weighted(ytest)
        for i in range(M.shape[2]):
            plt.subplot(1,M.shape[2],i+1)
            print(f'i {i} sum=',torch.sum(M[:,:,i])*(xx[1]-xx[0])*(yy[1]-yy[0]))
            plt.contour(yy.numpy(),xx.numpy(),M[:,:,i].numpy())
            plt.colorbar()
        plt.show()
        plt.plot(*m.sample(1000).numpy().T,'.')
        plt.grid()
        plt.show()

        _ = m + np.array([2,2])
        m2 = m*[2,2]


        th = np.pi/2
        m = m@torch.as_tensor([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]], dtype=torch.float)
