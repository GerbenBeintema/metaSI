
from metaSI.utils.fitting import nnModule_with_fit
from metaSI.data.norms import Norm
from metaSI.utils.networks import MLP_res_net
from metaSI.distributions.normals import Mixture_multivariate_normals, Mixture_normals
from torch.functional import F

#the target of this file is to get a good way of approximating p_theta (y | z)
#this is done by writting
#p_theta (y | z) = sum_i w_i(z, theta) M(y | mu_i(z, theta) , Sigma_i(z, theta))

#MISO is easy

#MIMO is strange
import numpy as np
import torch

#this is normal distribution or 
class Gaussian_mixture_network(nnModule_with_fit):
    def __init__(self, nz, ny, norm: Norm = Norm(), n_components=10, \
                weight_net=MLP_res_net, weight_net_kwargs={}, 
                loc_net=MLP_res_net, loc_net_kwargs={}, 
                logscale_net=MLP_res_net, logscale_net_kwargs={},
                logscale_od_net=None, logscale_od_net_kwargs={},
                epsilon=1e-7):
        super(Gaussian_mixture_network, self).__init__()
        self.norm = norm
        
        self.nz = nz #(None if z.ndim==1 else z.shape[-1]) if isinstance(z,(np.ndarray)) else z
        self.ny = ny #(None if y.ndim==1 else y.shape[-1]) if isinstance(y,(np.ndarray)) else y
        self.nz_val = 1 if self.nz==None else self.nz
        self.ny_val = 1 if self.ny==None else self.ny
        self.n_components = n_components
        self.epsilon=epsilon
        
        self.weight_net =   weight_net(self.nz_val, n_components, **weight_net_kwargs)
        self.loc_net =      loc_net(self.nz_val, n_components*self.ny_val, **loc_net_kwargs)
        self.logscale_net = logscale_net(self.nz_val, n_components*self.ny_val, **logscale_net_kwargs) #This is the diagonal only
        if logscale_od_net: #off diagonal terms
            self.logscale_od_net = logscale_od_net(self.nz_val, n_components*self.ny_val*self.ny_val, **logscale_od_net_kwargs)
        else:
            self.logscale_od_net = None

    def get_dist(self, z):
        znormed = self.norm.input_transform(z)
        ydist_normed = self.get_dist_normed(znormed)
        return self.norm.output_inverse_transform(ydist_normed) #)*self.ystd + self.y0
    
    def get_dist_normed(self,z): #both the input and output are/will be normalized
        z = z.view(z.shape[0],-1) #to (Nb, nz)
        logw = self.weight_net(z) #will be (Nb, n_components)
        logwminmax = logw - torch.max(logw,dim=-1,keepdim=True).values
        logw = logwminmax - torch.log(torch.sum(torch.exp(logwminmax),dim=-1)[...,None])
        
        locs = self.loc_net(z) #output is (Nb, n_components)
        scale = F.softplus(self.logscale_net(z)) + self.epsilon
        if self.ny is None:
            # dist = Mixture_normals(locs, scale, log_weights=logw)
            dist = Mixture_normals(locs, scale, log_weights=logw)
        else:
            locs = locs.view(locs.shape[0], self.n_components, self.ny)       #(Nb, n_components, ny)
            scale = scale.view(scale.shape[0], self.n_components, self.ny) #(Nb, n_components, ny)
            scale_trils = torch.diag_embed(scale)                        #(Nb, n_components, ny, ny)
            if self.logscale_od_net:
                out = self.logscale_od_net(z).view(locs.shape[0], self.n_components, self.ny, self.ny)
                scale_trils = scale_trils + torch.tril(out,diagonal=-1)
            dist = Mixture_multivariate_normals(locs=locs, scale_trils=scale_trils, log_weights=logw)
        return dist
    
    def loss(self, z, y):
        dist = self.get_dist_normed(z)
        return torch.mean(- dist.log_prob(y))/self.ny_val + - 1.4189385332046727417803297364056176 #times ny_val?

    def make_training_arrays(self, zy):
        z,y = zy
        ynorm = self.norm.output_transform(y) #(y-self.y0)/self.ystd
        znorm = self.norm.input_transform(z) #(z-self.z0)/self.zstd
        return [torch.as_tensor(di, dtype=torch.float32) for di in [znorm, ynorm]]

if __name__=='__main__' and True:
    from matplotlib import pyplot as plt
    def sample():
        if np.random.rand()>0.3:
            return np.random.uniform(-4,-4+8/3)
        else:
            return np.random.uniform(4-8/3,4)
    Nb = 10000
    ny = None
    z = np.zeros((Nb,0))
    ymin, ymax = -4, 4
    yminplot, ymaxplot = -8, 8
    if ny==None:
        ytrain = np.array([sample() for _ in range(Nb)])#np.random.uniform(ymin,ymax,size=Nb)
        yval = np.array([sample() for _ in range(Nb)])#np.random.uniform(ymin,ymax,size=Nb)
    train = (z,ytrain)
    val = (z,yval)

    n_components = 1000
    weight_net_kwargs = {'bias_scale':2.0}
    loc_net_kwargs = {'bias_scale':0.75}
    logscale_net_kwargs = {'bias_scale':0.75}
    from metaSI.data.norms import get_nu_ny_and_auto_norm

    model = Gaussian_mixture_network(*get_nu_ny_and_auto_norm(*train),n_components=n_components,\
        weight_net_kwargs=weight_net_kwargs, loc_net_kwargs=loc_net_kwargs, logscale_net_kwargs=logscale_net_kwargs)
    print('std before weight_net',torch.std(model.weight_net.net_lin.bias))
    print('std before loc_net',torch.std(model.loc_net.net_lin.bias))
    print('std before logscale_net',torch.std(model.logscale_net.net_lin.bias))
    import pickle
    load, name = False, 'models/Bimodal-uniform-model-2'
    if not load:
        model.fit(train, val, iterations=10_000, print_freq=500)
        pickle.dump(model, open(name,'wb'))
    else:
        model = pickle.load(open(name,'rb'))
    print('std after weight_net',torch.std(model.weight_net.net_lin.bias))
    print('std after loc_net',torch.std(model.loc_net.net_lin.bias))
    print('std after logscale_net',torch.std(model.logscale_net.net_lin.bias))

    ydist = model.get_dist(torch.zeros(1,0))[0]
    ytest = torch.linspace(yminplot,ymaxplot,1000)
    yprob_pred = ydist.prob(ytest).detach().numpy() #.sample()
    yprob_weighted_pred = ydist.prob_per_weighted(ytest).detach().numpy() #.sample()
    plt.plot(ytest, yprob_pred,'b',label='prob dist pred')
    plt.plot(ytest, yprob_weighted_pred,'r',alpha=0.3)
    # print(model, model.y0, model.ystd)
    plt.hist(ytrain,bins=int(Nb**0.5),density=True,alpha=0.3,color='y')
    plt.hist(yval,bins=int(Nb**0.5),density=True,alpha=0.3,color='m')
    dens1 = 1/(8/3)*(1-0.3)
    dens2 = 1/(8/3)*0.3
    plt.plot([yminplot,-4,-4,-4+8/3,-4+8/3,4-8/3,4-8/3,4,4,ymaxplot], [0,0,dens1,dens1,0,0,dens2,dens2,0,0],'k',label='True dist')
    plt.legend()
    plt.show()
    # plt.plot(model.iteration_monitor,model.loss_train_monitor)
    # plt.plot(model.iteration_monitor,model.loss_val_monitor)
    # plt.grid()
    # plt.show()

#MIMO
if __name__=='__main__' and False:
    from matplotlib import pyplot as plt
    def sample(n=1):
        if n!=1:
            return np.array([sample() for _ in range(n)])
        th = np.random.rand()*np.pi*2
        r = np.random.uniform(1,1.5)
        return np.array([r*np.cos(th),r*np.sin(th)])
    Nb = 100_000
    ny = 2
    z = np.zeros((Nb,0))
    ymin, ymax = -4, 4
    yminplot, ymaxplot = -8, 8
    if not (ny==None):
        # ytrain = np.random.uniform(ymin,ymax,size=(Nb,ny))
        # yval = np.random.uniform(ymin,ymax,size=(Nb,ny))
        # th = np.pi/4
        # ytrain = ytrain@np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        # yval = yval@np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        ytrain = sample(Nb)
        yval = sample(int(Nb/5))
    train = (np.zeros((len(ytrain),0)),ytrain)
    val = (np.zeros((len(yval),0)),yval)

    n_components = 20
    weight_net_kwargs = {'bias_scale':1.5} #not critical for good performance
    loc_net_kwargs = {'bias_scale':0.75}
    logscale_net_kwargs = {'bias_scale':0.75}
    logscale_od_net = MLP_res_net #None for 

    model = Gaussian_mixture_network(*get_nuy_and_auto_norm(*train),n_components=n_components,\
            weight_net_kwargs=weight_net_kwargs, \
            loc_net_kwargs=loc_net_kwargs, \
            logscale_net_kwargs=logscale_net_kwargs, \
            logscale_od_net=logscale_od_net) #weights
    print('std before weight_net',torch.std(model.weight_net.net_lin.bias))
    print('std before loc_net',torch.std(model.loc_net.net_lin.bias))
    print('std before logscale_net',torch.std(model.logscale_net.net_lin.bias))
    import pickle
    load, name = False, f'models/uniform-donut-ny-{ny}-model-off-diagonal-2'
    if not load:
        model.fit(train, val, iterations=10_000, print_freq=500)
        pickle.dump(model, open(name,'wb'))
    else:
        model = pickle.load(open(name,'rb'))
    print('std after weight_net',torch.std(model.weight_net.net_lin.bias))
    print('std after loc_net',torch.std(model.loc_net.net_lin.bias))
    print('std after logscale_net',torch.std(model.logscale_net.net_lin.bias))

    ydist = model.get_dist(torch.zeros(1,0))[0]
    Nbtest = 40_000
    samp = ydist.sample(Nbtest)
    ytest = sample(Nbtest)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(*ytest.T,'.',alpha=0.05)
    plt.title('test')
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.subplot(1,2,2)
    plt.plot(*samp.numpy().T,'.',alpha=0.05)
    plt.title('pred')
    plt.xlim(xlim), plt.ylim(ylim)
    plt.show()

if __name__=='__main__' and False:
    def sample_exp(x):
        if x<0:
            if np.random.rand()<0.2: #above:
                return np.random.normal(loc=-(x+3)*x/1.5**2,scale=0.05)
            else:
                return np.random.normal(loc=+(x+3)*x/1.5**2,scale=0.05)
        else:
            return np.random.normal(0,0.25)
    def get_data(N, seed=32, sample = sample_exp):
        rng = np.random.RandomState(seed=seed)
        Z_full = rng.uniform(-3,3,size=N)
        Y_full = np.array([sample(zi) for zi in Z_full])
        return Z_full, Y_full

    train = get_data(1000, seed=32) #has more data! changed it from 4000 due to overfitting
    val = get_data(500, seed=33)
    test = get_data(20000, seed=34)

    from matplotlib import pyplot as plt

    n_components = 20
    model = Gaussian_mixture_network(*get_nuy_and_auto_norm(*train), n_components=n_components)

    import pickle
    load, name = True, 'models/toy-example-depedent-static-dist-model-3'
    if not load:
        model.fit(train, val, iterations=10_000, print_freq=500)
        pickle.dump(model, open(name,'wb'))
    else:
        model = pickle.load(open(name,'rb'))
    model.checkpoint_load('lowest_train_loss') #early stopping is important!
    plt.plot(model.iteration_monitor,model.loss_train_monitor,label='train loss')
    plt.plot(model.iteration_monitor,model.loss_val_monitor,label='val loss')
    plt.grid()
    plt.xlabel('it')
    plt.ylabel('-logp loss')
    plt.legend()
    plt.show()
    

    ztest = torch.as_tensor(test[0],dtype=torch.float32)
    ytest = torch.as_tensor(test[1],dtype=torch.float32)
    ydist = model.get_dist(ztest)
    zlinspace = torch.linspace(-3,3,500)
    ylinspace = torch.linspace(-1.5,1.5,501)
    ydist_linspace = model.get_dist(zlinspace)
    ztest_point = torch.tensor([-1.5])

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(ylinspace, model.get_dist(ztest_point)[0].prob_per_weighted(ylinspace).detach().numpy())
    plt.hist([sample_exp(-1.5) for _ in range(10**5)], bins=int(10**(5/2)),density=True)
    # plt.show()
    logplinspace = ydist_linspace.log_prob(ylinspace[:,None])
    plt.subplot(1,2,2)
    plt.contourf(zlinspace.numpy(), ylinspace.numpy(), logplinspace.detach().numpy())
    # plt.show()

    ydist0 = Multimodal_Normal(loc=torch.mean(ytest)[None], scale=torch.std(ytest)[None], weights=torch.ones(1))
    meanneglogp = torch.mean(-ydist.log_prob(ytest).detach())
    meanneglogpnormal = torch.mean(-ydist0.log_prob(ytest).detach())
    print('mean neg log p of dist', meanneglogp)
    print('mean neg log p of normal', meanneglogpnormal)
    print('difference= ',meanneglogp-meanneglogpnormal)

    ysamp = ydist.sample()

    plt.plot(ztest.numpy(), ytest.detach().numpy(),'.',alpha=0.1,label='real')
    plt.plot(ztest.numpy(), ysamp.detach().numpy(),'.',alpha=0.1,label='est')
    plt.ylim(-1.5, 1.5)
    plt.xlim(-3,3)
    plt.legend()
    plt.tight_layout()
    plt.show()
