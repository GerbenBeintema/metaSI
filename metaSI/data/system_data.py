
import numpy as np

def to_n_shape(v):
    if v is None:
        return 0
    elif v.ndim==1:
        return None
    else:
        return v.shape[1] if v.ndim==2 else v.shape[1:]

class System_data():
    def __init__(self, u=None, y=None, x=None, normed=False, dt=None) -> None:
        super(System_data, self).__init__()
        
        # this does not make a copy if they are already an ndarray, saves some memory
        self.u = (u if isinstance(u,np.ndarray) else np.array(u)) if u is not None else None
        self.x = (x if isinstance(x,np.ndarray) else np.array(x)) if x is not None else None
        self.y = (y if isinstance(y,np.ndarray) else np.array(y)) if y is not None else None
        self.normed = normed
        self.dt = dt

        if self.y is not None and self.u is not None:
            assert len(self.u)==len(self.y), f'{self.u.shape[0]}!={self.y.shape[0]}'
        if self.x is not None: 
            if self.y is not None:
                assert self.x.shape[0]==self.y.shape[0]
            if self.u is not None:
                assert self.x.shape[0]==self.u.shape[0]
    
    def __len__(self):
        '''Number of samples len(system_data)'''
        if self.u is not None:
            return len(self.u)
        elif self.y is not None:
            return len(self.y)
    @property
    def ny(self):
        return to_n_shape(self.y)
    @property
    def nu(self):
        return to_n_shape(self.u)
    @property
    def nx(self):
        return to_n_shape(self.x)
    def __repr__(self):
        return f'System_data of length: {len(self)} nu={self.nu} ny={self.ny} normed={self.normed} dt={self.dt}'
    def plot(self,show=False):
        '''Very simple plotting function'''
        from matplotlib import pyplot as plt
        plt.ylabel('y' if self.y is not None else 'u')
        plt.xlabel('t')

        tar = np.arange(self.u.shape[0])*(1 if self.dt is None else self.dt)
        plt.plot(tar, self.y.reshape((self.y.shape[0],-1)) if self.y is not None else self.u)
        if show: plt.show()

    def train_test_split(self,split_fraction=0.25):
        '''returns 2 data sets of length n*(1-split_fraction) and n*split_fraction respectively (left, right) split'''
        if isinstance(split_fraction, (tuple, list)):
            if len(split_fraction)==1:
                return [self]
            assert sum(split_fraction)==1, str(split_fraction)
            now, rest = self.train_test_split(split_fraction=1-split_fraction[0])
            S = sum(split_fraction[1:])
            others = rest.train_test_split([s/S for s in split_fraction[1:]])
            return [now] + others
        n_samples = self.u.shape[0]
        split_n = int(round(n_samples*(1 - split_fraction)))
        ul,ur,yl,yr = self.u[:split_n], self.u[split_n:], \
                        self.y[:split_n], self.y[split_n:]
        if self.x is None:
            xl,xr = None,None
        else:
            xl,xr = self.x[:split_n], self.x[split_n:]
        left_data = System_data(u=ul, x=xl, y=yl, normed=self.normed,dt=self.dt)
        right_data = System_data(u=ur, x=xr, y=yr, normed=self.normed,dt=self.dt)
        return left_data, right_data

    def __getitem__(self,arg):
        '''Slice the System_data in time index'''
        assert isinstance(arg,slice),'Please use a slice (e.g. sys_data[20:100]) or use sys_data.u or sys_data.y'
        unew = self.u[arg]
        ynew = self.y[arg] if self.y is not None else None
        xnew = self.x[arg] if self.x is not None else None
        return System_data(u=unew, y=ynew, x=xnew, normed=self.normed, dt=self.dt)

    def to_past_future_data_format(self, na=10, nb=10, nf=5, na_right = 0, nb_right = 0, stride=1, force_multi_u=False, force_multi_y=False):
        '''Transforms the system data to encoder structure as structure (uhist,yhist,ufuture,yfuture) of 

        Made for simulation error and multi step error methods

        Parameters
        ----------
        na : int
            y past considered
        nb : int
            u past considered
        nf : int
            future inputs considered

        Returns
        -------
        upast : ndarray (samples, nb, nu) or (sample, nb) if nu=None
            array of [u[k-nb],....,u[k - 1 + nb_right]]
        ypast : ndarray (samples, na, ny) or (sample, na) if ny=None
            array of [y[k-na],....,y[k - 1 + na_right]]
        ufuture : ndarray (samples, nf, nu) or (sample, nf) if nu=None
            array of [u[k],....,u[k+nf-1]]
        yfuture : ndarray (samples, nf, ny) or (sample, nf) if ny=None
            array of [y[k],....,y[k+nf-1]]
        '''
        u, y = np.copy(self.u), np.copy(self.y)
        ypast = []
        upast = []
        ufuture = []
        yfuture = []
        k0 = max(nb, na)
        k0_right = max(nf, na_right, nb_right)
        for k in range(k0+k0_right,len(u)+1,stride):
            kmid = k - k0_right
            ypast.append(y[kmid-na:kmid+na_right])
            upast.append(u[kmid-nb:kmid+nb_right])
            yfuture.append(y[kmid:kmid+nf])
            ufuture.append(u[kmid:kmid+nf])
        upast, ypast, ufuture, yfuture = np.array(upast), np.array(ypast), np.array(ufuture), np.array(yfuture)
        if force_multi_u and upast.ndim==2: #(N, time_seq, nu)
            upast = upast[:,:,None]
            ufuture = ufuture[:,:,None]
        if force_multi_y and ypast.ndim==2: #(N, time_seq, ny)
            ypast = ypast[:,:,None]
            yfuture = yfuture[:,:,None]
        return upast, ypast, ufuture, yfuture

class System_data_list(System_data):
    '''A list of System_data, has most methods of System_data in a list form with only some exceptions listed below

    Attributes
    ----------
    sdl : list of System_data
    y : array
        concatenated y of system_data contained in sdl
    u : array
        concatenated u of system_data contained in sdl

    Methods
    -------
    append(System_data) adds element to sdl
    extend(list) adds elements to sdl
    __getitem__(number) get (ith system data, time slice) 
    '''
    def __init__(self,sys_data_list = None):
        self.sdl = [] if sys_data_list is None else sys_data_list
        self.sanity_check()
    def sanity_check(self):
        for sys_data in self.sdl:
            assert isinstance(sys_data,System_data)
            assert sys_data.ny==self.ny
            assert sys_data.nu==self.nu
            assert sys_data.normed==self.normed
    @property
    def normed(self):
        return self.sdl[0].normed
    @property
    def N_samples(self):
        return sum(sys_data.u.shape[0] for sys_data in self.sdl)
    @property
    def ny(self):
        return self.sdl[0].ny
    @property
    def nu(self):
        return self.sdl[0].nu
    @property
    def y(self): #concatenate or list of lists
        return np.concatenate([sd.y for sd in self.sdl],axis=0)
    @property
    def u(self): #concatenate or list of lists
        return np.concatenate([sd.u for sd in self.sdl],axis=0)    
    @property
    def n_cheat(self):
        return self.sdl[0].n_cheat
    @property
    def dt(self):
        return self.sdl[0].dt
    def __len__(self): #number of datasets
        return len(self.sdl)

    
    def train_test_split(self,split_fraction=0.25):
        '''return 2 data sets of length n*(1-split_fraction) and n*split_fraction respectively (left, right) split'''
        out = list(zip(*[sd.train_test_split(split_fraction=split_fraction) for sd in self.sdl]))
        left, right = System_data_list(out[0]), System_data_list(out[1])
        return left, right

    def __getitem__(self,arg): #by data set or by time?
        '''get (ith system data, time slice) '''
        if isinstance(arg,tuple) and len(arg)>1: #
            sdl_sub = self.sdl[arg[0]]
            if isinstance(sdl_sub,System_data):
                return sdl_sub[arg[1]]
            else: #sdl_sub is a list
                return System_data_list([sd[arg[1]] for sd in sdl_sub])
        else:
            if isinstance(arg,int): #sdl[i] -> ith data system set
                return self.sdl[arg]
            else: #slice of something
                return System_data_list(self.sdl[arg])
    
    def plot(self,show=False):
        '''Very simple plotting function'''
        from matplotlib import pyplot as plt
        for sd in self.sdl:
            sd.plot()
        if show: plt.show()

    def __repr__(self):
        if len(self)==0:
            return f'System_data_list with {len(self.sdl)} series'
        else:
            return f'System_data_list with {len(self.sdl)} series and total length {self.N_samples}, nu={self.nu}, ny={self.ny}, normed={self.normed} lengths={[len(sd) for sd in self.sdl]} dt={self.sdl[0].dt}'

    def to_past_future_data_format(self, na=10, nb=10, nf=5, na_right = 0, nb_right = 0, \
        stride=1, force_multi_u=False, force_multi_y=False):
        out = [sys_data.to_past_future_data_format(na=na,nb=nb,na_right=na_right,nb_right=nb_right,nf=nf,stride=stride,force_multi_u=force_multi_u,\
                force_multi_y=force_multi_y) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in zip(*out)] #(I,I,I),(ys,ys,ys)
