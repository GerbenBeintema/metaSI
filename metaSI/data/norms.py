import numpy as np
from metaSI.data.system_data import System_data, System_data_list

class Norm:
    def __init__(self, u=None, y=None, umean=0., ustd=1., ymean=0., ystd=1.):
        if u is None:
            self.umean = umean
            self.ustd = ustd
        else:
            self.input_norm_fit(u) #self.umean = #this as tensors or as numpy arrays?
        if y is None:    
            self.ymean = ymean
            self.ystd = ystd
        else:
            self.output_norm_fit(y)
    def input_norm_fit(self, u):
        self.umean = np.mean(u,axis=0)
        self.ustd = np.std(u,axis=0)
    def output_norm_fit(self, y):
        self.ymean = np.mean(y,axis=0)
        self.ystd = np.std(y,axis=0)
    def fit(self, u, y):
        self.input_norm_fit(u)
        self.output_norm_fit(y)

    #to normalized
    def output_transform(self, y): #un-normlized -> normalized
        return (y-self.ymean)/self.ystd
    def input_transform(self, u): #un-normlized -> normalized
        return (u-self.umean)/self.ustd
    def transform(self, system_data):
        assert isinstance(system_data, (System_data, System_data_list)) and system_data.normed==False
        if isinstance(system_data, System_data_list):
            return System_data_list([self.transform(sd) for sd in system_data.sdl])
        u, y = system_data.u, system_data.y
        return System_data(self.input_transform(u), self.output_transform(y), x=system_data.x, normed=True, dt=system_data.dt)
    
    #back normalization
    def output_inverse_transform(self, y): #un-normlized -> normalized
        return y*self.ystd + self.ymean #(y-self.ymean)/self.ystd
    def input_inverse_transform(self, u): #un-normlized -> normalized
        return u*self.ustd + self.umean #(u-self.umean)/self.ustd
    def inverse_transform(self, system_data):
        assert isinstance(system_data, (System_data, System_data_list)) and system_data.normed==True
        if isinstance(system_data, System_data_list):
            return System_data_list([self.inverse_transform(sd) for sd in system_data.sdl])
        u, y = system_data.u, system_data.y
        return System_data(self.input_inverse_transform(u), self.output_inverse_transform(y), x=system_data.x, normed=False, dt=system_data.dt)
    def __repr__(self):
        return f'System_data_norm: (uumeean={self.umean}, ustd={self.ustd}, ymean={self.ymean}, ystd={self.ystd})'

def get_nu_ny_and_auto_norm(u, y=None):
    if isinstance(u, (System_data, System_data_list)):
        u, y = u.u, u.y
    nu = None if u.ndim==1 else u.shape[1]
    ny = None if y.ndim==1 else y.shape[1]
    norm = Norm(u, y)
    return nu, ny, norm
