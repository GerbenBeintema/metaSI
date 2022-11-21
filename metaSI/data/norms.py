import numpy as np


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
    def output_transform(self, y): #un-normlized -> normalized
        return (y-self.ymean)/self.ystd
    def input_transform(self, u): #un-normlized -> normalized
        return (u-self.umean)/self.ustd
    def output_inverse_transform(self, y): #un-normlized -> normalized
        return y*self.ystd + self.ymean #(y-self.ymean)/self.ystd
    def input_inverse_transform(self, u): #un-normlized -> normalized
        return u*self.ustd + self.umean #(u-self.umean)/self.ustd

def get_nuy_and_auto_norm(u, y):
    nu = None if u.ndim==1 else u.shape[1]
    ny = None if y.ndim==1 else y.shape[1]
    norm = Norm(u, y)
    return nu, ny, norm
