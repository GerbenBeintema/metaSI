
from metaSI.data.norms import Norm
import torch
from warnings import warn

class Multi_step_result:
    def __init__(self, yfuture, y_dists, norm : Norm, **kwargs):
        #shape = [Nb, Nt, ny] and such
        self.kwargs = kwargs
        self.yfuture = yfuture
        self.y_dists = y_dists
        assert isinstance(norm, Norm)
        self.norm = norm #not the fitted norm but the test norm!
        self.ny = None if yfuture.ndim==2 else (yfuture.shape[2] if yfuture.ndim==3 else yfuture.shape[2:])
    def get_dim(self, batch_average=True, time_average=True, output_average=True):
        dim = []
        if batch_average:
            dim.append(0)
        if time_average:
            dim.append(1)
        if not (self.ny is None) and output_average:
            dim.extend(list(range(2,self.yfuture.ndim)))
        return tuple(dim)
    def RMS(self, batch_average=True, time_average=True, output_average=True):
        yhat = self.y_dists.mean # (Nb, Nt, ny)
        diff = (yhat-self.yfuture)**2
        dim = self.get_dim(batch_average, time_average, output_average)
        return torch.mean(diff, dim=dim).detach().numpy()**0.5
    def NRMS(self, batch_average=True, time_average=True, output_average=True):
        RMS = self.RMS(batch_average, time_average, output_average=False)
        NRMS = RMS/self.norm.ystd #shape = [Nb, Nt, ny] if all False
        if output_average and not (self.ny is None):
            assert False, 'not yet implemented'
            return np.mean(NRMS) #this is incorrect
        else:
            return NRMS
    def log_prob(self, batch_average=True, time_average=True, output_average=True):
        assert output_average==True, 'One cannot view the outputs as seperate'
        logp = self.y_dists.log_prob(self.yfuture) #already averged over outputs
        dim = self.get_dim(batch_average, time_average, output_average=False)
        return torch.mean(logp, dim=dim).detach().numpy()
    def normalized_log_prob(self, batch_average=True, time_average=True, output_average=True): #check this function against manual norm
        #how much better is 
        assert output_average==True, 'One cannot view the outputs as seperate'
        y_dists = self.norm.output_transform(self.y_dists)
        yfuture = self.norm.output_transform(self.yfuture)
        logp = y_dists.log_prob(yfuture)
        dim = self.get_dim(batch_average, time_average, output_average=False)
        return torch.mean(logp, dim=dim).detach().numpy() #div ny? /self.ny_val + + 1.4189385332046727417803297364056176
        #I can also remove the entropy thing, but that is overkill. 
    def __len__(self) -> int:
        '''The total number of samples i.e. Nb*Nt'''
        return self.yfuture.shape[0]*self.yfuture.shape[1]
    def __repr__(self) -> str:
        return f'Multi_step_result of batch_size={self.yfuture.shape[0]} time_size={self.yfuture.shape[1]} output_size={self.ny}'
    

class Multi_step_result_list(Multi_step_result):
    def __init__(self, lst):
        self.lst = lst
        for result in lst:
            assert isinstance(result, Multi_step_result)
    def log_prob(self, batch_average=True, time_average=True, output_average=True, list_averge=True):
        if list_averge:
            return sum(l.log_prob(batch_average, time_average, output_average)*len(l) for l in self.lst)/self.total_samples()
        else:
            return [l.log_prob(batch_average, time_average, output_average) for l in self.lst]
    def __len__(self) -> int:
        return len(self.lst)
    def total_samples(self) -> int:
        return sum(len(ls) for ls in self.lst)
    def __repr__(self) -> str:
        return f'Multi_step_result_list of length {len(self.lst)} of\n   ' + '\n   '.join([str(l) for l in self.lst])

