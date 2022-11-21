from torch import nn
import torch

class constant_net(nn.Module):
    def __init__(self, n_out=5, bias_scale=1.):
        super().__init__()
        self.n_out = n_out
        self.bias = nn.Parameter(bias_scale*(torch.rand(n_out)*2-1)*3**0.5) #init such that it is uniform with a std of 1
    
    def forward(self, x):
        return torch.broadcast_to(self.bias, (x.shape[0], self.n_out))

class MLP_res_net(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, \
                activation=nn.Tanh, zero_bias=True, bias_scale=1.):
        super(MLP_res_net, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_hidden_layers>0 and self.n_in!=0:
            seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
            for i in range(n_hidden_layers-1):
                seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
                seq.append(activation())
            seq.append(nn.Linear(n_nodes_per_layer,n_out))
            self.net = nn.Sequential(*seq)
        else:
            self.net = None
        
        self.net_lin = nn.Linear(n_in, n_out) if n_in>0 else constant_net(n_out,bias_scale=bias_scale)
        if zero_bias:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, val=0) #bias
        
    def forward(self,X):
        if self.net is None:
            return self.net_lin(X)
        else:
            return self.net(X) + self.net_lin(X)