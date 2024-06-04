from torch import nn
import torch
import numpy as np
from metaSI.utils.fitting import nnModule_with_fit
import metaSI

class constant_net(nn.Module):
    def __init__(self, n_out=5, bias_scale=1.):
        super().__init__()
        self.n_out = n_out
        self.bias = nn.Parameter(bias_scale*(torch.rand(n_out)*2-1)*3**0.5) #init such that it is uniform with a std of 1
    
    def forward(self, x):
        return torch.broadcast_to(self.bias, (x.shape[0], self.n_out))

class MLP_res_net(nn.Module): #a simple MLP
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, \
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


class ConvShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, \
        padding_mode='zeros'):
        super(ConvShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels*upscale_factor**2, kernel_size, padding=padding, \
            padding_mode=padding_mode)
    
    def forward(self, X):
        X = self.conv(X) #(N, Cout*upscale**2, H, W)
        return nn.functional.pixel_shuffle(X, self.upscale_factor) #(N, Cin, H*r, W*r)

class Upscale_Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', \
                 upscale_factor=2, main_upscale=ConvShuffle, shortcut=ConvShuffle, \
                 padding_mode='zeros', activation=nn.functional.relu, Ch=0, Cw=0):
        assert isinstance(upscale_factor, int)
        super(Upscale_Conv_block, self).__init__()
        #padding='valid' is weird????
        self.shortcut = shortcut(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, upscale_factor=upscale_factor)
        self.activation = activation
        self.upscale = main_upscale(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, upscale_factor=upscale_factor)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.Ch = Ch
        self.Cw = Cw
        
    def forward(self, X):
        #shortcut
        X_shortcut = self.shortcut(X) # (N, Cout, H*r, W*r)
        
        #main line
        X = self.activation(X) # (N, Cin, H, W)
        X = self.upscale(X)    # (N, Cout, H*r, W*r)
        X = self.activation(X) # (N, Cout, H*r, W*r)
        X = self.conv(X)       # (N, Cout, H*r, W*r)
        
        #combine
        # X.shape[:,Cout,H,W]
        H,W = X.shape[2:]
        H2,W2 = X_shortcut.shape[2:]
        if H2>H or W2>W:
            padding_height = (H2-H)//2
            padding_width = (W2-W)//2
            X = X + X_shortcut[:,:,padding_height:padding_height+H,padding_width:padding_width+W]
        else:
            X = X + X_shortcut
        return X[:,:,self.Ch:,self.Cw:] #slice if needed
        #Nnodes = W*H*N(Cout*4*r**2 + Cin)

class CNN_chained_upscales(nn.Module):
    def __init__(self, nx, ny, nu=-1, features_out = 1, kernel_size=3, padding='same', \
                 upscale_factor=2, feature_scale_factor=2, final_padding=4, main_upscale=ConvShuffle, shortcut=ConvShuffle, \
                 padding_mode='zeros', activation=nn.functional.relu):
        super(CNN_chained_upscales, self).__init__()
        self.feedthrough = nu!=-1
        if self.feedthrough:
            self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
            FCnet_in = nx + np.prod(self.nu, dtype=int)
        else:
            FCnet_in = nx
        
        self.activation  = activation
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        if len(ny)==2:
            self.nchannels = 1
            self.None_nchannels = True
            self.height_target, self.width_target = ny
        else:
            self.None_nchannels = False
            self.nchannels, self.height_target, self.width_target = ny
        
        if self.nchannels>self.width_target or self.nchannels>self.height_target:
            import warnings
            text = f"Interpreting shape of data as (Nnchannels={self.nchannels}, Nheight={self.height_target}, Nwidth={self.width_target}), This might not be what you intended!"
            warnings.warn(text)

        #work backwards
        features_out = int(features_out*self.nchannels)
        self.final_padding = final_padding
        height_now = self.height_target + 2*self.final_padding
        width_now  = self.width_target  + 2*self.final_padding
        features_now = features_out
        
        self.upblocks = []
        while height_now>=2*upscale_factor+1 and width_now>=2*upscale_factor+1:
            
            Ch = (-height_now)%upscale_factor
            Cw = (-width_now)%upscale_factor
            # print(height_now, width_now, features_now, Ch, Cw)
            B = Upscale_Conv_block(int(features_now*feature_scale_factor), int(features_now), kernel_size, padding=padding, \
                 upscale_factor=upscale_factor, main_upscale=main_upscale, shortcut=shortcut, \
                 padding_mode=padding_mode, activation=activation, Cw=Cw, Ch=Ch)
            self.upblocks.append(B)
            features_now *= feature_scale_factor
            #implement slicing 
            
            height_now += Ch
            width_now += Cw
            height_now //= upscale_factor
            width_now //= upscale_factor
        # print(height_now, width_now, features_now)
        self.width0 = width_now
        self.height0 = height_now
        self.features0 = int(features_now)
        
        self.upblocks = nn.Sequential(*list(reversed(self.upblocks)))
        self.FC = MLP_res_net(n_in=FCnet_in,n_out=self.width0*self.height0*self.features0, n_hidden_layers=1)
        self.final_conv = nn.Conv2d(features_out, self.nchannels, kernel_size=3, padding=padding, padding_mode='zeros')
        
    def forward(self, x, u=None):
        if self.feedthrough:
            xu = torch.cat([x,u.view(u.shape[0],-1)],dim=1)
        else:
            xu = x
        X = self.FC(xu).view(-1, self.features0, self.height0, self.width0) 
        X = self.upblocks(X)
        X = self.activation(X)
        Xout = self.final_conv(X)
        if self.final_padding>0:
            Xout = Xout[:,:,self.final_padding:-self.final_padding,self.final_padding:-self.final_padding]
        return Xout[:,0,:,:] if self.None_nchannels else Xout


class Down_Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', \
                 downscale_factor=2, padding_mode='zeros', activation=nn.functional.relu):
        assert isinstance(downscale_factor, int)
        super(Down_Conv_block, self).__init__()
        #padding='valid' is weird????
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, stride=downscale_factor)
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', padding_mode='zeros')
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, stride=downscale_factor)
        
    def forward(self, X):
        #shortcut
        X_shortcut = self.shortcut(X) # (N, Cout, H/r, W/r)
        
        #main line
        X = self.activation(X)  # (N, Cin, H, W)
        X = self.conv(X)        # (N, Cout, H, W)
        X = self.activation(X)  # (N, Cout, H, W)
        X = self.downscale(X)   # (N, Cout, H/r, W/r)
        
        #combine
        X = X + X_shortcut
        return X

class CNN_chained_downscales(nn.Module):
    def __init__(self, ny, kernel_size=3, padding='valid', features_ups_factor=1.5, \
                 downscale_factor=2, padding_mode='zeros', activation=nn.functional.relu):

        super(CNN_chained_downscales, self).__init__()
        self.activation  = activation
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        if len(ny)==2:
            self.nchannels = 1
            self.None_nchannels = True
            self.height, self.width = ny
        else:
            self.None_nchannels = False
            self.nchannels, self.height, self.width = ny
        
        #work backwards
        Y = torch.randn((1,self.nchannels,self.height,self.width))
        _, features_now, height_now, width_now = Y.shape
        
        self.downblocks = []
        features_now_base = features_now
        while height_now>=2*downscale_factor+1 and width_now>=2*downscale_factor+1:
            features_now_base *= features_ups_factor
            B = Down_Conv_block(features_now, int(features_now_base), kernel_size, padding=padding, \
                 downscale_factor=downscale_factor, padding_mode=padding_mode, activation=activation)
            
            self.downblocks.append(B)
            with torch.no_grad():
                Y = B(Y)
            _, features_now, height_now, width_now = Y.shape #i'm lazy sorry

        self.width0 = width_now
        self.height0 = height_now
        self.features0 = features_now
        self.nout = self.width0*self.height0*self.features0
        # print('CNN output size=',self.nout)
        self.downblocks = nn.Sequential(*self.downblocks)
        
    def forward(self, Y):
        if self.None_nchannels:
            Y = Y[:,None,:,:]
        return self.downblocks(Y).reshape(Y.shape[0],-1)

def get_image_norm(images):
    images.shape #[nbatch, nfeatures, 1, 1]
    ymean = np.mean(images, axis=(0,2,3), keepdims=True)[0]
    ystd = np.std(images, axis=(0,2,3), keepdims=True)[0]
    norm = Norm(umean=None, ustd=None, ymean=ymean, ystd=ystd)
    return (images.shape[1:], norm)

class Norm(nn.Module):
    def __init__(self, umean, ustd, ymean, ystd):
        super().__init__()
        t = lambda x: x if x is None else torch.nn.Parameter(torch.as_tensor(x,dtype=torch.float32),requires_grad=False)
        self.umean = t(umean)
        self.ustd = t(ustd)
        self.ymean = t(ymean)
        self.ystd = t(ystd)

    def output_transform(self, y):
        return (y - self.ymean)/self.ystd
    def input_transform(self, u):
        return (u - self.umean)/self.ustd
    def output_inverse_transform(self, y):
        return y*self.ystd + self.ymean
    def input_inverse_transform(self, u):
        return u*self.ustd + self.umean

class CNN_Encoder_Decoder(nnModule_with_fit):
    def __init__(self, image_size, norm, nx):
        #self.norm should have ymean [nfeatures, 1, 1]
        super().__init__()
        self.image_size = image_size
        self.norm = norm
        self.down_scales = CNN_chained_downscales(image_size) #gives a small image
        self.flatten_image_size = self.down_scales.width0*self.down_scales.height0*self.down_scales.features0
        self.to_vec = MLP_res_net(self.flatten_image_size, nx)
        self.up_scales = CNN_chained_upscales(nx, image_size) 

    def make_training_arrays(self, images):
        # images = self.norm.output_transform(images) #do later to save memory
        return [torch.as_tensor(di, dtype=torch.float32) for di in [images]]
    
    def encoder(self, images):
        images_normed = self.norm.output_transform(images)
        small_image = self.down_scales(images_normed)
        return self.to_vec(small_image)

    def decoder(self, x):
        images = self.up_scales(x)
        return self.norm.output_inverse_transform(images)

    def loss(self, images):
        return torch.mean(((self.decoder(self.encoder(images)) - images)/self.norm.ystd)**2)
    

if __name__=='__main__':
    Y = np.random.randn(100, 3, 100, 100)
    net = CNN_Encoder_Decoder(*get_image_norm(Y), 5)
    x = net.encoder(torch.as_tensor(Y[:10],dtype=torch.float32))
    print(x.shape)
    print(net.decoder(x).shape)
    net.fit(Y,Y)