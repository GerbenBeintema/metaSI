

import os
from copy import deepcopy
from secrets import token_urlsafe

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


class nnModule_with_fit(nn.Module):
    def fit(self, train, val, iterations=10_000, batch_size=256, loss_kwargs={}, \
            print_freq=100, loss_kwargs_val=None, call_back_validation=None, \
            val_freq=None, optimizer=None, save_freq=None, save_filename=None):
        '''The main fitting function    
         it uses 
          - self.make_training_arrays
          - self.loss
        '''
        loss_kwargs_val = loss_kwargs if loss_kwargs_val is None else loss_kwargs_val
        val_data = self.make_training_arrays(val, **loss_kwargs_val)
        train_data = self.make_training_arrays(train, **loss_kwargs)

        
        print('Number of datapoints:', len(train_data[0]), '\tBatch size: ', batch_size, '\tIterations per epoch:', len(train_data[0])//batch_size)
        print('Training arrays size:', array_byte_size(train_data), 'Validation arrays size:', array_byte_size(val_data))
        if optimizer is not None:
            self.optimizer = optimizer
        elif not hasattr(self,'optimizer'):
            self.optimizer = torch.optim.Adam(self.parameters())

        #monitoring and checkpoints
        if not hasattr(self, 'loss_train_monitor'):
            self.loss_train_monitor, self.loss_val_monitor, self.iteration_monitor = [], [], []
            iteration_counter_offset = 0
        else:
            print('*** Restarting training!!!, this might result in weird behaviour')
            self._check_and_refresh_optimizer_if_needed() #
            iteration_counter_offset = self.iteration_monitor[-1] if len(self.iteration_monitor)>0 else 0
        lowest_train_loss_seen, loss_train_acc, _ = float('inf'), 0, self.checkpoint_save('lowest_train_loss')
        lowest_val_loss_seen, loss_val, _ = float('inf'), float('inf'), self.checkpoint_save('lowest_val_loss')
        val_freq  = print_freq if val_freq==None  else val_freq
        save_freq = print_freq if save_freq==None else save_freq
        if save_filename is None and save_freq!=False:
            code = token_urlsafe(4).replace('_','0').replace('-','a')
            save_filename = os.path.join(get_checkpoint_dir(), f'{self.__class__.__name__}-{code}.pth')
    
        data_iter = enumerate(tqdm(Dataloader_iterations(train_data, batch_size=batch_size, iterations=iterations), initial=1),start=1)
        try:
            for iteration, batch in data_iter:
                def closure():
                    loss = self.loss(*batch,**loss_kwargs)
                    self.optimizer.zero_grad()
                    loss.backward()
                    return loss
                loss = self.optimizer.step(closure)
                loss_train_acc += loss.item()
                
                if iteration%val_freq==0:  #Validation
                    with torch.no_grad():
                        loss_val = self.loss(*val_data, **loss_kwargs_val).item() if call_back_validation is None else call_back_validation(locals(), globals())
                    if loss_val<lowest_val_loss_seen:
                        lowest_val_loss_seen = loss_val
                        self.checkpoint_save('lowest_val_loss')
                if iteration%print_freq==0: #Printing and monitor update
                    loss_train = loss_train_acc/print_freq
                    m = '!' if loss_train<lowest_train_loss_seen else ' '
                    M = '!!' if len(self.loss_val_monitor)==0 or np.min(self.loss_val_monitor)>lowest_val_loss_seen else '  '
                    print(f'it {iteration:7,} loss {loss_train:.3f}{m} loss val {loss_val:.3f}{M}')
                    self.loss_train_monitor.append(loss_train)
                    self.loss_val_monitor.append(loss_val)
                    self.iteration_monitor.append(iteration+iteration_counter_offset)
                    if loss_train<lowest_train_loss_seen:
                        lowest_train_loss_seen = loss_train
                        self.checkpoint_save('lowest_train_loss')
                    loss_train_acc = 0
                if save_freq!=False and (iteration%save_freq==0): #Saving
                    self.save_to_file(save_filename)
        except KeyboardInterrupt:
            print('stopping early, ', end='')
        print('Saving parameters to checkpoint self.checkpoints["last"] and loading self.checkpoints["lowest_val_loss"]')
        self.checkpoint_save('last')
        self.checkpoint_load('lowest_val_loss') #Should this also save the monitors?
        if save_freq!=False:
            self.save_to_file(save_filename)
    
    def checkpoint_save(self,name): #checkpoints do not use files
        if not hasattr(self, 'checkpoints'):
            self.checkpoints = {}
        self.checkpoints[name] = {'state_dict':deepcopy(self.state_dict()),'optimizer_state_dict':deepcopy(self.optimizer.state_dict())}
    def checkpoint_load(self, name):
        self.load_state_dict(self.checkpoints[name]['state_dict'])
        self.optimizer.load_state_dict(self.checkpoints[name]['optimizer_state_dict'])
    def save_to_file(self, file):
        torch.save(self, file)
    
    def _check_and_refresh_optimizer_if_needed(self):
        if hasattr(self.optimizer, '_cuda_graph_capture_health_check'): 
            try:
                self.optimizer._cuda_graph_capture_health_check()
            except AttributeError:
                print('*** Refreshing optimizer with _refresh_optimizer (probably due to a restart of training after loading the model from a file)')
                self._refresh_optimizer()

    def _refresh_optimizer(self):
        optimizer = self.optimizer.__class__(self.parameters(), **self.optimizer.defaults)
        optimizer.load_state_dict(self.optimizer.state_dict())
        self.optimizer = optimizer

def get_checkpoint_dir():
    '''A utility function which gets the checkpoint directory for each OS

    It creates a working directory called meta-SS-checkpoints 
        in LOCALAPPDATA/meta-SS-checkpoints/ for windows
        in ~/.meta-SS-checkpoints/ for unix like
        in ~/Library/Application Support/meta-SS-checkpoints/ for darwin

    Returns
    -------
    checkpoints_dir
    '''
    def mkdir(directory):
        if os.path.isdir(directory) is False:
            os.mkdir(directory)
    from sys import platform
    if platform == "darwin": #not tested but here it goes
        checkpoints_dir = os.path.expanduser('~/Library/Application Support/meta-SS-checkpoints/')
    elif platform == "win32":
        checkpoints_dir = os.path.join(os.getenv('LOCALAPPDATA'),'meta-SS-checkpoints/')
    else: #unix like, might be problematic for some weird operating systems.
        checkpoints_dir = os.path.expanduser('~/.meta-SS-checkpoints/')#Path('~/.deepSI/')
    mkdir(checkpoints_dir)
    return checkpoints_dir

class Dataloader_iterations:
    def __init__(self, data, batch_size, iterations):
        self.data = [torch.as_tensor(d,dtype=torch.float32) for d in data] #this copies the data again
        self.batch_size = batch_size
        self.iterations = iterations
    
    def __iter__(self):
        return Dataloader_iterationsIterator(self.data, self.batch_size, self.iterations)
    def __len__(self):
        return self.iterations
    
class Dataloader_iterationsIterator:
    def __init__(self, data, batch_size, iterations):
        self.ids = np.arange(len(data[0]),dtype=int)
        self.data = data
        self.L = len(data[0])
        self.batch_size = self.L if batch_size>self.L else batch_size            
        self.data_counter = 0
        self.it_counter = 0
        self.iterations = iterations

    def __iter__(self):
        return self
    def __next__(self):
        self.it_counter += 1
        if self.it_counter>self.iterations:
            raise StopIteration
        self.data_counter += self.batch_size
        ids_now = self.ids[self.data_counter-self.batch_size:self.data_counter]
        if self.data_counter+self.batch_size>self.L: #going over the limit next time, hence, shuffle and restart
            self.data_counter = 0
            np.random.shuffle(self.ids)
        return [d[ids_now] for d in self.data]


def array_byte_size(arrays, name='training'):
    Dsize = sum([d.detach().numpy().nbytes for d in arrays])
    if Dsize>2**30: 
        dstr = f'{Dsize/2**30:.1f} GB'
    elif Dsize>2**20: 
        dstr = f'{Dsize/2**20:.1f} MB'
    else:
        dstr = f'{Dsize/2**10:.1f} kB'
    return dstr