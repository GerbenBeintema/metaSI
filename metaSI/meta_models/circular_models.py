from metaSI.data.norms import Norm
from metaSI.utils.fitting import nnModule_with_fit
from metaSI.utils.networks import MLP_res_net
from metaSI.density_networks.normals import Gaussian_mixture_network

import torch
import numpy as np
from metaSI.distributions.base_distributions import stack_distributions
import random   
from metaSI.data.simulation_results import Multi_step_result, Multi_step_result_list
from torch import nn
from metaSI.data.system_data import System_data, System_data_list
from metaSI.meta_models.meta_models import Meta_SS_model_encoder

class Meta_SS_model_encoder_angle(Meta_SS_model_encoder):
    def __init__(self):
        pass