import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from utils import *
from models import *
from CaloDiffu import *



class ConsistencyTrainer(nn.Module):
    "Consistency model a al  https://arxiv.org/pdf/2303.01469.pdf"
    def __init__(self, config = None, diffu_model = None, online_model = None, ema_model = None):
        super(ConsistencyModel, self).__init__()
        self.dModel = diffu_model

        self.nsteps = diffu_model.nsteps
        self.discrete_time = diffu_model.discrete_time


        self.model = online_model
        self.ema_model = ema_model






