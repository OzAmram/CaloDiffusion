import os, sys, io, pickle, joblib
import h5py as h5
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange
from torch.masked import masked_tensor
import torch.sparse

import torch.utils.data as torchdata

import mplhep as hep


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
