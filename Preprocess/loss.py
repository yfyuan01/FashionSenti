import math
import time
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
class NormalizationLayer(torch.nn.Module):
    def __init__(self,normalize_scale=1.0,learn_scale=True):
        super(NormalizationLayer,self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))
    def forward(self, x):
        features = self.norm_s * x/torch.norm(x,dim=1,keepdim=True).expand_as(x)
        return features





