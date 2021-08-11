import torch.nn as nn
import torch
import math


def rand_t(*sz):
    return torch.randn(sz) / math.sqrt(sz[0])


def rand_p(*sz):
    return nn.Parameter(rand_t(*sz))
