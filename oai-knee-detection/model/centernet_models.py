from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
#from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net

_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'hourglass': get_large_hourglass_net,
}


def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model
