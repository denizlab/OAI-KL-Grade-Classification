from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot_encoding(y, num_classes):
    one_hot = torch.zeros((y.shape[0], num_classes))
    one_hot[:, y] = 1
    return one_hot


class FocalLoss():
    def __init__(self, alpha, gamma, num_classes=5):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=1)
        self.EPS = 1e-8
    def __call__(self, x, y):
        return self.focal_loss(x, y)

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''

        t = one_hot_encoding(y.data.cpu(), self.num_classes)
        if torch.cuda.is_available():
            t = Variable(t).cuda()
        else:
            t = Variable(t)
        p = self.softmax(x) + self.EPS
        # compute focal loss
        weight = torch.pow(-p + 1., self.gamma)
        loss = - self.alpha * weight * torch.log(p)
        loss = torch.sum(t * loss, dim=1)
        return loss.mean()

