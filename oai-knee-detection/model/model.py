from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F



from torchvision.models import resnet18, resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, pretrained, dropout, model='resnet18'):
        super(ResNet, self).__init__()
        assert model in ['resnet18', 'resnet34']
        if model == 'resnet18':
            self.net = resnet18(pretrained=pretrained)
        elif model == 'resnet34':
            self.net = resnet34(pretrained=pretrained)
        self.net.avgpool = nn.AvgPool2d(28)

        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512,8))

    def forward(self, inp):

        return self.net(inp)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()


    def forward(self, inp, target):
        diff = inp - target
        diff_sq = diff ** 2
        return diff_sq.mean()


class ResNetDetection(nn.Module):
    def __init__(self, pretrained, dropout, use_cuda, model):
        super(ResNetDetection, self).__init__()
        assert model in ['resnet18', 'resnet34']
        if model == 'resnet18':
            self.net = resnet18(pretrained=pretrained)
        elif model == 'resnet34':
            self.net = resnet34(pretrained=pretrained)
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        self.avgpool = nn.AvgPool2d(28, 28)

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
        self.detector = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 8))
        if use_cuda:
            self.net.cuda()
            self.classifier.cuda()
            self.detector.cuda()

    def forward(self, inp):
        x = self.net(inp)
        x = self.avgpool(x)
        x = x.squeeze()
        pred = self.classifier(x)
        pred = F.sigmoid(pred)
        coord = self.detector(x)
        return pred, coord


class MaskedMSELoss(nn.Module):
    def __init__(self, nr):
        super(MaskedMSELoss,self).__init__()
        self.nr = nr

    def forward(self, inp, target, knee_label):
        mask = (knee_label == 1)
        mask_neg = (knee_label != 1)
        #print(inp.shape)
        diff_pos = inp[mask, :] - target[mask, :]
        if self.nr:
            diff_neg = inp[mask_neg, :] - self.nr
            diff_neg = diff_neg ** 2
            diff_neg = diff_neg.mean()
            if torch.isnan(diff_neg).any():
                diff_neg = 0
        else:
            diff_neg = 0
        diff_pos = diff_pos ** 2
        # print(diff_pos.mean(), diff_neg)
        diff = diff_pos.mean() + diff_neg
        return diff


class CombinedLoss(nn.Module):
    def __init__(self, gamma, nr):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.mse = MaskedMSELoss(nr)
        self.gamma = gamma
        self.eps = 1e-9

    def forward(self, prob, coord, label, coord_label):
        prob = prob + self.eps
        bce = self.bce(prob, label)
        mse = self.gamma * self.mse(coord, coord_label, label)
        return bce + mse, bce.item(), mse.item()
