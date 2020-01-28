import numpy as np
import torch
import time
import os


def gaussian_kernel(x_total, y_total, x_s, y_s, sigma):
    mask = np.zeros((x_total, y_total))
    mask[x_s, y_s] = 1
    x_index = np.arange(0, x_total)
    y_index = np.arange(0, y_total)
    x_index = (x_index - x_s) ** 2
    y_index = (y_index - y_s) ** 2
    x_index = x_index.reshape(-1, 1)
    dist = x_index + y_index
    # gaussian kernel computation
    dist = np.exp(- dist / (2*(sigma ** 2)))
    return mask, dist


# pixel wise focal loss
def focal_loss(pred, true, mask, alpha=2, beta=4):
    focal_weights = torch.where(torch.eq(mask, 1), torch.pow(1. - pred, alpha),
                                torch.pow(pred, alpha) * torch.pow(1 - true, beta))
    # normalize the weigts such that it sums to 1.
    #print(focal_weights.mean(dim=(1, 2)), focal_weights.sum(dim=(1, 2)))
    focal_weights = focal_weights #/ focal_weights.sum(dim=(1, 2)).unsqueeze(dim=1).unsqueeze(dim=2)
    bce = - (mask * torch.log(pred + 1e-12) + (1 - mask) * torch.log(1 - pred + 1e-12))
    # print(bce.mean(dim=(1, 2)))
    # print(focal_weights.shape, bce.shape, mask.shape, mask.sum(dim=(1, 2)).shape)
    loss = focal_weights * bce # / N
    # loss = pos_loss + neg_loss
    loss = loss.mean(0).sum() # average focal loss for each sample
    # print('bce', bce.mean(0).sum().data, 'fl', loss.data)
    return loss

def criterion(prediction, mask, heights=None, widths=None, heatmap=None,
              size_average=True, loss_type='FL',
              alpha=2, beta=4, gamma=1):
    '''
    heights = x
    widths = y
    Implement BCE and pixel-wise focal loss
    alpha/beta are from center net paper
    :param prediction:
    :param mask:
    :param regr:
    :param heatmap:
    :param size_average:
    :param loss_type:
    :param alpha:
    :param beta:
    :return:
    '''
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    if loss_type == 'BCE':
        #    mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
        mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean(0).sum()
    elif loss_type == 'FL':  # focal loss
        mask_loss = focal_loss(pred_mask, heatmap, mask, alpha, beta)
    if (pred_mask > 0.5).any():

        # Regression L1 loss
        if widths is not None:
            pred_width = prediction[:, 1]
            width_loss = (torch.abs(pred_width - widths) * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-12)
            bool_mask = mask == 1
            width_loss = width_loss.mean(0)
        else:
            width_loss = 0

        if heights is not None:
            pred_height = prediction[:, 2]
            height_loss = (torch.abs(pred_height - heights) * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-12)
            height_loss = height_loss.mean(0)

        else:
            height_loss = 0
        # Sum
        loss = mask_loss + gamma * (width_loss + height_loss)
    else:
        loss = mask_loss
        width_loss, height_loss = torch.tensor(0), torch.tensor(0)
    if not size_average:
        loss *= prediction.shape[0]

    return loss, mask_loss, (width_loss + height_loss)


def save_model(model, dir, epoch):
    if not os.path.exists(dir):
        os.makedirs(dir)

    torch.save(model, dir + 'model_{}.pth'.format(epoch))


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    print('Total Module to load {}'.format(len(own_state.keys())))
    print('Total Module from weights file {}'.format(len(state_dict.keys())))
    count = 0
    for name, param in state_dict.items():
        if name not in own_state and name.replace('model.module.','') not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except KeyError as e:
            try:
                own_state[name.replace('model.module.', '')].copy_(param)
            except RuntimeError as e:
                continue
        count += 1

    print('Load Successful {} / {}'.format(count, len(own_state.keys())))
