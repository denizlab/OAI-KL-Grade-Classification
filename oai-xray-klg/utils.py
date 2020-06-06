# ==============================================================================
# Copyright (C) 2020 Bofei Zhang, Jimin Tan, Greg Chang, Kyunghyun Cho, Cem Deniz
#
# This file is part of OAI-KL-Grade-Classification
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
import sys
import os
import time
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import cv2
from torch.autograd import Variable
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_squared_error
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_image(filename):
    return np.array(h5py.File(filename, 'r')['data'][:]).astype('float32')


def show_mri(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    plt.show()


def progress_info(cur_epoch, num_epoch, cur_batch, num_batch, batch_speed, loss_dict, accuracy_dict):
    if cur_batch == 0 and cur_epoch != 0:
        t_loss_dict = "{:.4g}".format(loss_dict['train']) if loss_dict['train'] is not None else "NA"
        v_loss_dict = "{:.4g}".format(loss_dict['val']) if loss_dict['val'] is not None else "NA"
        sys.stdout.write("train / val loss {:} / {:}, epoch {:3}, {:.2f}s/b \n".format(t_loss_dict, v_loss_dict, cur_epoch - 1, batch_speed))
        t_accuracy_dict = "{:.2%}".format(accuracy_dict['train']) if accuracy_dict['train'] is not None else "NA"
        v_accuracy_dict = "{:.2%}".format(accuracy_dict['val']) if accuracy_dict['val'] is not None else "NA"
        sys.stdout.write("train / val accuracy {:} / {:} \n".format(t_accuracy_dict, v_accuracy_dict))
        sys.stdout.write(50 * '-' + '\n')
    time_left = time.strftime('%H:%M:%S', time.gmtime(((num_epoch - cur_epoch) * num_batch + num_batch - cur_batch) * batch_speed))
    sys.stdout.write("[{:<30}] {:.2%}".format('=' * int (29 * (cur_batch + 1 + cur_epoch * num_batch) / (num_epoch * num_batch)) + '>', 
                                               (cur_batch + 1 + cur_epoch * num_batch) / float(num_epoch * num_batch)))
    sys.stdout.write(" {:<30}".format(time_left))
    sys.stdout.flush()
    sys.stdout.write('\r')
    return


def progress_info_writer(save_dir, *argv):
    with open(os.path.join(save_dir, 'log.txt'), 'a+') as f:
        for output in argv:
            print(output)
            f.write(output + '\n')


def read_start_epoch(model_dir):
    return int(model_dir.split('.')[0].split('-')[-1])


def load_model(model_dir, min_loss_checkpoint, device):
    model = torch.load(model_dir).to(device)
    min_loss_checkpoint['lastest_dir'] = model_dir
    min_loss_checkpoint['best_dir'] = model_dir
    starting_epoch = read_start_epoch(model_dir)
    return model, starting_epoch


def save_model(min_loss_checkpoint, epoch_loss, epoch_acc, epoch_avg_acc, model, epoch, save_dir):
    if min_loss_checkpoint['val_loss'] is None: # New run/loaded
            min_loss_checkpoint['val_loss'] = epoch_loss['val']
            min_loss_checkpoint['val_acc'] = epoch_acc['val']
            min_loss_checkpoint['val_avg_acc'] = epoch_avg_acc['val']
            min_loss_checkpoint['lastest_dir'] = save_with_epoch(model, epoch, save_dir)
            min_loss_checkpoint['best_dir'] = save_with_epoch(model, epoch, save_dir, best_model = True)
            print(50 * '-' + '\n' + 'First checkpoint created at epoch ' + str(epoch) + 15 * ' ')
    else: # In training process
        min_loss_checkpoint['lastest_dir'] = overwrite_with_epoch(model, epoch, save_dir, min_loss_checkpoint['lastest_dir'])
        # save by lowest validation loss or average multi accuracy.
        if epoch_loss['val'] < min_loss_checkpoint['val_loss'] or epoch_avg_acc['val'] > min_loss_checkpoint['val_avg_acc']:
            old_best_dir = min_loss_checkpoint['best_dir']
            min_loss_checkpoint['best_dir'] = overwrite_with_epoch(model, epoch, save_dir, min_loss_checkpoint['best_dir'], best_model = True)
            min_loss_checkpoint['val_acc'] = epoch_acc['val']
            min_loss_checkpoint['val_avg_acc'] = epoch_avg_acc['val']
            min_loss_checkpoint['val_loss'] = epoch_loss['val']
            print('New best checkpoint created at epoch ' + str(epoch) + ' and ' + old_best_dir + ' removed. \n' + 50 * '-')


def save_with_epoch(model, epoch, save_dir, best_model = False):
    if not os.path.exists( save_dir + '/'):
        os.mkdir(save_dir + '/')
    if not os.path.exists( save_dir + '/' + '/save'):
        os.mkdir(save_dir + '/' + '/save')
    best_prefix = 'best-' if best_model else ''
    save_dir = f'{save_dir}/save/{best_prefix}model-epoch-{epoch}.pt'
    torch.save(model, save_dir)
    return save_dir


def overwrite_with_epoch(model, epoch, save_dir, old_model_dir, best_model = False):
    best_prefix = 'best-' if best_model else ''
    save_dir = f'{save_dir}/save/{best_prefix}model-epoch-{epoch}.pt'
    torch.save(model, save_dir)
    os.remove(old_model_dir)
    return save_dir


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    print('Total Module to load {}'.format(len(own_state.keys())))
    print('Total Module from weights file {}'.format(len(state_dict.keys())))
    count = 0
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        count +=1
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
    print('Load Successful {} / {}'.format(count, len(own_state.keys())))


def compute_metrics(truths, preds):
    '''
    Compute import metrics for klg classification.
    :param truths:
    :param preds:
    :return:
    '''
    truths = np.vstack(truths)
    preds = np.vstack(preds)
    cm = confusion_matrix(truths, preds)
    kappa = np.round(cohen_kappa_score(truths, preds, weights="quadratic"), 4)
    # avoid divide by 0 error when testing...
    avg_acc = np.round(np.mean(cm.diagonal().astype(float) / (cm.sum(axis=1) + 1e-12)), 4)
    mse = np.round(mean_squared_error(truths, preds), 4)
    cm = cm / cm.sum()
    return avg_acc, mse, kappa, cm


def load_img(fname):
    f = h5py.File(fname, 'r')
    img = f['data']
    img = np.expand_dims(img,axis=2)
    img = np.repeat(img[:, :], 3, axis=2)
    f.close()
    return img


def weigh_maps(weights, maps,use_cuda = False):
    maps = maps.squeeze()
    weights = weights.squeeze()
    if use_cuda:
        res = Variable(torch.zeros(maps.size()[-2:]).cuda(), requires_grad=False)
    else:
        res = Variable(torch.zeros(maps.size()[-2:]), requires_grad=False)
    for i, w in enumerate(weights):
        res += w * maps[i]

    return res


# Producing the GradCAM output using the equations provided in the article
def gradcam_resnet(fname, net, scale_tensor_transform, use_cuda=False, label=None):
    img = load_img(fname)
    img = scale_tensor_transform(img)
    inp = img.view(1, 3, 896, 896)
    net.train(False)
    net.zero_grad()
    features = nn.Sequential(net.module.conv1,
                             net.module.bn1,
                             net.module.relu,
                             net.module.maxpool,
                             net.module.layer1,
                             net.module.layer2,
                             net.module.layer3,
                             net.module.layer4)
    if use_cuda:
        maps = features(Variable(inp.cuda()))
    else:
        maps = features(Variable(inp))
    maps_avg = F.avg_pool2d(maps, 28).view(1, 512)

    grads = []
    maps_avg.register_hook(lambda x: grads.append(x));

    out = net.module.fc(maps_avg)

    ohe = OneHotEncoder(sparse=False, n_values=5)
    index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)
    if use_cuda:
        out.backward(torch.from_numpy(ohe.fit_transform(index)).float().cuda())
    else:
        out.backward(torch.from_numpy(ohe.fit_transform(index)).float())
    heatmap = F.relu(weigh_maps(grads[0], maps, use_cuda)).data.cpu().numpy()
    heatmap = cv2.resize(heatmap, (896, 896), cv2.INTER_CUBIC)
    #if label == 3:
    #    heatmap = heatmap.max() - heatmap
    probs = F.softmax(out, dim=1).cpu().data[0].numpy()
    return img, heatmap, probs
