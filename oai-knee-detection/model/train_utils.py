from torch.autograd import Variable
import gc
import torch.nn as nn
import torch
import time
import numpy as np
import os
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score
from tqdm import tqdm
import pandas as pd
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


def train_iterations(net, optimizer, train_loader, val_loader,
                     criterion, max_ep, args,use_cuda=True, iterations=1000,
                     log_dir=None, model_dir=None, is_classifier=False):
    '''
    :param net:
    :param optimizer:
    :param train_loader:
    :param val_loader:
    :param criterion:
    :param max_ep:
    :param use_cuda:
    :param iterations:
    :param log_dir:
    :param model_dir:
    :return:
    '''
    net.train(True)
    n_batches = len(train_loader)
    train_losses = []
    val_losses = []
    val_iou = []
    pre_model = None # best model
    train_start = time.time()
    for epoch in range(max_ep):
        running_loss = 0.0
        for i, (batch, targets, knee_labels, names) in enumerate(train_loader):
            optimizer.zero_grad()
            targets = torch.stack(targets).transpose(0, 1)
            # forward + backward + optimize
            if use_cuda:
                labels = Variable(targets.float().cuda())
                inputs = Variable(batch.cuda())
            else:
                labels = Variable(targets.float())
                inputs = Variable(batch)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            #for param in net.parameters():
            #    param.grad.data.clamp_(-1, 1)
            optimizer.step()
            if not torch.isnan(loss).any():
                running_loss += loss.item()
            log_output = '[%d | %d, %5d / %d] | Running loss: %.3f / loss %.3f ]' % (epoch + 1, max_ep, i + 1,
                                                                            n_batches, running_loss / (i + 1),
                                                                            loss.item())
            print(log_output)
            with open(log_dir,'a+') as f:
                f.write(log_output + '\n')

            if (i + 1) % iterations == 0 or (i + 1) == n_batches:
                val_loss, all_names, all_labels, all_preds = validate_epoch(net, criterion, val_loader)
                iou_l, iou_r = iou(all_labels, all_preds)
                val_losses.append(val_losses)
                val_iou.append((iou_l.mean() + iou_r.mean()) / 2)
                train_losses.append(running_loss / (i + 1))
                log_output = '[Epoch %d | Val Loss %.5f | Train Loss %.5f | Left IOU %.3f | Right IOU %.3f | Mean IOU %.3f ]' % (epoch + 1,
                                                                                                                 val_loss,
                                                                                                                 running_loss / (i + 1),
                                                                                                                 iou_l.mean(),iou_r.mean(), (iou_l.mean() + iou_r.mean()) / 2)
                print(log_output)
                with open(log_dir, 'a+') as f:
                    f.write(log_output + '\n')
                if pre_model is None or val_iou[-1] > pre_model:
                    print('Save the model at epoch {}, iter {}'.format(epoch + 1, i + 1))
                    snapshot = model_dir + '/' + 'epoch_{}.pth'.format(epoch + 1)
                    torch.save(net, snapshot)
                    pre_model = val_iou[-1]
            gc.collect()
        gc.collect()
    print('Training takes {} seconds'.format(time.time() - train_start))


def validate_epoch(net, criterion, val_loader):
    all_names = []
    net.eval()
    running_loss = 0.0
    n_batches = len(val_loader)
    all_labels = []
    all_preds = []
    bar = tqdm(total=len(val_loader), desc='Processing', ncols=90)
    USE_CUDA = torch.cuda.is_available()
    for i, (batch, targets, knee_labels, names) in enumerate(val_loader):
        targets = torch.stack(targets).transpose(0, 1)
        # forward + backward + optimize
        if USE_CUDA:
            labels = Variable(targets.float()).squeeze().cuda()
            inputs = Variable(batch).cuda()
        else:
            labels = Variable(targets.float()).squeeze()
            inputs = Variable(batch)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if not torch.isnan(loss).any():
            running_loss += loss.item()
        gc.collect()
        all_names.extend(names)
        all_labels.append(labels.data.cpu().numpy())
        all_preds.append(outputs.data.cpu().numpy())
        bar.update()
    net.train(True)
    return running_loss / n_batches, all_names, all_labels, all_preds


def metrics_iou(boxA,boxB):
    '''
    Two numpy array as input
    :param boxA:
    :param boxB:
    :return:
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    # first compute left
    xA = np.maximum(boxA[:, 0], boxB[:, 0])
    yA = np.maximum(boxA[:, 1], boxB[:, 1])
    xB = np.minimum(boxA[:, 2], boxB[:, 2])
    yB = np.minimum(boxA[:, 3], boxB[:, 3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_l = interArea / (boxAArea + boxBArea - interArea)

    # compute right side
    xA = np.maximum(boxA[:, 4], boxB[:, 4])
    yA = np.maximum(boxA[:, 5], boxB[:, 5])
    xB = np.minimum(boxA[:, 6], boxB[:, 6])
    yB = np.minimum(boxA[:, 7], boxB[:, 7])

    # compute the area of intersection rectangle
    interArea = (xB - xA ) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:, 6] - boxA[:, 4]) * (boxA[:, 7] - boxA[:, 5])
    boxBArea = (boxB[:, 6] - boxB[:, 4]) * (boxB[:, 7] - boxB[:, 5])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_r = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value

    return iou_l,iou_r


def iou(all_labels, all_preds, all_knee_labels=None):
    '''
    compute left iou and right iou from given coordinates
    :param all_labels:
    :param all_preds:
    :return:
    '''
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    if all_knee_labels is not None:
        try:
            all_knee_labels = np.concatenate(all_knee_labels, axis=0).astype(int)
        except ValueError as e:
            pass
        all_labels = all_labels[all_knee_labels == 1, :]
        all_preds = all_preds[all_knee_labels == 1, :]
    df = [all_labels, all_preds]
    df = np.hstack(df)
    df = pd.DataFrame(df)
    boxA = df.values[:, :8]
    boxB = df.values[:, 8:]
    iou_l, iou_r = metrics_iou(boxA, boxB)
    return iou_l,iou_r


def compute_binary_acc(all_knee_labels, all_knee_preds):
    try:
        all_knee_labels = np.concatenate(all_knee_labels, axis=0).astype(int)
    except ValueError as e:
        pass
    try:
        all_knee_preds = np.concatenate(all_knee_preds, axis=0)
    except ValueError as e:
        pass
    all_knee_preds[all_knee_preds > 0.5] = 1
    all_knee_preds[all_knee_preds <= 0.5] = 0
    all_knee_preds = all_knee_preds.transpose()
    correct = (all_knee_preds == all_knee_labels).sum()
    return correct / all_knee_labels.shape[0]