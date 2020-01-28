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
from model.centernet_utils import criterion, save_model
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


def train_centernet(net, optimizer, scheduler,data_loader, save_dir, args):
    best_loss = 1e6
    train_loader, dev_loader = data_loader['train'], data_loader['val']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(args.epoch):
        total_loss, final_loss = train_model(
            save_dir=save_dir,
            model=net,
            epoch=epoch,
            train_loader=train_loader,
            optimizer=optimizer,
            args=args
        )

        best_loss, eval_total_loss, eval_final_loss, eval_clf_loss, eval_regr_loss, num_knees_total, num_knees_detect = evaluate_model(
            model=net,
            epoch=epoch,
            dev_loader=dev_loader,
            device=device,
            best_loss=best_loss,
            save_dir=save_dir,
            args=args
        )

        scheduler.step()
        with open(save_dir + 'log.txt', 'a+') as f:
            print('Eval {}: Total loss {:.2f}; Final Loss {:.2f}; Eval total loss {:.2f}; Eval final loss {:.2f}; Clf loss {:.2f}; Regr loss: {:.2f};Knee detected {}; Knee Total {}'\
                  .format(epoch, total_loss,
                          final_loss,
                          eval_total_loss,
                          eval_final_loss,
                          eval_clf_loss,
                          eval_regr_loss,
                          num_knees_detect,
                          num_knees_total), file=f)


def train_model(save_dir, model, epoch, train_loader, optimizer,
                args):
    model.train()
    total_batches = len(train_loader)
    stack_losses = np.zeros(args.num_stacks) if 'res' not in args.model_type else np.zeros(1)
    epoch_loss = 0
    clf_losses = 0
    regr_losses = 0
    EVAL_INTERVAL = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bar = tqdm(total=len(train_loader), desc='Processing', ncols=90)
    for (batch_idx, (img_batch, mask_batch, heatmap_batch, width_batch, height_batch, _, _)) in enumerate(train_loader):
        # print('Train loop:', img_batch.shape)
        img_batch = img_batch.float().to(device)
        mask_batch = mask_batch.float().to(device)
        heatmap_batch = heatmap_batch.float().to(device)
        width_batch = width_batch.float().to(device)
        height_batch = height_batch.float().to(device)
        output = model(img_batch)
        loss = 0
        for idx, stack_output in enumerate(output):
            stack_output = stack_output['hm'] if type(stack_output) is dict else stack_output
            loss_turn, clf_loss, regr_loss = criterion(stack_output, mask_batch,
                                                       height_batch,
                                                       width_batch,
                                                       heatmap_batch,
                                                       size_average=True, loss_type=args.loss_type,
                                                       alpha=args.alpha, beta=args.beta,
                                                       gamma=args.gamma)

            loss += loss_turn
            stack_losses[idx] += loss_turn.item()
            if idx == len(output) - 1:
                epoch_loss += loss_turn.item()

        gc.collect()
        # add final stack  value
        clf_losses += clf_loss.item()
        regr_losses += regr_loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.update(1)
        #print(stack_losses.tolist(), clf_losses, regr_losses)
        if (batch_idx + 1) % EVAL_INTERVAL == 0 or batch_idx == total_batches - 1:
            total_loss = np.mean(stack_losses)
            with open(save_dir + 'log.txt', 'a+') as f:
                line = '{} | {} | Total Loss: {:.3f}, Stack Loss:{:.3f}; Clf loss: {:.3f}; Regr loss: {:.3f}.\n' \
                    .format(batch_idx + 1, total_batches, total_loss / (batch_idx + 1),
                            stack_losses[-1] / (batch_idx + 1), clf_losses / (batch_idx + 1),
                            regr_losses / (batch_idx + 1))
                f.write(line)

    total_loss = np.mean(stack_losses)
    final_loss = stack_losses[-1]

    return total_loss / len(train_loader), final_loss / len(train_loader)


def evaluate_model(model, epoch, dev_loader, device, best_loss, save_dir, args):
    model.eval()
    bar = tqdm(total=len(dev_loader), desc='Processing', ncols=90)
    with torch.no_grad():
        stack_loss = np.zeros(args.num_stacks) if 'res' not in args.model_type else np.zeros(1)
        clf_losses = 0
        regr_losses = 0
        num_knees_total = 0
        num_knees_detect = 0
        for img_batch, mask_batch, heatmap_batch, height_batch, width_batch, _, _ in dev_loader:
            img_batch = img_batch.float().to(device)
            mask_batch = mask_batch.float().to(device)
            heatmap_batch = heatmap_batch.float().to(device)
            height_batch = height_batch.float().to(device)
            width_batch = width_batch.float().to(device)
            output = model(img_batch)


            for idx, stack_output in enumerate(output):
                stack_output = stack_output['hm'] if type(stack_output) is dict else stack_output
                loss_turn, clf_loss, regr_loss = criterion(stack_output,
                                                           mask_batch,
                                                           height_batch,
                                                           width_batch,
                                                           heatmap_batch,
                                                           size_average=True, loss_type=args.loss_type,
                                                           alpha=args.alpha, beta=args.beta, gamma=args.gamma)
                stack_loss[idx] += loss_turn.item()
            clf_losses += clf_loss.item()
            regr_losses += regr_loss.item()
            # inferences
            output = output[-1].data.cpu().numpy()
            mask_batch = mask_batch.data.cpu().numpy()
            for out, mask in zip(output, mask_batch):
                coords = extract_coords(prediction=out, threshold=0)
                num_knees_total += mask.sum()
                num_knees_detect += len(coords)
            bar.update(1)
    total_loss = np.mean(stack_loss)
    final_loss = stack_loss[-1]
    total_loss /= len(dev_loader)
    stack_loss /= len(dev_loader)
    final_loss /= len(dev_loader)
    clf_losses /= len(dev_loader)
    regr_losses /= len(dev_loader)

    if final_loss < best_loss:
        best_loss = final_loss
    save_model(model, save_dir, epoch)

    return best_loss, total_loss, final_loss, clf_losses, regr_losses, num_knees_total, num_knees_detect


def extract_coords(prediction, threshold=0):

    logits = prediction[0].copy()
    regr_output = prediction[1:]
    logits -= threshold
    points = np.argwhere(logits > 0)
    done = len(points) <= 0
    print('Logits stats:', logits.max(), logits.mean())
    coords = []
    while not done:
        coords = []
        for r, c in points:
            regr_dict = dict()
            regr_dict['width'] = regr_output[0, r, c]
            regr_dict['height'] = regr_output[1, r, c]
            coords.append(regr_dict)
            coords[-1]['x'] = r
            coords[-1]['y'] = c
            coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords = clear_duplicates(coords)
        done = len(coords) >= 1
        logits += 0.1
        # reset
        points = np.argwhere(logits > 0)

    print('Found', len(coords))
    # right knee goes first
    # print(coords)
    if len(coords) == 2:
        coords = sorted(coords, key=lambda x: x['y'])
        right = get_box(coords[0])
        left = get_box(coords[1])
        pred_output = np.array(left + right) / 224
        pred_label = 1
    elif len(coords) == 1:
        coord = coords[0]
        x, y = coord['x'], coord['y']
        knee = get_box(coord)
        if y <= 112: # right knee found
            pred_output = [-1] * 4 + knee
            pred_label = 2
        else: #left knee found
            pred_output = knee + [-1] * 4
            pred_label = 3
    else:
        pred_output = [-1] * 8
        pred_label = 0
    return coords, pred_output, pred_label


def get_box(coord):
    x = coord['x']
    y = coord['y']
    width = coord['width']
    height = coord['height']
    x1 = x - width
    x2 = x + width
    y1 = y - height
    y2 = y + height
    return [y1, x1, y2, x2]


def clear_duplicates(coords, distance_threshold=16):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y']])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < distance_threshold:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

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
    val_accs = []
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
                knee_labels = Variable(knee_labels.float().cuda())
                labels = Variable(targets.float().cuda())
                inputs = Variable(batch.cuda())
            else:
                knee_labels = Variable(knee_labels.float())
                labels = Variable(targets.float())
                inputs = Variable(batch)
            if 'resnet' in args.model:
                knee_pred, outputs = net(inputs)
                loss, bce, mse = criterion(knee_pred, outputs, knee_labels, labels)
                loss.backward()
            elif 'hourglass' in args.model:
                outputs = net(inputs)
                for out in outputs:
                    print('Output:', out.shape)
                print(targets.shape, knee_labels.shape)

            #for param in net.parameters():
            #    param.grad.data.clamp_(-1, 1)
            optimizer.step()
            if not torch.isnan(loss).any():
                running_loss += loss.item()
            log_output = '[%d | %d, %5d / %d] | Running loss: %.3f / loss %.3f | BCE: %.3f | MSE: %.3f]' % (epoch + 1, max_ep, i + 1,
                                                                            n_batches, running_loss / (i + 1),
                                                                            loss.item(), bce, mse)
            print(log_output)
            with open(log_dir,'a+') as f:
                f.write(log_output + '\n')

            if (i + 1) % iterations == 0 or (i + 1) == n_batches:
                val_loss, all_names, all_labels, all_preds, all_knee_labels, all_knee_preds = validate_epoch(net, criterion, val_loader, use_cuda, is_classifier)
                iou_l, iou_r = iou(all_labels, all_preds, all_knee_labels)
                acc = compute_binary_acc(all_knee_labels, all_knee_preds)
                val_losses.append(val_losses)
                val_accs.append(val_accs)
                val_iou.append((iou_l.mean() + iou_r.mean()) / 2)
                train_losses.append(running_loss / (i + 1))
                log_output = '[Epoch %d | Val Loss %.5f | Train Loss %.5f | Left IOU %.3f | Right IOU %.3f | Mean IOU %.3f | Accuracy: %3.f]' % (epoch + 1,
                                                                                                                 val_loss,
                                                                                                                 running_loss / (i + 1),
                                                                                                                 iou_l.mean(),iou_r.mean(), (iou_l.mean() + iou_r.mean()) / 2, acc)
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


def validate_epoch(net, criterion, val_loader, use_cuda, is_classifier=False):
    all_names = []
    net.eval()
    running_loss = 0.0
    n_batches = len(val_loader)
    all_labels = []
    all_preds = []
    all_knee_preds = []
    all_knee_labels = []
    bar = tqdm(total=len(val_loader), desc='Processing', ncols=90)
    for i, (batch, targets, knee_labels, names) in enumerate(val_loader):
        targets = torch.stack(targets).transpose(0, 1)
        # forward + backward + optimize
        if use_cuda:
            knee_labels = Variable(knee_labels.float().cuda()).squeeze()
            labels = Variable(targets.float().cuda()).squeeze()
            inputs = Variable(batch.cuda())
        else:
            knee_labels = Variable(knee_labels.float()).squeeze()
            labels = Variable(targets.float()).squeeze()
            inputs = Variable(batch)
        if is_classifier:
            knee_pred, outputs = net(inputs)
            knee_pred = knee_pred.squeeze()
            # print(knee_pred.shape, outputs.shape, knee_labels.shape, )
            all_knee_labels.append(knee_labels.data.cpu().numpy())
            all_knee_preds.append(knee_pred.data.cpu().numpy())
            loss, _, _ = criterion(knee_pred, outputs, knee_labels, labels)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        if not torch.isnan(loss).any():
            running_loss += loss.item()
        gc.collect()
        all_names.extend(names)
        all_labels.append(labels.data.cpu().numpy())
        all_preds.append(outputs.data.cpu().numpy())

    net.train(True)
    if is_classifier:
        return running_loss / n_batches, all_names, all_labels, all_preds, all_knee_labels, all_knee_preds
    else:
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