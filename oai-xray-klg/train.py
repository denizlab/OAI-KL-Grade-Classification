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
import torch
import time
from tensorboardX import SummaryWriter
from utils import *
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.autograd import Variable
import time
import gc


def train(model, dataloaders_dict, criterion, optimizer, val_interval, num_epoch, load_model_bool, model_dir, run_name, args, device):
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    exp_dir = 'runs'
    if args.demo:
        run_name += '_demo'
    run_name += '_' + current_time
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(exp_dir + '/' + run_name):
        os.mkdir(exp_dir + '/' + run_name)
        with open(exp_dir + '/' + run_name + '/config.txt', 'w') as f:
            f.write(str(args))
        print(f'New directory {run_name} created')
    save_dir = exp_dir + '/' + run_name
    # Parameters used between epochs
    starting_epoch = 0
    epoch_CE_loss = {'train' : None, 'val' : None}
    epoch_loss = {'train' : None, 'val' : None}
    epoch_acc = {'train' : None, 'val' : None}
    epoch_avg_acc = {'train': None, 'val': None}
    min_loss_checkpoint = {'val_loss' : None, 'best_dir' : None,
                           'lastest_dir' : None, 'val_overall_acc':None,
                           'kappa_score': None, 'val_avg_acc': None,
                           'val_kappa': None, 'val_mse': None}
    # Load (saved) model to device
    if load_model_bool:
        model, starting_epoch = load_model(model_dir, min_loss_checkpoint, device)
        progress_info_writer(save_dir, "Load model from {}".format(model_dir))

    # Tensorboard logging
    writer_CE_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/CE/train')
    writer_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/train')
    writer_CE_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/CE/val')
    writer_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/val')

    for epoch in range(starting_epoch, starting_epoch + num_epoch):
        for phase in ['train', 'val']: # Either train or val
            if phase == 'val' and epoch % val_interval != 0: # Not the epoch for val
                epoch_loss['val'] = None
                epoch_acc['val'] = None
                continue

            num_batch = len(dataloaders_dict[phase])
            print('Number of Batch:', num_batch)
            print('Length Of Dataset', len(dataloaders_dict[phase].dataset))
            running_loss = 0.0
            running_CE_loss = 0.0
            running_KLD_loss = 0.0
            running_corrects = 0
            num_sample = 0
            if phase == 'train': model.train()  # Set model to training mode
            else:                model.eval()   # Set model to eval

            # Iterate batches
            bar = tqdm(total=len(dataloaders_dict[phase]), desc='Processing', ncols=90)
            truths = []
            pred_labels = []
            for batch, (inputs, labels, _) in enumerate(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                num_sample += labels.shape[0]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)


                    preds = torch.argmax(outputs, 1)
                    if phase == 'train': # Update in train phase
                        loss.backward()
                        optimizer.step()
                truths.append(labels.reshape(-1, 1).data.cpu().numpy())
                pred_labels.append(preds.reshape(-1, 1).data.cpu().numpy())
                
                running_CE_loss += loss.item() * inputs.shape[0]
                running_loss += loss.item() * inputs.shape[0] # Batch loss
                running_corrects += torch.sum(preds == labels).item()
                bar.update(1)
                if phase == 'train':
                    #progress_info(epoch, starting_epoch + num_epoch, batch, num_batch, batch_speed, epoch_loss, epoch_acc)
                    if batch % 200 == 0 or batch == num_batch - 1:
                        avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
                        output = 'Epoch {}: {:} CE Loss:{:.4g}; Accuracy:{}/{}={:.4g}; Average Acc:{:4g}; Kappa:{:4g}; MSE:{:4g}'\
                              .format(epoch, phase,running_CE_loss / num_sample,
                                      running_corrects,
                                      num_sample,
                                      running_corrects / num_sample,
                                      avg_acc,
                                      kappa,
                                      mse)
                        progress_info_writer(save_dir, output)
                if phase == 'val' and batch == num_batch - 1:
                    avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
                    output1 = '*' * 20
                    output = 'Epoch {}: {:} CE Loss:{:.4g}; Accuracy:{}/{}={:.4g}; Average Acc:{:4g}; Kappa:{:4g}; MSE:{:4g}' \
                        .format(epoch, phase, running_CE_loss / num_sample,
                                running_corrects,
                                num_sample,
                                running_corrects / num_sample,
                                avg_acc,
                                kappa,
                                mse)
                    progress_info_writer(save_dir, output1, output, str(cm), output1)
            avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
            epoch_CE_loss[phase] = running_CE_loss / len(dataloaders_dict[phase].dataset)
            epoch_loss[phase] = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc[phase] = running_corrects / len(dataloaders_dict[phase].dataset)
            epoch_avg_acc[phase] = avg_acc
            print('###################')
            print( running_corrects / len(dataloaders_dict[phase].dataset), running_corrects / num_sample)
            print('###################')
            if phase == 'train':
                writer_train.add_scalar('loss', epoch_loss['train'], epoch)
                writer_train.add_scalar('accuracy', epoch_acc['train'], epoch)
            else:
                save_model(min_loss_checkpoint, epoch_loss, epoch_acc, epoch_avg_acc, model.cpu(), epoch, save_dir)
                model.to(device)
                writer_val.add_scalar('loss', epoch_loss['val'], epoch)
                writer_val.add_scalar('accuracy', epoch_acc['val'], epoch)

    writer_train.close()
    writer_val.close()
    writer_CE_train.close() 
    writer_CE_val.close()
    writer_KLD_val.close()
    print('Training Complete!' + ' ' * 10)

def validate_epoch(net, val_loader, criterion, use_cuda = True,loss_type='CE'):
    net.train(False)
    running_loss = 0.0
    sm = nn.Softmax(dim=1)

    truth = []
    preds = []
    bar = tqdm(total=len(val_loader), desc='Processing', ncols=90)
    names_all = []
    n_batches = len(val_loader)
    for i, (batch, targets, names) in enumerate(val_loader):
        if use_cuda:
            if loss_type == 'CE':
                labels = Variable(targets.long().cuda())
                inputs = Variable(batch.cuda())
            elif loss_type == 'MSE':
                labels = Variable(targets.float().cuda())
                inputs = Variable(batch.cuda())
        else:
            if loss_type == 'CE':
                labels = Variable(targets.float())
                inputs = Variable(batch)
            elif loss_type == 'MSE':
                labels = Variable(targets.float())
                inputs = Variable(batch)

        outputs = net(inputs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        if loss_type =='CE':
            probs = sm(outputs).data.cpu().numpy()
        elif loss_type =='MSE':
            probs = outputs
            probs[probs < 0] = 0
            probs[probs > 4] = 4
            probs = probs.view(1,-1).squeeze(0).round().data.cpu().numpy()
        preds.append(probs)
        truth.append(targets.cpu().numpy())
        names_all.extend(names)
        running_loss += loss.item()
        bar.update(1)
        gc.collect()
    gc.collect()
    bar.close()
    if loss_type =='CE':
        preds = np.vstack(preds)
    else:
        preds = np.hstack(preds)
    truth = np.hstack(truth)
    return running_loss / n_batches, preds, truth, names_all
