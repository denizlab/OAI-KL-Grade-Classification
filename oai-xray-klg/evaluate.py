from __future__ import print_function
import sys

from tqdm import tqdm
import numpy as np
import argparse
import os

from attn_resnet.models.model_resnet import *
from data import KneeGradingDatasetNew
from augmentation import *
#from utils import load_model
from train import validate_epoch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import gc

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score
from torchvision.models import resnet34
import time
import pickle
import pandas as pd
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lm', '--load-model', type=str, action='store', dest='load_model', default=None)
parser.add_argument('-hp', '--home-path', type=str, action='store', dest='home_path', default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    HOME_PATH = args.home_path
    test = pd.read_csv(HOME_PATH + 'test.csv')# .sample(n=20).reset_index() # split train - test set.

    start_test = 0
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDatasetNew(test, HOME_PATH,tensor_transform_test, False)

    test_loader = data.DataLoader(dataset_test, batch_size=6)
    print('Test data:', len(dataset_test))
    load_model = args.load_model
    print(load_model)
    if USE_CUDA:
        model = torch.load(load_model, map_location='cpu')
    else:
        model = torch.load(load_model)

    if USE_CUDA:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    print("model")
    print(model)
    model.eval()
    print('############### Model Finished ####################')

    test_losses = []
    test_mse = []
    test_kappa = []
    test_acc = []


    test_started = time.time()

    test_loss, probs, truth, file_names = validate_epoch(model, test_loader, criterion, use_cuda=USE_CUDA)
    preds = probs.argmax(1)
    # set log file
    output_dir = args.load_model.split('/')
    output_dir = output_dir[:-2]
    output_dir = '/'.join(output_dir)
    log_dir = open(output_dir + '/output.txt', 'w')
    # Validation metrics
    cm = confusion_matrix(truth, preds)
    print('Confusion Matrix:\n', cm.diagonal() / cm.sum(axis=1), file=log_dir)
    print('Confusion Matrix:\n', cm, file=log_dir)

    kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
    acc = np.round(np.mean(cm.diagonal().astype(float) / cm.sum(axis=1)), 4)
    print('Oulu Acc {}'.format(acc))
    print(cm.diagonal().astype(float) / cm.sum(axis=1))
    # mse
    mse = np.round(mean_squared_error(truth, preds), 4)
    test_time = np.round(time.time() - test_started, 4)
    test_losses.append(test_loss)
    test_mse.append(mse)
    test_acc.append(acc)
    test_kappa.append(kappa)

    gc.collect()
    print('Test losses {}; Test mse {}; Test acc {}; Test Kappa {};\n'.format(test_loss,test_mse,test_acc,kappa), file=log_dir)
    print('Testing took:', time.time() - test_started, 'seconds\n', file=log_dir)


    #  output all results
    eval_result = pd.DataFrame(probs)
    eval_result['preds'] = preds
    eval_result['truth'] = truth
    eval_result['file_names'] = file_names
    eval_result.to_csv(output_dir + '/eval_result.csv', index=False)
    # confusion matrix
    cm = pd.DataFrame(cm)
    cm.to_csv(output_dir + '/cm_result.csv', index=True)
    log_dir.close()




