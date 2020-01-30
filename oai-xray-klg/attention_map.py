from __future__ import print_function
import sys
import matplotlib.pyplot as plt
from data import KneeGradingDatasetNew
from tqdm import tqdm
import numpy as np
import argparse
import os
import itertools
import torch
import torch.nn as nn
from augmentation import CenterCrop
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time
import pandas as pd
import sys
from utils import gradcam_resnet

parser = argparse.ArgumentParser()
parser.add_argument('-lm', '--load-model', action='store', dest='load_model',
                    default=None, type=str)
if __name__ == '__main__':
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    job_number = int(1)
<<<<<<< HEAD
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new4/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new4/'
    test = pd.read_csv(summary_path + 'test_proj15.csv')#.sample(n=50).reset_index() # split train - test set.
=======
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new3/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new3/'
    test = pd.read_csv(summary_path + 'test.csv')#.sample(n=50).reset_index() # split train - test set.
>>>>>>> 3daec0b3c4c24e2e7ea2c7d701ac6af1c7323480

    start_test = 0
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDatasetNew(test, HOME_PATH, tensor_transform_test, 'float')

    test_loader = data.DataLoader(dataset_test, batch_size=8)
    print('Test data:', len(dataset_test))
    # Network
    if USE_CUDA:
        net = torch.load(args.load_model)
    else:
        net = torch.load(args.load_model, map_location='cpu')
    if USE_CUDA:
        net = nn.DataParallel(net)
        net.to(device)
    net.eval()
    print('############### Model Finished ####################')
    print(test.head())
    path_name = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix'
    save_dir = '/'.join(args.load_model.split('/')[:-1] + ['attention_map'])
    for idx, row in test.iterrows():
        month = row['Visit']
        pid = row['ID']
        target = int(row['KLG'])
        side = int(row['SIDE'])
        if side == 1:
            fname = '{}_{}_RIGHT_KNEE.hdf5'.format(pid, month)
            path = os.path.join(HOME_PATH, month, fname)
        elif side == 2:
            fname = '{}_{}_LEFT_KNEE.hdf5'.format(pid, month)
            path = os.path.join(HOME_PATH, month, fname)
        img, heatmap, probs = gradcam_resnet(path, net, tensor_transform_test, use_cuda=USE_CUDA, label=target)
        pred = probs.argmax()
        plt.figure(figsize=(7, 7))
        img = np.array(img)
        plt.imshow(img[1, :, :], cmap=plt.cm.Greys_r)
        plt.imshow(heatmap, cmap=plt.cm.jet, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        plt.title('{} KLG: {}; Prediction: {}'.format(side, target, pred))
        if pred == target:
            subfolder = 'correct_pred'
            target =str(target)
            if not os.path.exists(os.path.join(save_dir, subfolder,target, target)):
                os.makedirs(os.path.join(save_dir, subfolder,target, target))
            plt.savefig(os.path.join(save_dir, subfolder, target, target, fname.replace('.hdf5', '.png')), dpi=300)
        else:
            target = str(target)
            pred = str(pred)
            subfolder = 'wrong_pred'
            if not os.path.exists(os.path.join(save_dir, subfolder,target, pred)):
                os.makedirs(os.path.join(save_dir, subfolder, target, pred))
            plt.savefig(os.path.join(save_dir, subfolder, target, pred, fname.replace('.hdf5', '.png')), dpi=300)


