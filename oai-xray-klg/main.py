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
import argparse
import numpy as np
from torchvision.models import resnet34
import torch.nn as nn

from data import KneeGradingDatasetNew, KneeGradingDataSetUnsupervised, dataloaders_dict_init
from train import *
from utils import *
from attn_resnet.models.model_resnet import ResidualNet
# Arguments
parser = argparse.ArgumentParser(description='Networks training')
parser.add_argument('-d', '--data', action="store", dest="data_folder", type=str,
                    help="Training data directory", default='../data/OAI_processed/')
parser.add_argument('-dc', '--data-contents', action="store", dest="contents", type=str,
                    help="Validation data directory", default='../data/OAI_processed/')
parser.add_argument('-bs', '--batch-size', action="store", dest="batch_size", type=int,
                    help="Batch size for training", default=8)
parser.add_argument('-b', '--beta', action="store", dest="beta", type=float,
                    help="Beta coefficient for KLD loss", default=10)
parser.add_argument('-vi', '--val-interval', action="store", dest="val_interval", type=int, 
                    help="Number of epochs between validation", default=1)
parser.add_argument('-ep', '--epochs', action="store", dest="epochs", type=int, 
                    help="Number of epochs for training", default=15)
parser.add_argument('-lr', '--learning-rate', action="store", dest="learning_rate", type=float, 
                    help="Learning rate", default=0.0001)
parser.add_argument('-lm', '--load-model', action="store", dest="load_model", type=bool, 
                    help="Whether to load model", default=False)
parser.add_argument('-md', '--model-dir', action="store", dest="model_dir", type=str, 
                    help="Where to load the model", default=None)
parser.add_argument('-n', '--run-name', action="store", dest="run_name", type=str, 
                    help="Name for this run", default='default')
parser.add_argument('-m', '--model', action='store', dest='model_type', type=str,
                    default='baseline', choices=['baseline', 'CBAM', 'wide_resnet'],
                    help='Choose which model to train')
parser.add_argument('-au','--augmentation', action='store_true',
                    dest='augmentation', help='If apply augmentation on training')
parser.add_argument('-nau','--no-augmentation', action='store_false',
                    dest='augmentation', help='If apply augmentation on training')
parser.set_defaults(augmentation=False)
parser.add_argument('-do','--dropout',  action='store_true',
                    dest='dropout', help='If apply dropout from linear layer (Oulu lab)')
parser.add_argument('-ndo','--no-dropout',  action='store_false',
                    dest='dropout', help='If apply dropout from linear layer (Oulu lab)')
parser.set_defaults(dropout=False)
parser.add_argument('-fl','--freeze-layer',  action='store', type=int, default=0,
                    dest='freeze_layers', help='If only train layer number > this number')
parser.add_argument('-dt','--data-type',  action='store', default='float', type=str,
                    dest='data_type', help='Use integer or float.')
parser.add_argument('-dm', '--demo', action='store', dest='demo', type=str2bool,
                    default='yes')

def main():
    # Import parameters
    np.set_printoptions(suppress=True)
    args = parser.parse_args()
    # Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Computation device: ", device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    # Instantiate network
    print('Model configuration: model_type:{}; dropout:{}; Training augmentation:{}'\
          .format(args.model_type, args.dropout, args.augmentation))

    # KL Grade has 5 levels
    out_label = 5
    if args.model_type == 'baseline':
        net = resnet34(pretrained=True)
        num_ftrs = net.fc.in_features
        if args.dropout:
            net.fc = nn.Sequential(nn.Dropout(0.4),
                                   nn.Linear(num_ftrs, out_label))
        else:
            net.fc = nn.Sequential(nn.Linear(num_ftrs, out_label))
        net.avgpool = nn.AvgPool2d(28)
    elif args.model_type == 'CBAM':
        model = resnet34(pretrained=True)
        net = ResidualNet('ImageNet', 34, 1000, args.model_type)
        load_my_state_dict(net, model.state_dict())
        del model
        num_ftrs = net.fc.in_features
        if args.dropout:

            net.fc = nn.Sequential(nn.Dropout(0.4),
                                   nn.Linear(num_ftrs, out_label))
        else:
            net.fc = nn.Sequential(nn.Linear(num_ftrs, out_label))
    else:
        raise ValueError('Check the model_type arguments. Wrong input:', args.model_type)

    # print(net)

    ct = 0
    for child in net.children():
        # print(ct, child)
        ct += 1
        if ct < args.freeze_layers:
            for param in child.parameters():
                param.requires_grad = False

    net = net.to(device)

    # Dataloaders
    csv_dir_dict = args.contents
    data_dir_dict = args.data_folder
    print('Get contents from {}; Get data from {}'.format(csv_dir_dict, data_dir_dict))
    dataloaders_dict = dataloaders_dict_init(csv_dir_dict, data_dir_dict, args)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Model training
    train(net, dataloaders_dict, criterion, optimizer, args.val_interval, args.epochs, args.load_model, args.model_dir,
          args.run_name, args, device)


if __name__ == "__main__":
    main()
