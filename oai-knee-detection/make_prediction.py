from torch.utils.data import Dataset, DataLoader
import cv2
from model.dataloader import KneeDetectionDataset
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import pandas as pd
from model.train_utils import str2bool
from tqdm import tqdm
from model.val_utils import evaluate_model
import os


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-lm', '--load-model', type=str, dest='load_model', default=None)
    args.add_argument('-t', '--threshold', type=float, dest='threshold', default=0)
    args.add_argument('-db', '--debug', type=str2bool, dest='debug', default='yes')
    return args.parse_args()


def main():
    args = parse_args()
    val_contents = './bounding_box_oulu/test_with_neg_no_replacement.csv'
    debug = args.debug
    save_dir = args.load_model.split('/')
    fig_save_dir = '/'.join(save_dir[:-1]) + '/fig'
    save_dir = '/'.join(save_dir[:-1])
    if not os.path.exists(fig_save_dir):
        os.mkdir(fig_save_dir)

    test = pd.read_csv(val_contents)
    if debug:
        pos = test.iloc[:10]
        neg = test.iloc[-5:]
        test = pos.append(neg)

    tensor_transform_val = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.float(),
    ])
    test_dataset = KneeDetectionDataset(test, tensor_transform_val, stage='test', mode='HG',
                                        training=False, debug=args.debug)
    load_model = args.load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = torch.load(load_model)
    else:
        model = torch.load(load_model, map_location='cpu')

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)

    evaluate_model(model, test_loader, save_dir, fig_save_dir, device, args)


if __name__ == '__main__':
    main()