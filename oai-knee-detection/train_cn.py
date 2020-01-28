from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import torch
import argparse
import gc
from torchvision.transforms import ToPILImage, ToTensor, RandomRotation, RandomHorizontalFlip, \
    Compose, Resize
from model.model_hg import HourglassNet
from model.model_hg2 import PoseNet
from model.dataloader import KneeDetectionDataset
import os
import time
import torch.nn as nn
import torch.utils.data as data

#from albumentations import (
#    RandomBrightnessContrast, Compose, RandomGamma, HueSaturationValue,
#    RGBShift, MotionBlur, Blur, GaussNoise, ChannelShuffle, Normalize
#)
from model.train_utils import train_centernet, str2bool
from model.centernet_utils import load_my_state_dict
import torch.optim as optim
import warnings
from torchvision import transforms
import pandas as pd
warnings.filterwarnings("ignore")


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-sd', '--save-dir', type=str, dest='save_dir', default='run_test/')
    args.add_argument('-m', '--model', type=str, dest='model_type', default='HG2',
                      choices=['HG', 'HG2'])
    args.add_argument('-ns', '--n-stacks', type=int, dest='num_stacks', default=2)
    args.add_argument('-nc', '--n-classes', type=int, dest='num_classes', default=8)
    args.add_argument('-nf', '--n-features', type=int, dest='num_features', default=256)
    args.add_argument('-bs', '--batch_size', type=int, dest='batch_size', default=2)
    args.add_argument('-e', '--epoch', type=int, dest='epoch', default=30)
    args.add_argument('-lf', '--loss-func', type=str,
                      dest='loss_type', default='FL', choices=['BCE', 'FL', 'MSE'],
                      help='Loss function for supervising detection')
    args.add_argument('-a', '--alpha', type=int, dest='alpha', default=2)
    args.add_argument('-b', '--beta', type=int, dest='beta', default=4)
    args.add_argument('-db', '--debug', type=str2bool, dest='debug', default='no')
    args.add_argument('-tp', '--transform-prob', type=float, dest='prob', default=0.2)
    args.add_argument('-g', '--gamma', type=float, dest='gamma', default=1, help='Weights for regression loss')
    args.add_argument('-lr', '--learning-rate', type=float, dest='lr',
                      help='learning rate', default=1e-3)
    args.add_argument('-pt', '--pre-train', type=str2bool, dest='pre_train',
                      help='learning rate', default='yes')
    return args.parse_args()


def main():
    args = parse_args()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_name = 'model_{}_stack_{}_feat_{}_g_{}_{}_' if args.prob <= 0 else 'model_aug_{}_stack_{}_feat_{}_g_{}_{}_'
    save_dir = args.save_dir + model_name.format(args.model_type, args.num_stacks, args.num_features, args.gamma, args.loss_type)\
               + current_time + '/'
    augmentation = args.prob > 0
    # Augmentation
    # training
    #albu_list = [RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
    #             RandomGamma(p=0.2), HueSaturationValue(p=0.3), RGBShift(p=0.3), MotionBlur(p=0.1), Blur(p=0.1),
    #             GaussNoise(var_limit=(20, 100), p=0.2),
    #             ChannelShuffle(p=0.2),
                 #Normalize(mean=[145.3834, 136.9748, 122.7390], std=[95.1996, 94.6686, 85.9170])
    #             ]
    # val
    tensor_transform_val = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.float(),
    ])
    # get dataloader
    train_contents = './bounding_box_oulu/train_with_neg_no_replacement.csv'
    train_df = pd.read_csv(train_contents)  # .sample(n=200).reset_index()
    val_contents = './bounding_box_oulu/val_with_neg_no_replacement.csv'
    val_df = pd.read_csv(val_contents)  # .sample(n=100).reset_index()
    if args.debug:
        train_df = train_df.sample(n=100)#.reset_index()
        val_df = val_df.sample(n=20)#.reset_index()

    # Create dataset objects
    train_dataset = KneeDetectionDataset(train_df, tensor_transform_val,
                                         stage='train', mode='HG',
                                         save_dir=save_dir,
                                         debug=args.debug,
                                         reshape=896,
                                         training=True,
                                         augmentation=augmentation)
    dev_dataset = KneeDetectionDataset(val_df, tensor_transform_val,
                                       stage='train', mode='HG',
                                       save_dir=save_dir,
                                       debug=args.debug,
                                       reshape=896,
                                       training=False,
                                       augmentation=augmentation)

    BATCH_SIZE = args.batch_size
    # Create data generators - they will produce batches
    # transform not using yet
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = data.DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # Gets the GPU if there is one, otherwise the cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print('Running on', torch.cuda.get_device_name(device), 'x', torch.cuda.device_count())


    if args.model_type == 'HG':
        model = HourglassNet(nStacks=args.num_stacks, nModules=1, nFeat=args.num_features, nClasses=args.num_classes)
        model.cuda()
    elif args.model_type == 'HG2':
        model = PoseNet(nstack=args.num_stacks, inp_dim=args.num_features,
                        oup_dim=args.num_classes)
        model = model.cuda()
        if args.num_stacks <= 16 and args.pre_train:
            save = torch.load('./weights/checkpoint_2hg.pt')
        elif args.pre_train:
            save = torch.load('./weights/checkpoint_8hg.pt')
        save = save['state_dict']
        load_my_state_dict(model, save)
        del save

    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], gamma=0.5)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)
    # save configuration
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        with open(save_dir + 'config.txt', 'w') as f:
            f.write(str(args))
    data_loader = {'train': train_loader, 'val': dev_loader}
    train_centernet(model, optimizer, scheduler, data_loader, save_dir, args)


if __name__ == '__main__':
    main()
