from model.model import ResNet, MSELoss
from model.dataloader import *
from model.train_utils import *
from model.augmentation import Rotate, RandomLRFlip
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
import time
import torch.optim as optim
import os
import argparse
parser = argparse.ArgumentParser(description='Networks training')
parser.add_argument('-md', '--model_dir', action="store", dest="model_dir", type=str,
                    help="model directory", default='./run')
parser.add_argument('-g', '--gamma', action="store", dest="gamma", type=int,
                    help="gamma parameter", default=10)
parser.add_argument('-lmd', '--load-model-dir', action="store", dest="load_model_dir", type=str,
                    help="load model", default=None)


def main():
    print('Start training')
    args = parser.parse_args()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_dir = args.model_dir + '/model_' + current_time
    train_contents = '../data/detector/train.csv'
    train_df = pd.read_csv(train_contents)
    val_contents = '../data/detector/val.csv'
    val_df = pd.read_csv(val_contents)
    log_dir = os.path.join(model_dir, 'log.txt')
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        with open(model_dir + '/config.txt', 'w') as f:
            f.write(str(args))

    USE_CUDA = torch.cuda.is_available()
    tensor_transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        lambda x: x.float(),
                    ])
    tensor_transform_val = transforms.Compose([
                        transforms.ToTensor(),
                        lambda x: x.float(),
                    ])

    dataset_train = KneeDetectionDataset(train_df, tensor_transform_train, stage='train')
    dataset_val = KneeDetectionDataset(val_df, tensor_transform_val, stage='val')

    train_loader = data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=10)
    val_loader = data.DataLoader(dataset_val, batch_size=4, shuffle=True, num_workers=10)
    net = ResNet(pretrained=True, dropout=0.2)
    if USE_CUDA:
        net.cuda()
    print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = MSELoss()

    eval_iterations = 1000
    EPOCH = 50
    train_iterations(net, optimizer, train_loader, val_loader,
                     criterion, EPOCH, args, USE_CUDA, eval_iterations,
                     log_dir, model_dir)


if __name__ == '__main__':
    main()






