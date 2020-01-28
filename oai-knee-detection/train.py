from model.model import *
<<<<<<< HEAD
from model.model_hg2 import *
from model.dataloader import *
from model.train_utils import *
from model.augmentation import Rotate, RandomLRFlip
=======
from model.dataloader import *
from model.train_utils import *
from model.augmentation import Rotate
>>>>>>> 5e03d589fe09b235ac8a079b6ffa919209703438
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
import time
import torch.optim as optim
import os
import argparse
parser = argparse.ArgumentParser(description='Networks training')
parser.add_argument('-md', '--model_dir', action="store", dest="model_dir", type=str,
                    help="model directory", default='./model')
parser.add_argument('-g', '--gamma', action="store", dest="gamma", type=int,
                    help="gamma parameter", default=10)
parser.add_argument('-id', '--job-id', action="store", dest="job_number", type=int,
                    help="job_number", default=1)
parser.add_argument('-lm','--load-model',  action='store_true',
                    dest='load_model', help='If load state dict')
parser.add_argument('-nlm','--no-load-model',  action='store_false',
                    dest='load_model', help='If load state dict')
parser.set_defaults(dropout=False)
parser.add_argument('-lmd', '--load-model-dir', action="store", dest="load_model_dir", type=str,
                    help="load model", default=None)
parser.add_argument('-nr','--negative-regression', action='store', dest='nr', default=None, type=float)
parser.add_argument('-m', '--model', action='store', dest='model', default='resnet18', type=str,
                    choices=['resnet18', 'resnet34', 'hourglass'])



def main():
    print('Start training')
    args = parser.parse_args()
    job_number = args.job_number
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_dir = args.model_dir + '_' + current_time
    log_dir = os.path.join(model_dir, 'log.txt')
    train_contents = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/KneeDetection/bounding_box_oulu/train_with_neg_no_replacement.csv'
    train_df = pd.read_csv(train_contents)#.sample(n=200).reset_index()
    val_contents = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/KneeDetection/bounding_box_oulu/val_with_neg_no_replacement.csv'
    model_dir = args.model_dir + str(job_number)
    log_dir = os.path.join(model_dir, 'log.txt')
    train_contents = '/gpfs/data/denizlab/Users/bz1030/data/bounding_box/train_with_neg.csv'
    train_df = pd.read_csv(train_contents)#.sample(n=200).reset_index()
    val_contents = '/gpfs/data/denizlab/Users/bz1030/data/bounding_box/val_with_neg.csv'
    with open(model_dir + '/config.txt', 'w') as f:
        f.write(str(args))

    USE_CUDA = torch.cuda.is_available()
    tensor_transform_train = transforms.Compose([
                        #RandomLRFlip(0.3),
                        #Rotate(-3, 3),
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
    is_classifier = True
    if is_classifier:
        if 'resnet' in args.model:
            net = ResNetDetection(pretrained=True, dropout=0.2, use_cuda=USE_CUDA, model=args.model)
        else:
            net = HourglassNet(nStacks=4, nModules=1, nFeat=256, nClasses=4)
            net.cuda()
    else:
        net = ResNet(pretrained=True, dropout=0.2, use_cuda=USE_CUDA, model=args.model)
    load_model = args.load_model
    if USE_CUDA and load_model:
        net.load_state_dict(torch.load(args.load_model_dir))
    elif load_model:
        net.load_state_dict(torch.load(args.load_model_dir), map_location='cpu')
    print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    criterion = CombinedLoss(args.gamma, args.nr)

    eval_iterations = 1000
    EPOCH = 50
    train_iterations(net, optimizer, train_loader, val_loader,
                     criterion, EPOCH, args,USE_CUDA, eval_iterations,
                     log_dir, model_dir, is_classifier)


if __name__ == '__main__':
    main()






