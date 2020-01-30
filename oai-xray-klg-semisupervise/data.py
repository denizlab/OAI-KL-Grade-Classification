import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import load_image
from torchvision import transforms
from augmentation import *
import h5py


class KneeGradingDatasetNew(Dataset):
    '''
    Knee KLG grade dataset class
    '''
    def __init__(self, dataset, home_path, transform, data_type='uint8'):
        '''

        :param dataset:
        :param home_path:
        :param transform:
        :param data_type: float matrix or uint 8
        :param augmentation:
        '''
        self.dataset = dataset
        self.transform = transform
        self.home_path = home_path
        self.data_type = data_type
    def __getitem__(self, index):
        row = self.dataset.loc[index]
        month = row['Visit']
        pid = row['ID']
        target = int(row['KLG'])
        side = int(row['SIDE'])
        if side == 1:
            fname = '{}_{}_RIGHT_KNEE.hdf5'.format(pid,month)
        elif side == 2:
            fname = '{}_{}_LEFT_KNEE.hdf5'.format(pid, month)
        path = os.path.join(self.home_path, month, fname)
        img = np.array(h5py.File(path, 'r')['data'][:]).astype('float32')
        # uint 8 might be unnecessary
        if self.data_type == 'uint8':
            img = np.uint8(img)
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        if self.transform:
            img = self.transform(img)
        return img, target, fname

    def __len__(self):
        return self.dataset.shape[0]


class NYUUnsupervisedDataset(Dataset):
    def __init__(self, dataset, transform, data_type='uint8'):
        self.dataset = dataset
        self.transform = transform
        self.data_type = data_type

    def __getitem__(self, item):
        row = self.dataset.loc[item]
        fpath = row['file_path']
        img = np.array(h5py.File(fpath, 'r')['data'][:]).astype('float32')
        # uint 8 might be unnecessary
        if self.data_type == 'uint8':
            img = np.uint8(img)

        img = np.expand_dims(img, axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

    def __len__(self):
        return self.dataset.shape[0]


class KneeGradingDataSetUnsupervised(KneeGradingDatasetNew):
    '''
    Knee Grading dataset.
    '''
    def __getitem__(self, idx):
        random_idx = np.random.randint(0, self.__len__())
        image_1, _, _ = super().__getitem__(random_idx)
        image_2, _, _ = super().__getitem__(random_idx)
        return image_1, image_2


def dataloaders_dict_init(csv_dir_dict, data_dir_dict, args):
    # get params from args
    batch_size = args.batch_size
    augmentation = args.augmentation
    transform_val = transforms.Compose([
        CenterCrop(896),
        transforms.ToTensor(),
        lambda x: x.float(),
    ])
    augment_transforms = transforms.Compose([
        GaussianNoise(0.5, 0, 1),
        RandomLRFlip(0.5),
        RandomCrop(896),
        Rotate(-10, 10),
        transforms.ToTensor()
    ])
    if augmentation:
        transform_train = transforms.Compose([
            augment_transforms,
            lambda x: x.float(),
        ])
    else:
        transform_train = transforms.Compose([
            RandomCrop(896),
            transforms.ToTensor(),
            lambda x: x.float(),
        ])
    # read data into csv
    train = pd.read_csv(csv_dir_dict + 'train.csv')#.sample(n=100).reset_index()
    val = pd.read_csv(csv_dir_dict + 'val.csv')#.sample(n=100).reset_index()
    if args.demo:
        train = train.sample(n=600).reset_index()
        val = val.sample(n=100).reset_index()

    # create dataloader, and turn on/off augmentation
    dataset_train = KneeGradingDatasetNew(train, data_dir_dict, transform_train, args.data_type)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                            shuffle=True, num_workers=10)

    dataset_val = KneeGradingDatasetNew(val, data_dir_dict, transform_val, args.data_type)

    dataloader_val = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=True, num_workers=10)

    return {'train': dataloader_train, 'val': dataloader_val}
