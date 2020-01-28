"""
Dataset classes and samplers for knee localization

"""

import torch.utils.data as data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import os

import torchvision.transforms as transforms
import h5py
import pydicom as dicom
import time


class KneeDetectionDataset(data.Dataset):
    def __init__(self, dataset, transform, stage='train', mode='resnet',
                 reshape=896, debug=False, save_dir=None, training=True, augmentation=False):
        assert mode in ['resnet', 'HG']
        self.dataset = dataset
        self.transform = transform
        self.stage = stage
        self.mode = mode
        self.reshape = reshape
        self.training = training
        self.debug = debug
        self.save_dir = save_dir
        self.augmentation = augmentation

    def __getitem__(self, index):

        row = self.dataset.iloc[index].tolist()
        fname = row[0]
        bbox = row[1:]
        target = bbox

        if -1 in target:
            knee_label = 0
        else:
            knee_label = 1
        # change due to the Cem move this folder
        changed_dir = '/gpfs/data/denizlab/Datasets/i17-01339/SubData4Unsupervised/test/'
        if 'bz1030' not in fname:
            fname = fname.split('/')[-1]
            fname = changed_dir + fname
        f = h5py.File(fname, 'r')
        img = f['data'].value
        f.close()
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        if self.transform:
            img = self.transform(img)

        return img, target, knee_label, fname


    def _invert(self, img):
        img = img.max() - img
        return img

    def _preprocessing(self, img, reshape=896):
        img = img.astype(np.float)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        row, col = img.shape
        img = cv2.resize(img, (reshape, reshape), interpolation=cv2.INTER_CUBIC)
        ratio_x = reshape / col
        ratio_y = reshape / row
        return img, row, col, ratio_x, ratio_y

    def __len__(self):
        return self.dataset.shape[0]


class DicomDataset(data.Dataset):
    def __init__(self, dataset, home_dir,transform, stage='train',reshape=896):
        self.dataset = dataset
        self.home_dir = home_dir
        self.transform = transform
        self.stage = stage
        self.reshape = reshape

    def __getitem__(self, index):
        row = self.dataset.loc[index].tolist()
        visit = row[1]
        data_path = row[0]
        data_path = os.path.join(self.home_dir,visit,data_path)
        file_name = os.listdir(data_path)[0]
        data_path = os.path.join(data_path, file_name)
        img, row, col, ratio_x, ratio_y = self._preprocessing(data_path)
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img, row, col, ratio_x, ratio_y, data_path

    def _preprocessing(self,data_path):
        dicom_img = dicom.dcmread(data_path, force=True)
        img = dicom_img.pixel_array.astype(float)
        if dicom_img.PhotometricInterpretation == 'MONOCHROME1':
            img = self._invert(img)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        row, col = img.shape
        img = cv2.resize(img, (self.reshape, self.reshape), interpolation=cv2.INTER_CUBIC)
        ratio_x = self.reshape / col
        ratio_y = self.reshape / row
        return img, row, col, ratio_x,ratio_y

    def _invert(self, img):
        img = img.max() - img
        return img

    def __len__(self):
        return self.dataset.shape[0]
