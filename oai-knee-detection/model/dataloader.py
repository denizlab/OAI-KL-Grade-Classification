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
from model.centernet_utils import gaussian_kernel
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
        if mode is 'HG':
            self.scale = 4 # scaling factor of Hourglass network
        else:
            self.scale = None

        self.debug = debug
        self.save_dir = save_dir
        self.augmentation = augmentation

    def __getitem__(self, index):

        row = self.dataset.iloc[index].tolist()
        fname = row[0]
        bbox = row[1:]
        target = bbox
        if self.mode is 'resnet':
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
            if 'i17-01339' in fname:
                photo_inter = f['PhotoInterpretation'].value
                if photo_inter == 'MONOCHROME1':
                    img = self._invert(img)
                img, _, _, _, _ = self._preprocessing(img, self.reshape)

            f.close()
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img[:, :], 3, axis=2)
            if self.transform:
                img = self.transform(img)

            return img, target, knee_label, fname
        else:
            # change due to the Cem move this folder
            changed_dir = '/gpfs/data/denizlab/Datasets/i17-01339/SubData4Unsupervised/test/'
            if 'bz1030' not in fname:
                fname = fname.split('/')[-1]
                fname = changed_dir + fname
            f = h5py.File(fname, 'r')
            img = f['data'].value
            if 'i17-01339' in fname:
                photo_inter = f['PhotoInterpretation'].value
                if photo_inter == 'MONOCHROME1':
                    img = self._invert(img)
                img, _, _, _, _ = self._preprocessing(img, self.reshape)

            f.close()
            # Augmentation
            flip = False
            if self.training:
                flip = np.random.randint(2) == 1
            if flip and self.augmentation:
                img = np.fliplr(img)
                img = np.ascontiguousarray(img)
            else:
                flip = False
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img[:, :], 3, axis=2)
            if self.transform:
                img = self.transform(img)

            mask, heatmap, width, height = self._preprocessing_center(target, flip=flip)
            #print('Inside data loader (SUM, MAX):', mask.sum(), mask.max())

            if self.debug:
                # print figure to see if it make sense or not
                print(fname, img.shape)

            return img, mask, heatmap, width, height, fname, target


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

    def _preprocessing_center(self, target, flip=False):
        masks, heatmaps, widths, heights = self.get_heatmap(target.copy())
        size = self.reshape // self.scale
        heatmap_total = np.zeros((size, size))
        mask_total = np.zeros((size, size))
        width_total = np.zeros((size, size))
        height_total = np.zeros((size, size))
        for mask, heatmap, w, h in zip(masks, heatmaps, widths, heights):
            mask_total += mask
            heatmap_total = np.maximum(heatmap_total, heatmap)
            coor = np.where(mask == 1)
            width_total[coor] = w  # for y
            height_total[coor] = h  # for x
        if flip:
            mask_total = np.fliplr(mask_total)
            heatmap_total = np.fliplr(heatmap_total)
            width_total = np.fliplr(width_total)
            height_total = np.fliplr(height_total)

            mask_total = np.ascontiguousarray(mask_total)
            heatmap_total = np.ascontiguousarray(heatmap_total)
            width_total = np.ascontiguousarray(width_total)
            height_total = np.ascontiguousarray(height_total)

        return mask_total, heatmap_total, width_total, height_total

    def get_heatmap(self, target):
        '''
        For original coordinates, (x1, y1) = (second coordinate (col), first axis (row))
        Therefore here switch them by computing  width and height
        :param target:
        :return:
        '''
        masks, heatmaps, widths, heights = [], [], [], []
        target = np.array(target)
        size = self.reshape // self.scale
        if -1 in target[:4]:
            mask = np.zeros((size, size))
            heatmap = np.zeros((size, size))
            width = 0
            height = 0

        else:
            #print('Left has targets!!!!!!!!!!!')
            x1, y1, x2, y2 = target[:4] * size
            #print('Left has targets!!!!!!!!!!!', x1, y1, x2, y2)

            height = 0.5 * (y2 - y1)
            width = 0.5 * (x2 - x1)
            xs = int(np.floor(0.5 * (y2 + y1)))
            ys = int(np.floor(0.5 * (x2 + x1)))
            #print('sigma=', 0.5 * (height + width) / 3)
            sigma = 0.5 * (height + width) / 3
            #sigma = 1
            mask, heatmap = gaussian_kernel(size, size, xs, ys, sigma=sigma)

        masks.append(mask)
        heatmaps.append(heatmap)
        widths.append(width)
        heights.append(height)

        if -1 in target[4:]:
            mask = np.zeros((size, size))
            heatmap = np.zeros((size, size))
            width = 0
            height = 0
        else:
            #print('Right has targets!!!!!!!!!!!')
            x1, y1, x2, y2 = target[4:] * size
            #print('Right has targets!!!!!!!!!!!', x1, y1, x2, y2)

            height = 0.5 * (y2 - y1)
            width = 0.5 * (x2 - x1)
            xs = int(np.floor(0.5 * (y2 + y1)))
            ys = int(np.floor(0.5 * (x2 + x1)))
            sigma = 0.5 * (height + width) / 3
            mask, heatmap = gaussian_kernel(size, size, xs, ys, sigma=sigma)
        masks.append(mask)
        heatmaps.append(heatmap)
        widths.append(width)
        heights.append(height)
        # heights x, width y
        return masks, heatmaps, heights, widths

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
        data_path = os.path.join(data_path,file_name)
        img, row, col, ratio_x, ratio_y = self._preprocessing(data_path)
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img, row, col, ratio_x, ratio_y, data_path

    def _preprocessing(self,data_path):
        dicom_img = dicom.dcmread(data_path)
        img = dicom_img.pixel_array.astype(float)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        row, col = img.shape
        img = cv2.resize(img, (self.reshape, self.reshape), interpolation=cv2.INTER_CUBIC)
        ratio_x = self.reshape / col
        ratio_y = self.reshape / row
        return img, row, col, ratio_x,ratio_y

    def __len__(self):
        return self.dataset.shape[0]
