'''
This script contains a class read input from dataset
and output predicted bbox
'''
import numpy as np
import cv2
import os
import sys
import pydicom as dicom
class Detector:
    def __init__(self,model,dataset,shape = 898):
        self.model = model
        self.dataset = dataset
        self.shape = shape

    def _preprocessing(self,img):
        img = np.maximum(img,0) / img.max() * 255.0
        row,col = img.shape

