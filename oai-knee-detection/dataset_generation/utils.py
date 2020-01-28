import os
import numpy as np
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import h5py
import pandas as pd
import time
import os
import numpy as np
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def image_preprocessing(file_path = '../data/9000296'):
    '''

    :param file_path:
    :return:
    '''
    # read data from DICOM file
    data = dicom.read_file(file_path)
    photoInterpretation = data[0x28,0x04].value # return a string of photometric interpretation
    #print('######### PHOTO INTER {} #########'.format(photoInterpretation))
    if photoInterpretation not in ['MONOCHROME2','MONOCHROME1']:
        raise ValueError('Wrong Value of Photo Interpretation: {}'.format(photoInterpretation))
    img = interpolate_resolution(data).astype(np.float64) # get fixed resolution
    img_before = img.copy()
    if photoInterpretation == 'MONOCHROME1':
        img = invert_Monochrome1(img)
    # apply normalization, move into hist_truncation.
    # img = global_contrast_normalization(img)
    # apply hist truncation
    img = hist_truncation(img)
    rows, cols = img.shape
    # get center part of image if image is large enough
    if rows >= 2048 and cols >= 2048:
        img = get_center_image(img)
    else:
        img,_,_ = padding(img)
        img = get_center_image(img) # after padding get the center of image


    return img,data,img_before
def invert_Monochrome1(image_array):
    '''
    Image with dicome attribute [0028,0004] == MONOCHROME1 needs to
    be inverted. Otherwise, our way to detect the knee will not work.

    :param image_array:
    :return:
    '''
<<<<<<< HEAD
    image_array = image_array.max() - image_array
=======
    #print('Invert Monochrome ')
    #print(image_array.shape, np.mean(image_array), np.min(image_array), np.max(image_array))
    # image_array = -image_array + 255.0 # our method
    image_array = image_array.max() - image_array
    #print(image_array.shape, np.mean(image_array), np.min(image_array), np.max(image_array))
>>>>>>> 5e03d589fe09b235ac8a079b6ffa919209703438
    return image_array

def interpolate_resolution(image_dicom, scaling_factor=0.2):
    '''
    Obtain fixed resolution from image dicom
    :param image_dicom:
    :param scaling_factor:
    :return:
    '''
<<<<<<< HEAD
    image_array = image_dicom.pixel_array
=======
    #print('Obtain Fix Resolution:')
    image_array = image_dicom.pixel_array
    #print(image_array.shape,np.mean(image_array),np.min(image_array),np.max(image_array))
>>>>>>> 5e03d589fe09b235ac8a079b6ffa919209703438
    try:
        x = image_dicom[0x28, 0x30].value[0]
        y = image_dicom[0x28, 0x30].value[1]
        image_array = ndimage.zoom(image_array, [x / scaling_factor, y / scaling_factor])
<<<<<<< HEAD
=======
    #print(image_array.shape,np.mean(image_array),np.min(image_array),np.max(image_array))
>>>>>>> 5e03d589fe09b235ac8a079b6ffa919209703438
    except KeyError:
        pass
    return image_array

def get_center_image(img,img_size = (2048,2048)):
    '''
    Get the center of image
    :param img:
    :param img_size:
    :return:
    '''
    rows,cols = img.shape
    center_x = rows // 2
    center_y = cols // 2
    img_crop = img[center_x - img_size[0] // 2: center_x + img_size[0] // 2,
                   center_y - img_size[1] // 2: center_y + img_size[1] // 2]
    return img_crop

def padding(img,img_size = (2048,2048)):
    '''
    Padding image array to a specific size
    :param img:
    :param img_size:
    :return:
    '''
    rows,cols = img.shape
    x_padding = img_size[0] - rows
    y_padding = img_size[1] - cols
    if x_padding > 0:
        before_x,after_x = x_padding // 2, x_padding - x_padding // 2
    else:
        before_x,after_x = 0,0
    if y_padding > 0:
        before_y,after_y = y_padding // 2, y_padding - y_padding // 2
    else:
        before_y,after_y = 0,0
    return np.pad(img,((before_x,after_x),(before_y,after_y)),'constant'),before_x,before_y

def global_contrast_normalization_oulu(img,lim1,multiplier = 255):
    '''
    This part is taken from oulu's lab. This how they did global contrast normalization.
    :param img:
    :param lim1:
    :param multiplier:
    :return:
    '''
    img -= lim1
    img /= img.max()
    img *= multiplier
    return img

def global_contrast_normalization(img, s=1, lambda_=10, epsilon=1e-8):
    '''
    Apply global contrast normalization based on image array.
    Deprecated since it is not working ...
    :param img:
    :param s:
    :param lambda_:
    :param epsilon:
    :return:
    '''
    # replacement for the loop
    print('Global contrast normalization:')
    print(img.shape, np.mean(img), np.min(img), np.max(img))
    X_average = np.mean(img)
    #print('Mean: ', X_average)
    img_center = img - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lambda_ + np.mean(img_center ** 2))

    img = s * img_center / max(contrast, epsilon)
    print(img.shape, np.mean(img), np.min(img), np.max(img))
    # scipy can handle it
    return img

def hist_truncation(img,cut_min=5,cut_max = 99):
    '''
    Apply 5th and 99th truncation on the figure.
    :param img:
    :param cut_min:
    :param cut_max:
    :return:
    '''
    #print('Trim histogram')
    #print(img.shape, np.mean(img), np.min(img), np.max(img))
    lim1,lim2 = np.percentile(img,[cut_min, cut_max])
    img_ = img.copy()
    img_[img < lim1] = lim1
    img_[img > lim2] = lim2
    #print(img_.shape, np.mean(img_), np.min(img_), np.max(img_))
    img_ = global_contrast_normalization_oulu(img_,lim1,multiplier=255)
    #print(img_.shape, np.mean(img_), np.min(img_), np.max(img_))
    return img_

def drawFigureOnOriginal(img,labels,preds,f_name,folder):
    '''
        draw a png figure with rect of ground truth and prediction
        col == x, row == y
        :param img:
        :param labels:
        :param preds:
        :param f_name:
        :return:
    '''
    fig, ax = plt.subplots(1)
    row, col = img.shape
    ax.imshow(img)
    # draw true patch
    if labels is not None:
        x1, y1, x2, y2 = labels[:4]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        x1, y1, x2, y2 = labels[4:]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect2)
    if preds is not None:
        # draw predict patch
        preds = preds
        x1, y1, x2, y2 = preds[:4]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        print('Left:({},{}) - ({},{})'.format(x1, y1, x2, y2))
        ax.add_patch(rect1)
        x1, y1, x2, y2 = preds[4:]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        print('Right:({},{}) - ({},{})'.format(x1,y1,x2,y2))
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect2)
    # save image
    plt.savefig(os.path.join(folder, f_name), dpi=300)
    plt.close()
def drawKneeWithBbox(img,bbox,output_dir,f_name):
    row,col = img.shape
    x1, y1, x2, y2 = bbox[:4]

    x1 = int(x1 * col)
    x2 = int(x2 * col)
    y1 = int(y1 * row)
    y2 = int(y2 * row)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # max is used to avoid negative index
    left = img[max(cy -512,0): max(cy - 512,0) + 1024,max(cx - 512,0): max(cx - 512,0) + 1024]
    drawFigureOnOriginal(left,None,None,f_name.replace('.png','l.png'),output_dir)
    x1, y1, x2, y2 = bbox[4:]
    x1 = int(x1 * col)
    x2 = int(x2 * col)
    y1 = int(y1 * row)
    y2 = int(y2 * row)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    right = img[max(cy -512,0): max(cy - 512,0) + 1024,max(cx - 512,0): max(cx - 512,0) + 1024]
    drawFigureOnOriginal(right,None,None,f_name.replace('.png','r.png'),output_dir)

def getKneeWithBbox(img,bbox):
    row, col = img.shape
    x1, y1, x2, y2 = bbox[:4]

    x1 = int(x1 * col)
    x2 = int(x2 * col)
    y1 = int(y1 * row)
    y2 = int(y2 * row)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # max is used to avoid negative index
    left = img[max(cy - 512, 0): max(cy - 512, 0) + 1024, max(cx - 512, 0): max(cx - 512, 0) + 1024]
    x1, y1, x2, y2 = bbox[4:]
    x1 = int(x1 * col)
    x2 = int(x2 * col)
    y1 = int(y1 * row)
    y2 = int(y2 * row)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    right = img[max(cy - 512, 0): max(cy - 512, 0) + 1024, max(cx - 512, 0): max(cx - 512, 0) + 1024]
    return left,right

def create_h5(save_dir,f_name,img):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_path = os.path.join(save_dir,f_name)
    f = h5py.File(data_path, 'w')
    f.create_dataset('data', data=img)
    f.close()