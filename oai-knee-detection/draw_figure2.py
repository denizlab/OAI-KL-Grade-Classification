'''

Generate model predict bbox
'''
import pandas as pd
import numpy as np
import os
import time
import h5py
import pydicom as dicom
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sys

def preprocessing(data_path,reshape):
    dicom_img = dicom.dcmread(data_path)
    img = dicom_img.pixel_array.astype(float)
    img = (np.maximum(img, 0) / img.max()) * 255.0
    img_original = img.copy()
    row, col = img.shape
    img = cv2.resize(img, (reshape, reshape), interpolation=cv2.INTER_CUBIC)
    ratio_x = reshape / col
    ratio_y = reshape / row
    return img,img_original,row,col,ratio_x,ratio_y

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
def drawFigure(img,labels,preds,f_name,folder):
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
        labels = labels * row
        x1, y1, x2, y2 = labels[:4]
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        x1, y1, x2, y2 = labels[4:]
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect2)
    if preds is not None:
        # draw predict patch
        preds = preds * row
        x1, y1, x2, y2 = preds[:4]
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect1)
        x1, y1, x2, y2 = preds[4:]
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect2)
    # save image
    plt.savefig(os.path.join(folder, f_name), dpi=300)
    plt.close()

if __name__ == '__main__':
    Month = str(sys.argv[1])
    output_folder = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/test/bbox_pred{}_original/'.format(Month)
    df = pd.read_csv('output_data/output{}.csv'.format(Month))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, row in df.iterrows():
        row = row.tolist()
        bbox = np.array(row[:8])
        data_path = row[-1]
        img, img_original,row, col, ratio_x, ratio_y = preprocessing(data_path,898)
        f_name = data_path
        f_name = f_name.split('/')[-6:-1]
        f_name = '_'.join(f_name) + '.png'
        #drawFigure(img,None,bbox,f_name,output_folder)
        drawFigureOnOriginal(img_original,None,bbox,f_name,output_folder)