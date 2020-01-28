'''

Generate model predict bbox
'''
import pandas as pd
import numpy as np
import os
import time
import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
df = pd.read_csv('test_output.csv')

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
    labels = labels * row
    preds = preds * row
    # draw true patch
    x1, y1, x2, y2 = labels[:4]
    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect1)
    x1, y1, x2, y2 = labels[4:]
    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect2)
    # draw predict patch
    x1, y1, x2, y2 = preds[:4]
    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect1)
    x1, y1, x2, y2 = preds[4:]
    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect2)
    # save image
    plt.savefig(os.path.join(folder, f_name), dpi=300)
    plt.close()
output_folder = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/test/bbox_pred/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for idx, row in df.iterrows():
    row = row.tolist()
    file_path = row[-1]
    labels = np.array(row[:8])
    preds = np.array(row[8:-1])
    f = h5py.File(file_path)
    img = f['data'].value
    f.close()
    f_name = file_path.split('/')[-1].replace('.h5','.png')

    drawFigure(img,labels,preds,f_name,output_folder)
