import pandas as pd
import os
import sys
import pydicom as dicom
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches
df1 = pd.read_csv('bounding_box_oulu/OAI_test.csv',header = None,sep=' ')
df2 = pd.read_csv('bounding_box_oulu/OAI_val.csv',header = None,sep=' ')

print(df1.head())
print(df2.head())

df = df1.append(df2)
print(df.shape)

month = '00m'
OAI_DATASET = '/gpfs/data/denizlab/Datasets/OAI_original'
output_folder = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/test/bbox_img_oulu_resize/' + month
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def writePngTo(folder, img,file_name,bbox,ratio_x= None,ratio_y = None):
    fig,ax = plt.subplots(1)
    print(img.shape)
    ax.imshow(img)
    x1, y1, x2, y2 = bbox[:4]
    if ratio_x:
        x1 = x1 * ratio_x
        x2 = x2 * ratio_x
    if ratio_y:
        y1 = y1 * ratio_y
        y2 = y2 * ratio_y
    print(x1,y1,x2,y2)
    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect1)
    x1, y1, x2, y2 = bbox[4:]
    if ratio_x:
        x1 = x1 * ratio_x
        x2 = x2 * ratio_x
    if ratio_y:
        y1 = y1 * ratio_y
        y2 = y2 * ratio_y
    print(x1, y1, x2, y2)
    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect2)
    plt.savefig(os.path.join(folder,file_name),dpi=300)
    plt.close()
def recursive_dir(file_path):
    if not os.path.isdir(file_path):
        return file_path
    if os.path.exists(file_path):
        next_folder = os.listdir(file_path)
        file_path = recursive_dir(os.path.join(file_path,next_folder[0]))
    return file_path
total_samples = df.shape[0]
count = 0

resize_shape = 898
for idx,row in df.iterrows():
    start = time.time()
    file_name = row[0]
    bbox = [int(i) for i in row[1:]]

    data_path = os.path.join(OAI_DATASET,month,file_name)
    count += 1
    print('Process {}/{}'.format(count,total_samples))
    try:
        dicom_img = dicom.dcmread(data_path)
        img = dicom_img.pixel_array.astype(float)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        row,col = img.shape
        if resize_shape:
            img = cv2.resize(img, (resize_shape,resize_shape), interpolation=cv2.INTER_CUBIC)
            ratio_x = resize_shape / col
            ratio_y = resize_shape / row
        #else:
        #    ratio_y = None
        #    ratio_x = None
        x = dicom_img[0x28, 0x30].value[0]
        y = dicom_img[0x28, 0x30].value[1]
        img_f_name = file_name.replace('/','_')
        writePngTo(output_folder,img,img_f_name + '.png',bbox,ratio_x,ratio_y)

    except Exception as e:
        print(e)
        print(data_path)
