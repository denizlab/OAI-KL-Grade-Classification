from utils import *
import pandas as pd
import numpy as np
import os
import time
import sys
import h5py
from tqdm import tqdm
import scipy.ndimage as ndimage

'''
This file used new pipeline to get images preprocessed.
'''
MONTH = str(sys.argv[1])
df = pd.read_csv('../output_data/output{}.csv'.format(MONTH))
output_folder = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/test/bbox_prepprocessing'
save_dir ='/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new3'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
bar = tqdm(total=df.shape[0], desc='Processing', ncols=90)

for idx, row in df.iterrows():
    row = row.tolist()
    data_path = row[-1]
    data_path = data_path.replace('//','/')
    month = data_path.split('/')[-6]
    p_id = data_path.split('/')[-4]
    print(month,p_id)
    bbox = np.array(row[:8])
    img, data, img_before = image_preprocessing(data_path)
    left,right = getKneeWithBbox(img,bbox)
    f_name_l = '{}_{}_LEFT_KNEE.hdf5'.format(p_id,month)
    f_name_r = '{}_{}_RIGHT_KNEE.hdf5'.format(p_id,month)
    print(f_name_l,f_name_r)
    path2save = os.path.join(save_dir,month)
    print(left.shape,right.shape)
    left, _, _ = padding(left, img_size=(1024,1024))
    right, _, _ = padding(right, img_size=(1024,1024))
    create_h5(path2save,f_name_l,left)
    create_h5(path2save, f_name_r, right)
    bar.update(1)


''' to display the data

for idx, row in df.iterrows():
    row = row.tolist()
    data_path = row[-1]
    data_path = data_path.replace('//','/')
    bbox = np.array(row[:8])
    img, data, img_before = image_preprocessing(data_path)
    f_name = data_path
    f_name = f_name.split('/')[-6:-1]
    f_name = '_'.join(f_name) + '.png'
    drawKneeWithBbox(img,bbox,output_folder,f_name)
    print(data_path)
'''