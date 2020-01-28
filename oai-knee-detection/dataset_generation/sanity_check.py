from utils import *
import pandas as pd
import numpy as np
import os
import time
import sys
import h5py
from tqdm import tqdm
'''
This script checks if all images in OAI are in our dataset.
'''
df = pd.read_csv('/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/Dataset/OAI_summary.csv')
output_folder = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/test/bbox_prepprocessing'
save_dir ='/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new'

print(df.head())
df = df[['ID','Visit']].drop_duplicates()
total_img = df.shape[0] * 2
count = 0
for idx,row in df.iterrows():
    p_id = row[0]
    visit = row[1]
    f_name_l = '{}_{}_LEFT_KNEE.hdf5'.format(p_id,visit)
    f_name_r = '{}_{}_RIGHT_KNEE.hdf5'.format(p_id,visit)
    f_name_l = os.path.join(save_dir,visit,f_name_l)
    f_name_r = os.path.join(save_dir,visit,f_name_r)
    if os.path.isfile(f_name_l):
        count += 1
    else:
        print('{} Not Found'.format(f_name_l))

    if os.path.isfile(f_name_r):
        count += 1
    else:
        print('{} Not Found'.format(f_name_r))

print('Total image left: {}/{}'.format(count,total_img))


