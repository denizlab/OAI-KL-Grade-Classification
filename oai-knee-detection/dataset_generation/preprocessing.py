from utils import *
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse

'''
This file used new pipeline to get images preprocessed.
'''
parser = argparse.ArgumentParser(description='Annotation')
parser.add_argument('-m', '--month', action="store", dest="month", type=str,
                    help='month file for OAI e.g. 00m 12m 24m etc.', default=None)
parser.add_argument('-md', '--model-dir', type=str, dest='model_dir',
                    default='../model_weights/KneeJointLocalModel/resnet18_detector.pth')
parser.add_argument('-sd', '--save-dir', action="store", dest="save_dir", type=str,
                    help="where you want to save after conversion from DICOM to H5",
                    default='../data/OAI_processed/')
def main():
    args = parser.parse_args()
    Month = args.month
    df = pd.read_csv('../output_data/output{}.csv'.format(MONTH))
    output_folder = '../data/bbox_prepprocessing'
    save_dir ='../data/OAI_processed_new3'
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


if __name__ == '__main__':
    main()