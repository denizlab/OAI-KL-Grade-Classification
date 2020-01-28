from model.detector import Detector
import pandas as pd
from model.dataloader import DicomDataset
import torchvision.transforms as transforms
import torch.utils.data as data
from model.model import ResNet
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import sys
'''
Output bounding box from given model
'''
if __name__ =='__main__':
    Month = str(sys.argv[1])
    contents = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/Dataset/OAI_summary.csv'
    data_home = '/gpfs/data/denizlab/Datasets/OAI_original'
    load_model = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/KneeDetection/Experiment/ResNet18_/epoch_45.pth'
    USE_CUDA = torch.cuda.is_available()
    df = pd.read_csv(contents)
    df = df.loc[df.Visit == Month]
    annot_dataset = df[['Folder', 'Visit']].drop_duplicates().reset_index()
    annot_dataset.drop('index', axis =1, inplace=True)
    print(annot_dataset.head())
    dataset2annot = DicomDataset(annot_dataset, data_home, None)
    annot_loader = data.DataLoader(dataset2annot, batch_size=16)
    net = ResNet(pretrained=True, dropout=0.2, use_cuda=USE_CUDA)
    net.eval()
    if USE_CUDA:
        net.cuda()
        net.load_state_dict(torch.load(load_model))
    else:
        net.load_state_dict(torch.load(load_model, map_location='cpu'))
    rows =[]
    cols =[]
    ratios_x = []
    ratios_y = []
    bboxs = []
    all_names = []
    bar = tqdm(total=len(annot_loader), desc='Processing', ncols=90)
    for i, (batch,row,col,ratio_x,ratio_y,f_name) in enumerate(annot_loader):
        if USE_CUDA:
            inputs = Variable(batch.cuda())
        else:
            inputs = Variable(batch)
        output = net(inputs)

        bboxs.append(output.data.cpu().numpy())
        rows.extend(row.data.cpu().numpy().tolist())
        cols.extend(col.data.cpu().numpy().tolist())
        ratios_x.extend(ratio_x.data.cpu().numpy().tolist())
        ratios_y.extend(ratio_y.data.cpu().numpy().tolist())
        all_names.extend(f_name)
        bar.update(1)
    bboxs = np.vstack(bboxs)
    print(bboxs.shape)
    print(len(rows),len(cols),len(ratios_y),len(ratios_x),len(all_names))
    df = pd.DataFrame(bboxs)
    df['rows'] = rows
    df['cols'] = cols
    df['ratios_x'] = ratios_x
    df['ratios_y'] = ratios_y
    df['fname'] = all_names
    df.to_csv('output{}.csv'.format(Month),index=False)