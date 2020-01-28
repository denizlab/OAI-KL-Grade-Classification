import pandas as pd
import os
'''
Generate training data for the detector and classifier
It will contains negative sample from annotation of NYU dataset
'''
fname = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/UnsupervisedData/data/no_knee_no_replacement.txt'

result = []
home_path = '/gpfs/data/denizlab/Datasets/i17-01339_SubData/test'
with open(fname, 'r') as f:
    for line in f:
        line = line.rstrip()
        line = line.split('_')
        file_path = home_path + '/' + '_'.join(line[-2:]).replace('.png','.h5')
        result.append([file_path] + [-1] * 8)

result = pd.DataFrame(result)
#result.sample(n=3000, replace=True).to_csv('no_knee.csv', index=False)
print(result.head())
# train test val append

data_path = '/gpfs/data/denizlab/Users/bz1030/data/bounding_box/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
val = pd.read_csv(data_path + 'val.csv')
print(train.head())
cols = ['col' + str(i) for i in range(9)]
train.columns = cols
test.columns = cols
val.columns = cols
result.columns = cols


train_with_neg.to_csv('train.csv', index=False)
test_with_neg.to_csv('test.csv', index=False)
val_with_neg.to_csv('val.csv', index=False)
