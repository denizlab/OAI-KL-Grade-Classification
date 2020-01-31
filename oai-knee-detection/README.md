# Knee Joint Localization Model
The training data is obtained from Tiulpin et al.[Citation]. The files are located at `../data/detector`. To reproduce the experiment in paper. You will need to:
1. run `train_test_build.py` to generate train test dataset for the detector from raw OAI dataset. The script takes images from OAI 00m and perform pre-processing.
1. Prepare train/test split by following example in [../data/train.csv](https://github.com/denizlab/OAI-KL-Grade-Classification/tree/master/data/detector)
The file looks like
```
0,1,2,3,4,5,6,7,8
/gpfs/data/denizlab/Users/bz1030/data/bounding_box/train/0.E.1_9679795_20050907_01087203_001.h5,0.6247968423496634,0.29277998862990334,0.9034130485256558,0.6338828880045481,0.13628976085442304,0.316372939169983,0.41490596703041555,0.6574758385446277
/gpfs/data/denizlab/Users/bz1030/data/bounding_box/train/0.C.2_9687273_20040914_00267603_001.h5,0.5451764705882354,0.34777618364418933,0.8218823529411765,0.685222381635581,0.16282352941176473,0.32969870875179336,0.4395294117647059,0.667144906743185
```
The first column is the directory where you save the image file. The columns 2 - 9 are the coordinates `[x1, y1, x2, y2, x'1, y'1, x'2, y'2]` that normalized by the original image size. For example, if image size is 1000x1000, and original `x1` is 500, then it will be 0.5 in the file. Note, `x` stands for width and `y` stands for height.

1. run `train.py` to train a model and evaluate
```bash
python3 train.py
```
1. run `main.py` to annotate all OAI images from given model:
```bash
python main.py -m 00m -md <trained model weights dir>
```
1. Last step will generate a `output_month.csv` file e.g. `output_12m.csv`. 
The file will look like
```
0,1,2,3,4,5,6,7,rows,cols,ratios_x,ratios_y,fname
0.4217677,0.1803451,0.5549975,0.40280363,0.035544623,0.19921528,0.2138884,0.40788278,3487,4376,0.20475319926873858,0.2569544020648122,/gpfs/data/denizlab/Datasets/OAI_original/12m/1.E.1/9000099/20060713/01653203/001
0.4116512,0.16901684,0.5621055,0.39205313,0.08820985,0.18897364,0.2743088,0.40614438,2048,2494,0.35926222935044105,0.4375,/gpfs/data/denizlab/Datasets/OAI_original/12m/1.C.2/9000296/20051007/01140204/001
```
Columns 1-8 will be the coordinates of bounding box. The last column is the data directory where you can read the image
.
Then you can use `dataset_generation/preprocessing.py` to generate train/test data for the classifier.
```bash
# this code look for 'output_00m.csv' to generate dataset.
python preprocessing.py -m 00m -sd ../../data/OAI_processed
```
Next, please refer to `../oai-xray-klg` for train/test of the classifier.
