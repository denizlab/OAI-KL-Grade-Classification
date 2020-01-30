main.py
---
Used pretrained model to make prediction, and output bbox files
Used `scripts/annote.lsf` to submite jobs


draw_figure2.py
---
Draw a figure with predicted bbox of given month
```
python draw_figure2.py 96m &
```
dataset_generation/
---
Generate figures that has whole knee x-ray image with predicted bbox
Generate dataset (h5 files) based on the summary file.


Current model follows Fast RCNN model architecture to detect knees. The procedure
will be
1. run `train_test_build.py` to generate train test from OAI dataset (Oulu Lab)
1. run `train_test_split.py` to combine the positive labeled data from OAI (Oulu Lab) with negative labeled data from
NYU. If found any false positive case during evaluation, one can simply add that case 
into file `./bounding_box_oulu/no_knee.txt`. When resuming train, the following code will take new added negative labeled
data.
1. run `train.py` to train a model and evaluate
1. run `train_cn.py
` to train a model using center net