# Knee Joint Localization Model
Generate figures that has whole knee x-ray image with predicted bbox
Generate dataset (h5 files) based on the summary file.


Current model use ResNet as the backbone. Training is a simple regression between 
target bounding box and predictions.
1. run `train_test_build.py` to generate train test from OAI dataset (Oulu Lab)
1. run `train_test_split.py` to split dataset into three csv file, train/test/val
1. Train model by the following command
```
python3 train.py
```
1. Evaluate model by the following command
```
python3 evaluate.py -lm <model dir>
```
1. Annotate all data file from OAI by input content file from each month folder
```
python3 main.py -md <model dir> -cd <content-dir> -dh <data home> -m <month>
```