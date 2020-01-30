# Knee Joint Localization Model
The training data is obtained from Tiulpin et al.[Citation]. The files are located at `../data/detector`. To reproduce the experiment in paper. You will need to:
1. run `train_test_build.py` to generate train test dataset for the detector from raw OAI dataset. 
1. run `train_test_split.py` to split generated dataset into train test dataset content files for dataloader in later step.
1. run `train.py` to train a model and evaluate
```bash
python3 train.py
```
1. run `main.py` to annotate all OAI images from given model:
```bash
python main.py -m 00m -md <trained model weights dir>
```
1. Last step will generate a `output_month.csv` file e.g. `output_00m.csv`. Then you can use `dataset_generation/preprocessing.py` to generate train/test data for the classifier.
```bash
python preprocessing.py -m 00m -sd ../../data/OAI_processed
```
Next, please refer to `../oai-xray-klg` for train/test of the classifier.
