# Knee Joint Localization Model
The training data is obtained from Tiulpin et al.[Citation]. The files are located at `../data/detector`. To reproduce the experiment in paper. You will need to:
1. run `train_test_build.py` to generate train test dataset for the detector from raw OAI dataset. The script takes images from OAI 00m and perform pre-processing.
1. Generate train/test split by following example in [../data/train.csv](https://github.com/denizlab/OAI-KL-Grade-Classification/tree/master/data/detector)
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
