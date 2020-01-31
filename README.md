# About
This Repo contains code for paper **Attention-based CNN for KL Grade Classification: Data from the Osteoarthritis Initiative**

# Instruction
Please refer to `requirements.txt` for install all dependencies for this project. `./data` folder contains example content file for train/test data used in dataloader for both detector and classifier. `./model_weights` folder contains model weights that achieved the performance metrics mentioned in paper.

This repo consists of two parts. To reproduce the entire experiments, you will need to
1. Train a detector and use the detector to annotate all OAI dataset, and generate train/test data for the classifier. See documentation in `./oai-knee-detection`.
2. Train and test the classifier by following documentation in `./oai-xray-klg`

# How to cite
```latex
Some latex code
```
