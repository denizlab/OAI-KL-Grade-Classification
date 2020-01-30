# About
This Repo contains code for paper, Attention-based CNN for KL Grade Classification: Data from the Osteoarthritis Initiative
### Under construction
Please refer to `environment.yml` for creating a conda environment for this project. This repo consists of two parts. To reproduce the entire experiments, you will need to
1. Train a detector and use the detector to annotate all OAI dataset, and generate train/test data for the classifier. See documentation in `./oai-knee-detection`.
2. Train and test the classifier by following documentation in `./oai-xray-klg`
