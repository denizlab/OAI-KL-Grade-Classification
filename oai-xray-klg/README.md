# KL-grade Classifer
## Instruction
The training of classifier happens after the detector has been trained and all OAI data has been pre-processed. Please refer to `../data/classifier` for how to prepare a content csv file for dataloader.

To reproduce the baseline ResNet experiment and ResNet with CBAM[Citation], please run the following command:
```bash
# for baseline model
python3 main.py -n test -m baseline -do \
-au -ep 30 -lr 0.0001 -bs 6 -dm yes
# for CBAM model
python3 main.py -n test -m baseline -do \
-au -ep 30 -lr 0.0001 -bs 6 -dm yes
```
