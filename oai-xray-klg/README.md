# KL-grade Classifer
This part of Repo contains code that trains a KL-grade classifier with ResNet and ResNet with [CBAM](https://github.com/Jongchan/attention-module).

# Instruction
The training of classifier happens after the detector has been trained and all OAI data has been annotated by the detector and pre-processed. Please refer to [../data/classifier](https://github.com/denizlab/OAI-KL-Grade-Classification/tree/master/data/classifier) for how to prepare a content csv file for dataloader. The file looks like
```
ID,SIDE,READPRJ,KLG,Visit,Folder,StudyDate,Barcode
9000099,1,15,2.0,00m,0.E.1/9000099/20050531/00839603,20050531,839603
9000099,2,15,3.0,00m,0.E.1/9000099/20050531/00839603,20050531,839603
```
Where ID is patient ID. SIDE 1, 2 stands for **right**, **left** knee joint respectively. This file also keeps track of KL-grade. From these identifiers, you can construct a file directory where you saved target data. Refer to `./data.py` for dataloader to learn how to load data for training.


To reproduce the baseline ResNet experiment and ResNet with CBAM, please run the following command:
```bash
python3 main.py -n baseline -m CBAM -do\
 -au -ep 30 -lr 0.0001 -bs 6 -dm no\
 -d /gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new4/\
 -dc /gpfs/data/denizlab/Users/bz1030/data/OAI_proj15/
```
Above code will get content file from `OAI_proj15` file and then load data from `OAI_processed_new4`. Arguments `-m` can have `baseline` or `CBAM` which stand for the experiments mentioned in paper. Please refer `./main.py` for description of other arguments.
