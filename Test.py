import pandas as pd
import csv
TRAIN_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/train/train.tsv'
#train = pd.read_csv(TRAIN_PATH,sep='\t', chunksize=10, nrows = 100, quoting=csv.QUOTE_NONE)
Trainlist =[]
count = 0
for chunk in pd.read_csv(TRAIN_PATH,sep='\t',chunksize=10000):
    Trainlist.append(chunk)
    count = count + 1
    print(count)
TrainMatrix = pd.concat(Trainlist,axis=0)
del Trainlist
