import pandas as pd
import base64
import swifter
from tqdm import tqdm
import csv
import pickle
import joblib
import gc
from time import sleep
from DataTransfer import convertBoxes
from DataTransfer import convertFeature
from DataTransfer import convertLabel
from DataTransfer import convertLabelWord
from DataTransfer import convertPos

TRAIN_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/train/train.tsv'
VAL_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/valid/valid.tsv'
VAL_ANS_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/valid/valid_answer.json'
SAMPLE_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/train/train.sample.tsv'
LABEL_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/multimodal_labels.txt'
TEST_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/testA/testA.tsv'
# 读10000条训练数据    
#train = pd.read_csv(TRAIN_PATH,sep='\t', chunksize=100000, nrows = 100000, quoting=csv.QUOTE_NONE)
count = 0
outp = open('../data/temp_data.pkl', 'wb')
for train in pd.read_csv(TRAIN_PATH,sep='\t',chunksize=50000):
    count = count + 1
    if count>=7:
       break
    print("The {%d} file",count)
    product_set = set()
    num_boxes_list = []
    image_h_list = []
    image_w_list = []
    words_len_list = []
    words_list = []
    label_list = []
    label_words_list = []
    boxes_list = []
    boxes_feature_list = []
    pos_list = []
    tt = train
#i = 0
#for tt in tqdm(train):
    #i = i + 1
    print("starting")
    gc.collect()
    temp = list(tt['query'])
    words_len_list.extend([len(q.split()) for q in temp])
    words_list.extend(temp)
    tt['labels_convert_words'] = tt.swifter.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
    temp = list(tt['labels_convert_words'])
    label_words_list.extend(temp)
    tt['boxes_convert'] = tt.swifter.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
    temp = list(tt['boxes_convert'])
    boxes_list.extend(temp)
    tt['feature_convert'] = tt.swifter.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
    temp = list(tt['feature_convert'])
    boxes_feature_list.extend(temp)
    tt['pos'] = tt.swifter.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
    temp = list(tt['pos'])
    pos_list.extend(temp)
    del temp
    del train
    del tt
    gc.collect()
    data = pd.DataFrame({
             'words':words_list,
             'label_words':label_words_list,
             'features':boxes_feature_list,
             'pos':pos_list
            })
    # print(data)
    #Data1 = data
    joblib.dump(data, outp)
outp.close()

# with open('../data/temp_data.pkl', 'wb') as outp:
#     pickle.dump(data, outp)
# print("temp data finish")

