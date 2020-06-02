import numpy as np
import base64

LABEL_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/multimodal_labels.txt'
def get_label(path):
    with open(path) as f:
        lines = f.readlines()
        label2id = {l.split('\n')[0].split('\t')[1]:int(l.split('\n')[0].split('\t')[0]) for l in lines[1:]}
        id2label = {int(l.split('\n')[0].split('\t')[0]):l.split('\n')[0].split('\t')[1] for l in lines[1:]}
    return label2id, id2label

label2id, id2label = get_label(LABEL_PATH)

def convertBoxes(num_boxes, boxes):
    return np.frombuffer(base64.b64decode(boxes), dtype=np.float32).reshape(num_boxes, 4)
def convertFeature(num_boxes, features):
    return np.frombuffer(base64.b64decode(features), dtype=np.float32).reshape(num_boxes, 2048)
def convertLabel(num_boxes, label):
    return np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
def convertLabelWord(num_boxes, label):
    temp = np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
    return '###'.join([id2label[t] for t in temp])
def convertPos(num_boxes, boxes, H, W):
    pos_list = []
    for i in range(num_boxes):
        temp = boxes[i]
        pos_list.append([temp[0]/W,
                         temp[2]/W,
                         temp[1]/H,
                         temp[3]/H,
                         ((temp[2] - temp[0]) * (temp[3] - temp[1]))/ (W*H)])
    return pos_list