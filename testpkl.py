import pandas as pd
import pickle
import swifter
from DataTransfer import convertBoxes
from DataTransfer import convertFeature
from DataTransfer import convertLabel
from DataTransfer import convertLabelWord
from DataTransfer import convertPos

TEST_PATH = '/media/hilbert/76f20eb8-f082-44ed-b471-831eb5c2cf00/Tianchi_Competition/Multimodal-Recall/data/testA/testA.tsv'
test = pd.read_csv(TEST_PATH,sep='\t')
test['boxes_convert'] = test.swifter.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
test['feature_convert'] = test.swifter.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
test['labels_convert'] = test.swifter.apply(lambda x: convertLabel(x['num_boxes'], x['class_labels']), axis=1)
test['label_words'] = test.swifter.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
test['pos'] = test.swifter.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
del test['boxes'], test['features'], test['class_labels']
with open('../data/test_data.pkl', 'wb') as outp:
    pickle.dump(test, outp)
print("test data finish")