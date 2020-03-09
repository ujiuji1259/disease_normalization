import os
import pickle
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path('__file__').resolve().parent))
from evaluation_func import find_similar_words, load_normal_disease_set
from expand_abbrev import Converter
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

cwd = os.getcwd()
output_path = "output/bert-base-wikipedia-sections-mean-tokens-2020-02-20_14-38-44"

with open(os.path.join(cwd, 'resource/MANBYO_v8_SABCDEF_v15.csv'), 'r') as f:
    lines = f.read().split('\n')
    lines = [line.split(',') for line in lines if line != '']
    coms = [line[1] for line in lines[1:]]

with open('resource/med_dic_all.pkl', 'rb') as f:
    med_dic = pickle.load(f)

converter = Converter(med_dic, convert_type='all')
normal_set = list(load_normal_disease_set('datasets/normal_set.txt'))
model = SentenceTransformer(output_path)
input_set = [converter.convert(token) for token in coms]

input_set_length = [len(sent) for sent in input_set]
input_set = sum(input_set, [])
normal_list = model.encode(normal_set)

targets = model.encode(input_set)

"""
with open('resource/normal_vecs.pkl', 'wb') as f:
    pickle.dump(normal_list, f)
with open('resource/target_list.pkl', 'wb') as f:
    pickle.dump(targets, f)
with open('resource/input_set_length.pkl', 'rb') as f:
    input_set_length = pickle.load(f)

#with open('resource/normal_vecs.pkl', 'rb') as f:
    #normal_list = pickle.load(f)

with open('resource/target_list.pkl', 'rb') as f:
    targets_list = pickle.load(f)

"""

batch_size = 100
all_cnt = 0
res_words = []

for length in range(0, len(input_set_length), batch_size):
    tmp_l = input_set_length[length:min(length+batch_size, len(input_set_length))]
    tmp_targets = targets[all_cnt:all_cnt+sum(tmp_l)]
    all_cnt += sum(tmp_l)
    
    idx, sim = find_similar_words(tmp_targets, normal_list, k=1)

    cnt = 0
    for l in tmp_l:
        tmp_idx = idx[cnt:cnt+l, :].reshape(-1)
        tmp_sim = sim[cnt:cnt+l, :].reshape(-1)
        rank = np.argsort(tmp_sim)[::-1][0]
        res_words.append(normal_set[tmp_idx[rank]])
        cnt += l

output = [','.join(lines[0] + ['標準病名（SBERT）'])]
for l, c in zip(lines[1:], res_words):
    output.append(','.join(l + [c]))

with open('resource/MANBYO_v8_SABCDEF_v15_SBERT.csv', 'w') as f:
    f.write('\n'.join(output))

assert len(res_words) == len(coms), 'owijfe'
