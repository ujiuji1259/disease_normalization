import os
import pickle
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path('__file__').resolve().parent))
from evaluation_func import most_similar_words, load_normal_disease_set, load_test_data, most_similar_words_edit_distance, find_similar_words
from expand_abbrev import convert_alphabet_to_ja, convert_alphabet_to_ja_allpath
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

normal_set = list(load_normal_disease_set())
model = SentenceTransformer(output_path)
input_set = [convert_alphabet_to_ja_allpath(token, med_dic) for token in coms]

input_set_length = [len(sent) for sent in input_set]
input_set = sum(input_set, [])

targets = model.encode(input_set)
normal_list = model.encode(normal_set)

idx, sim = find_similar_words(targets, normal_list, k=1)
res_words = []

cnt = 0
for l in input_set_length:
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
