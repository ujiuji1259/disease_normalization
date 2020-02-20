"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.

See docs/pretrained-models/wikipedia-sections-modesl.md for further details.

You can get the dataset by running examples/datasets/get_data.py
"""

import pickle
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader, MyReader
from sentence_transformers.evaluation import TripletEvaluator, EmbeddingSimilarityEvaluator
from datetime import datetime

import numpy as np
import csv
import logging
from evaluation_func import most_similar_words, load_normal_disease_set, load_test_data


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

output_path = "output/bert-base-wikipedia-sections-mean-tokens-2020-02-20_14-38-44"

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(output_path)
normal_set = list(load_normal_disease_set())
test_x, test_normal = load_test_data('datasets/test.txt')
#normal_list = np.array(model.encode(normal_set))
target = np.array(model.encode(test_x))

with open('normal_vec.pkl', 'rb') as f:
    normal_list = pickle.load(f)

word, sim = most_similar_words(target, normal_list, metric='cosine')
normal_set = np.array(normal_set)

res = ["出現形\t正解\t予測"]
for origin, normal, test in zip(test_x, normal_set[word], test_normal):
    res.append("\t".join([origin, test, normal]))

with open('result/cosine_result.txt', 'w') as f:
    f.write('\n'.join(res))


