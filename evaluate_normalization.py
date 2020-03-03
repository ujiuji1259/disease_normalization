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
from tqdm import tqdm

import numpy as np
import csv
import logging
from evaluation_func import most_similar_words, load_normal_disease_set, load_test_data, most_similar_words_edit_distance
from expand_abbrev import convert_alphabet_to_ja, convert_alphabet_to_ja_allpath


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

output_path = "output/bert-base-wikipedia-sections-mean-tokens-2020-02-20_14-38-44"
#output_path = "output/bert-base-alphabet-augment-mean-tokens-2020-02-22_13-23-58"

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

def evaluate_BERT():
    normal_set = list(load_normal_disease_set())
    test_x, test_normal = load_test_data('datasets/test.txt')
    word_embedding_model = models.BERT('bert-base-japanese-char')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    normal_list = np.array(model.encode(normal_set))
    target = np.array(model.encode(test_x))

    word = most_similar_words(target, normal_list, metric='cosine')
    normal_set = np.array(normal_set)

    res = ["出現形\t正解\t予測"]
    for origin, normal, test in zip(test_x, normal_set[word], test_normal):
        res.append("\t".join([origin, test, normal]))

    with open('result/BERT_result.txt', 'w') as f:
        f.write('\n'.join(res))

def evaluate_SBERT():
    normal_set = list(load_normal_disease_set())
    test_x, test_normal = load_test_data('datasets/test_convert_alphabet.txt')
    model = SentenceTransformer(output_path)

    with open('resource/med_dic.pkl', 'rb') as f:
        med_dic = pickle.load(f)

    input_set = [convert_alphabet_to_ja(token, med_dic) for token in normal_set]
    print(input_set)
    #normal_list = np.array(model.encode(normal_set))
    normal_list = np.array(model.encode(input_set))
    target = np.array(model.encode(test_x))

    word = most_similar_words(target, normal_list, metric='cosine')
    normal_set = np.array(normal_set)

    print(normal_set[:10])
    print(word)
    res = ["出現形\t正解\t予測"]
    for origin, normal, test in zip(test_x, normal_set[word], test_normal):
        res.append("\t".join([origin, test, normal]))

    with open('result/SBERT_convert_alphabet_result.txt', 'w') as f:
        f.write('\n'.join(res))

    return normal_set, normal_list

def evaluate_SBERT_convert():
    normal_set = list(load_normal_disease_set())
    test_x, test_normal = load_test_data('datasets/test.txt')
    model = SentenceTransformer(output_path)

    with open('resource/med_dic.pkl', 'rb') as f:
        med_dic = pickle.load(f)

    input_set = [convert_alphabet_to_ja(token, med_dic) for token in normal_set]
    test_input_set = [convert_alphabet_to_ja(token, med_dic) for token in test_x]
    print(input_set)
    #normal_list = np.array(model.encode(normal_set))
    normal_list = np.array(model.encode(input_set))
    target = np.array(model.encode(test_input_set))

    word = most_similar_words(target, normal_list, metric='cosine')
    normal_set = np.array(normal_set)

    print(normal_set[:10])
    print(word)
    res = ["出現形\t正解\t予測"]
    for origin, normal, test in zip(test_x, normal_set[word], test_normal):
        res.append("\t".join([origin, test, normal]))

    with open('result/SBERT_convert_alphabet_result.txt', 'w') as f:
        f.write('\n'.join(res))

    return normal_set, normal_list


def evaluate_SBERT_convert_allpath():
    normal_set = list(load_normal_disease_set())
    test_x, test_normal = load_test_data('datasets/test.txt')
    model = SentenceTransformer(output_path)

    with open('resource/med_dic_all.pkl', 'rb') as f:
        med_dic = pickle.load(f)

    input_set = [convert_alphabet_to_ja_allpath(token, med_dic) for token in test_x]
    normal_list = model.encode(normal_set)
    word = []
    for t in tqdm(input_set):
        tmp = np.array([])
        tmp_sim = np.array([])
        target = np.array(model.encode(t))
        for token in target:
            w, sim = most_similar_words([token], normal_list, metric='cosine', k=10)
            tmp = np.concatenate([tmp, w], 0)
            tmp_sim = np.concatenate([tmp_sim, sim], 0)

        max_idx = np.argmax(tmp_sim)
        word.append(int(tmp[max_idx]))

    normal_set = np.array(normal_set)

    print(normal_set[:10])
    print(word)
    res = ["出現形\t正解\t予測"]
    for origin, normal, test in zip(test_x, normal_set[word], test_normal):
        res.append("\t".join([origin, test, normal]))

    with open('result/SBERT_convert_alphabet_allpath_result.txt', 'w') as f:
        f.write('\n'.join(res))

    return normal_set, normal_list

def evaluate_edit_distance():
    normal_set = list(load_normal_disease_set())
    test_x, test_normal = load_test_data('datasets/test.txt')
    word = most_similar_words_edit_distance(test_x, normal_set)

    res = ["出現形\t正解\t予測"]
    for origin, normal, test in zip(test_x, word, test_normal):
        res.append("\t".join([origin, test, normal]))

    with open('result/edit_distance_result.txt', 'w') as f:
        f.write('\n'.join(res))

normal_set, normal_list = evaluate_SBERT_convert_allpath()

with open('normal_vocab_convert.pkl', 'wb') as f:
    pickle.dump({'vocab':normal_set, 'vec':normal_list}, f)
