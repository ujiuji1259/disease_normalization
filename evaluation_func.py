import numpy as np
import Levenshtein
from tqdm import tqdm

def load_normal_disease_set():
    with open('datasets/S.txt', 'r') as f:
        lines = [line.split('\t')[2] for line in f.read().split('\n') if line != '']

    normal = set(lines)
    return normal

def most_similar_words_edit_distance(targets, normal_list, k=1):
    res = []
    for t in tqdm(targets):
        sim = []
        for token in normal_list:
            dist = Levenshtein.distance(t, token)
            dist = 1 - 2 * dist / (len(t) + len(token))
            sim.append(dist)

        sim = np.array(sim)
        rank = np.argsort(sim)[::-1][0]
        res.append(normal_list[rank])
    return res  


def most_similar_words(targets, normal_list, metric='euclid', k=None):
    if metric == 'cosine':
        norm_target = np.linalg.norm(targets, ord=2, axis=1)[:, np.newaxis]
        norm_normal_list = np.linalg.norm(normal_list, ord=2, axis=1)[:, np.newaxis]
        targets /= norm_target
        normal_list /= norm_normal_list
        sim = normal_list @ targets.T
        if k is None:
            idx = np.argmax(sim, axis=0).reshape(-1)
            return idx
        else:
            idx = np.argsort(sim, axis=0).reshape(-1)[::-1]
            return idx[:k], sim[idx, :].reshape(-1)[:k]
    elif metric == 'euclid':
        idx = 1000
        dist = normal_list[:idx] - targets[:, np.newaxis]
        sim = np.linalg.norm(dist, ord=2, axis=2)
        for i in tqdm(range(idx, normal_list.shape[0], idx)):
            dist = normal_list[i:i+idx] - targets[:, np.newaxis]
            sim = np.hstack([sim, np.linalg.norm(dist, ord=2, axis=2)])

        idx = np.argmin(sim, axis=1)
        return idx, sim[:, idx].reshape(-1)



def load_test_data(path):
    with open(path, 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']
    lines = [line.split('\t') for line in lines]

    test_x = [line[1] for line in lines]
    test_y = [line[0] for line in lines]

    return test_x, test_y
    
    
    
    
