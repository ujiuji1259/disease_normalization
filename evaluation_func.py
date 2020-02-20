import cupy as cp
import numpy as np

def load_normal_disease_set():
    with open('datasets/S.txt', 'r') as f:
        lines = [line.split('\t')[2] for line in f.read().split('\n') if line != '']

    normal = set(lines)
    return normal

def most_similar_words(targets, normal_list, metric='euclid'):
    if metric == 'cosine':
        norm_target = np.linalg.norm(targets, ord=2, axis=1)[:, np.newaxis]
        norm_normal_list = np.linalg.norm(normal_list, ord=2, axis=1)[:, np.newaxis]
        targets /= norm_target
        normal_list /= norm_normal_list
        sim = normal_list @ targets.T
        idx = np.argmax(sim, axis=0)
        return idx, sim[idx, :]
    elif metric == 'euclid':
        idx = 100
        dist = normal_list[:idx] - targets[:, np.newaxis]
        sim = np.linalg.norm(dist, ord=2, axis=2)
        for i in range(idx, normal_list.shape[0], idx):
            dist = normal_list[i:i+idx] - targets[:, np.newaxis]
            sim = np.hstack([sim, np.linalg.norm(dist, ord=2, axis=2)])
        idx = np.argmax(sim, axis=1)
        return idx, sim[:, idx]



def load_test_data(path):
    with open(path, 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']
    lines = [line.split('\t') for line in lines]

    test_x = [line[1] for line in lines]
    test_y = [line[0] for line in lines]

    return test_x, test_y
    
    
    
    
