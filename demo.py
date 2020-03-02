from flask import Flask, render_template, request, Markup
import Levenshtein
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from evaluation_func import most_similar_words
from expand_abbrev import convert_alphabet_to_ja, convert_alphabet_to_ja_allpath

app = Flask(__name__, template_folder='.')
#app.config['DEBUG'] = True

def topk_word_edit_distance(target, words):
    sim = []
    for word in words:
        dist = Levenshtein.distance(word, target)
        dist = 1 - 2 * dist / (len(word) + len(target))
        sim.append(dist)

    sim = np.array(sim)
    words = np.array(words)
    topk = np.argsort(sim)[::-1]
    return words[topk[:10]], sim[topk[:10]]

def topk_word_bert(target, words, vecs, med_dic, ca):
    print(target)
    if ca:
        target = convert_alphabet_to_ja(target, med_dic)
    print(target)
    target = model.encode([target])[0][np.newaxis, :]
    word, sim = most_similar_words(target, vecs, metric='cosine', k=10)
    words = np.array(words)
    return words[word], sim

def topk_word_bert_all(target, words, vecs, med_dic):
    target = convert_alphabet_to_ja_allpath(target, med_dic)
    target = model.encode(target)
    tmp = np.array([])
    tmp_sim = np.array([])
    for t in target:
        word, sim = most_similar_words([t], vecs, metric='cosine', k=10)
        tmp = np.concatenate([tmp, word], 0)
        tmp_sim = np.concatenate([tmp_sim, sim], 0)
    word_list = set()
    res_word = []
    res_sim = []
    rank = np.argsort(tmp_sim)[::-1]

    for r in rank:
        if words[tmp[r]] not in word_list:
            res_word.append(words[tmp[r]])
            res_sim.append(tmp_sim[r])
            word_list.add(words[tmp[r]])

        if len(res_word) > 10:
            break

    return res_word, res_sim

@app.route('/', methods=['GET', 'POST'])
def IR():
    if request.method == "POST":
        target = request.form["text"]
        target = target.strip()
        ca = request.form.get("convert") == "True"
        if request.form.get("type") == "edit":
            l,s = topk_word_edit_distance(target, normal_vocab)
        else:
            l,s = topk_word_bert(target, normal_vocab, normal_vecs, med_dic, ca)
    else:
        l = [] 
        s = []

    return render_template('index.html', normal=zip(l, np.round(s, decimals=3)))

if __name__ == "__main__":
    with open('/home/ujiie/disease_normalization/normal_vocab.pkl', 'rb') as f:
        normal = pickle.load(f)

    with open('/home/ujiie/disease_normalization/resource/med_dic.pkl', 'rb') as f:
        med_dic = pickle.load(f)

    print(med_dic['K'])
    output_path = 'output/bert-base-wikipedia-sections-mean-tokens-2020-02-20_14-38-44'
    model = SentenceTransformer(output_path)
    normal_vocab = normal['vocab']
    
    normal_vecs = normal["vec"]
    app.run(host='0.0.0.0', port=8000)
