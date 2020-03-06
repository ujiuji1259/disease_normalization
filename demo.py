from flask import Flask, render_template, request, Markup
import Levenshtein
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from evaluation_func import most_similar_words, find_similar_words
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

def topk_word_bert(target, words, vecs, dic):
    print(target)
    target = model.encode([target])
    idx, sim = most_similar_words(target, vecs, k=10)
    words = np.array(words)
    idx = idx.reshape(-1)
    sim = sim.reshape(-1)
    return words[idx], sim

def topk_word_bert_all(target, words, vecs, dic):
    print(target)
    target = convert_alphabet_to_ja_allpath(target, dic)
    print(target)
    target = model.encode(target)
    idx, sim = find_similar_words(target, vecs, k=10)
    idx, sim = idx.reshape(-1), sim.reshape(-1)

    rank = np.argsort(sim)[::-1]
    words = np.array(words)

    res_word = []
    res_sim = []
    for w, s in zip(words[idx[rank]], sim[rank]):
        if w not in res_word:
            res_word.append(w)
            res_sim.append(s)
        if len(res_word) > 10:
            break

    return res_word, res_sim

@app.route('/', methods=['GET', 'POST'])
def IR():
    if request.method == "POST":
        target = request.form["text"]
        target = target.strip()
        ca = request.form.get("convert") == "all"

        if request.form.get("type") == "edit":
            l,s = topk_word_edit_distance(target, tmp_set)
        else:
            if ca:
                l,s = topk_word_bert_all(target, normal_vocab, normal_vecs, med_dic_all)
            else:
                print(target)
                target = convert_alphabet_to_ja(target, med_dic)
                l,s = topk_word_bert(target, convert_normal_vocab, convert_normal_vecs, med_dic)
    else:
        l = [] 
        s = []

    return render_template('index.html', normal=zip(l, np.round(s, decimals=3)))

if __name__ == "__main__":
    with open('/home/ujiie/disease_normalization/normal_vocab_convert.pkl', 'rb') as f:
        convert_normal = pickle.load(f)

    with open('/home/ujiie/disease_normalization/normal_vocab.pkl', 'rb') as f:
        normal = pickle.load(f)

    with open('/home/ujiie/disease_normalization/resource/med_dic.pkl', 'rb') as f:
        med_dic = pickle.load(f)

    with open('/home/ujiie/disease_normalization/resource/med_dic_all.pkl', 'rb') as f:
        med_dic_all = pickle.load(f)

    print(med_dic_all['AML'])
    output_path = 'output/bert-base-wikipedia-sections-mean-tokens-2020-02-20_14-38-44'
    model = SentenceTransformer(output_path)
    normal_vocab = normal['vocab']
    normal_vecs = normal['vec']

    convert_normal_vocab = convert_normal['vocab']
    convert_normal_vecs = convert_normal["vec"]
    app.run(host='0.0.0.0', port=8000)
