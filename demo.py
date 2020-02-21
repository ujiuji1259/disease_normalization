from flask import Flask, render_template, request, Markup
import Levenshtein
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from evaluation_func import most_similar_words

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
    return words[topk[:10]]

def topk_word_bert(target, words, vecs):
    target = model.encode([target])[0][np.newaxis, :]
    word, sim = most_similar_words(target, vecs, metric='cosine', k=10)
    words = np.array(words)
    return words[word]

@app.route('/', methods=['GET', 'POST'])
def IR():
    if request.method == "POST":
        target = request.form["text"]
        target = target.strip()
        if request.form.get("type") == "edit":
            l = topk_word_edit_distance(target, normal_vocab)
        else:
            l = topk_word_bert(target, normal_vocab, normal_vecs)
    else:
        l = [] 

    return render_template('index.html', normal=l)

if __name__ == "__main__":
    with open('/home/ujiie/disease_normalization/normal_vocab.pkl', 'rb') as f:
        normal = pickle.load(f)
    output_path = 'output/bert-base-wikipedia-sections-mean-tokens-2020-02-20_14-38-44'
    model = SentenceTransformer(output_path)
    normal_vocab = normal['vocab']
    
    normal_vecs = normal["vec"]
    app.run(host='0.0.0.0', port=8000)
