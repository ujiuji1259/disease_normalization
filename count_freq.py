import glob
import re
import mojimoji
from tqdm import tqdm

with open('resource/DATA_IM_v9.txt', 'r') as f:
    lines = [line.split('\t')[7] for line in f.read().split('\n')[1:] if line != '']
    docs = [line.replace('＜ＢＲ＞', '') for line in lines]

files = glob.glob('datasets/2016-略語16000/*.txt')
med_dic = {}
for f in files:
    with open(f, 'r') as g:
        lines = [line.split('　') for line in g.read().split('\n')[1:] if line != '']
        lines = [line for line in lines if len(line) > 2]
        for line in lines:
            tokens = re.sub(r'(（.*?）|［.*?］|\(.*?\))', r'', line[2])
            expands = re.sub(r'(（.*?）|［.*?］|\(.*?\))', r'', line[1])
            tokens = [token for token in re.split('[，,]', tokens.split('。')[0])]
            expands = [token for token in re.split('[，,]', expands)]
            abbrev = re.sub(r'(（.*?）|［.*?］|\(.*?\))', r'', line[0])
            abbrev = [token for token in re.split('[，,]', abbrev)]
            for token in tokens:
                try:
                    med_dic[token].extend(expands)
                    med_dic[token].extend(abbrev)
                except:
                    med_dic[token] = expands + abbrev
                    if token == 'カルシウム':
                        print(med_dic[token], expands, abbrev)

words = med_dic.keys()
freq = {}
for word in tqdm(words):
    for doc in docs:
        iters = re.findall(mojimoji.han_to_zen(word), doc)
        try:
            freq[word] += len(iters)
        except:
            freq[word] = len(iters)

output = []
for word in words:
    for token in med_dic[word]:
        if word == "アルブミン":
            print(token, word)
        output.append(','.join([token, word, str(freq[word])]))

with open('resource/freq.csv', 'w') as f:
    f.write('\n'.join(output))
