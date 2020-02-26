import glob
import pickle
import mojimoji
import os
import re
cd = os.getcwd()
disease_set = set()
neg = 0
total = 0
with open(cd + '/datasets/train.txt', 'r') as f:
    lines = [line for line in f.read().split('\n') if line != ""]
    for line in lines:
        if re.search(r"[a-zA-Z]", mojimoji.zen_to_han(line)) is not None:
            print(line)
            neg += 1
        tokens = line.split('\t')
        disease_set.add(tokens[-1])
        total += 1
print(neg, total)
files = glob.glob(cd + "/datasets/2016-略語16000/*.txt")
dic = {}
for file in files:
    with open(file, 'r') as f:
        lines = [line for line in f.read().split('\n')[1:]]
        lines = [line.split("　") for line in lines]

    for line in lines:
        if len(line) == 3 and line[1] not in dic:
            word = line[2].split('。')[0]
            word = re.sub(r"（(.*?)）", r"，\1", word)
            word = word.split('，')
            for w in word:
                if w in disease_set:
                    neg += 1
                total += 1

print(neg, total)



