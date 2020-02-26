with open('result/SBERT_aug_alphabet_result.txt', 'r') as f:
    lines = [line for line in f.read().split('\n') if line != '']

pos = 0
neg = 0
positive_example = []
negative_example = []
for line in lines:
    word, true, pred = line.split('\t')
    if true == pred:
        pos += 1
        positive_example.append(line)
    else:
        neg += 1
        negative_example.append(line)

print(pos/(pos+neg))
with open('result/aug_alphabet_true_example.txt', 'w') as f:
    f.write('\n'.join(positive_example))
with open('result/aug_alphabet_false_example.txt', 'w') as f:
    f.write('\n'.join(negative_example))
