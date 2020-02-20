import numpy as np
with open("datasets/A.txt", 'r') as f:
    lines = [l for l in f.read().split("\n") if l != '']
    lines = [l.split('\t') for l in lines]
    lines = [l for l in lines if l[2] != "" and l[2] != '-1']
    lines = [l for l in lines if len(l[2].split('，')) == 1]

with open("datasets/B.txt", 'r') as f:
    tmp = [l for l in f.read().split("\n") if l != '']
    tmp = [l.split('\t') for l in tmp]
    tmp = [l for l in tmp if l[2] != "" and l[2] != '-1']
    tmp = [l for l in tmp if len(l[2].split('，')) == 1]
    lines.extend(tmp)

with open("datasets/C.txt", 'r') as f:
    tmp = [l for l in f.read().split("\n") if l != '']
    tmp = [l.split('\t') for l in tmp]
    tmp = [l for l in tmp if l[2] != "" and l[2] != '-1']
    tmp = [l for l in tmp if len(l[2].split('，')) == 1]
    lines.extend(tmp)

ICD = {}

for line in lines:
    try:
        ICD[line[1][0]].append([line[0], line[2]])
    except:
        ICD[line[1][0]] = [[line[0], line[2]]]

output = []
output_idx = {}
idx = 0
key_sorted = sorted(ICD.keys())
for key in key_sorted:
    output.extend(ICD[key])
    output_idx[key] = [idx, idx+len(ICD[key])]
    idx += len(ICD[key])

triplet = []
for key in key_sorted:
    left, right = output_idx[key]
    if left == 0:
        negative_set = output[right:]
    else:
        negative_set = output[:left] + output[right:]

    negative_set = [n[0] for n in negative_set]

    num = len(output[left:right])
    negatives = np.random.choice(negative_set, num)

    for tokens, negative in zip(output[left:right], negatives):
        anchor = tokens[1]
        positive = tokens[0]

        triplet.append('\t'.join([anchor, positive, negative]))

with open('datasets/triplet.txt', 'w') as f:
    f.write('\n'.join(triplet))
