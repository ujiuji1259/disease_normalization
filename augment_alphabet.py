import re
import mojimoji
with open('datasets/train.txt', 'r') as f:
    lines = [line for line in f.read().split('\n') if line != '']

output = []
for line in lines:
    l = mojimoji.zen_to_han(line)
    if re.search(r'[a-zA-Z]', l) is not None:
        output.extend([line] * 10)
    else:
        output.append(line)

with open('datasets/train_augmented.txt', 'w') as f:
    f.write('\n'.join(output))
