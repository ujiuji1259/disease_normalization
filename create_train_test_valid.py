from sklearn.model_selection import train_test_split

with open('datasets/triplet.txt', 'r') as f:
    lines = [l for l in f.read().split('\n') if l != '']

train, test = train_test_split(lines, test_size=0.2, shuffle=True, random_state=0)
valid, test = train_test_split(test, test_size=0.5, shuffle=True, random_state=0)

with open('datasets/train.txt', 'w') as f:
    f.write('\n'.join(train))
with open('datasets/valid.txt', 'w') as f:
    f.write('\n'.join(valid))
with open('datasets/test.txt', 'w') as f:
    f.write('\n'.join(test))
