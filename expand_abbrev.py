import re
import mojimoji
import pickle

def convert_alphabet_to_ja(sent, dic):
    sent = mojimoji.zen_to_han(sent, kana=False, digit=False)
    iters = re.finditer(r'([a-zA-Z][a-zA-Z\s]*)', sent)
    output_word = ""
    pos = 0
    for i in iters:
        s_pos, e_pos = i.span()
        word = i.groups()[0]
        word = re.sub('^\s', r'', word)
        word = re.sub('\s$', r'', word)
        s_word = ""

        while pos < s_pos:
            output_word += sent[pos]
            pos += 1

        if len(word) == 1:
            s_word = word
        elif word in dic:
            s_word = dic[word]
        elif word.lower() in dic:
            s_word = dic[word.lower()]
        else:
            s_word = word

        output_word += s_word
        pos = e_pos
    while pos < len(sent):
        output_word += sent[pos]
        pos += 1

    return mojimoji.han_to_zen(output_word)

if __name__ == "__main__":
    with open('resource/freq.csv', 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']

    med_dic = {}
    for line in lines:
        if len(line.split(',')) <= 3:
            abbrev, word, freq = [re.sub(r'(^\s|\s$)', r'', l) for l in line.split(',')]
            freq = int(freq)
            if abbrev == "Ca":
                print(word)
            if (abbrev in med_dic and med_dic[abbrev][1] < freq) or abbrev not in med_dic:
                med_dic[abbrev] = [word, freq]

    sent = "高Ｃａ血症 AML WAS"
    for word in med_dic.keys():
        med_dic[word] = med_dic[word][0]

    with open('datasets/test.txt', 'r') as f:
        sents = [sent.split('\t') for sent in f.read().split('\n') if line != '']

    output = []
    change_word = []
    for sent in sents:
        tmp = []
        tmp.append(sent[0])
        tmp.append(convert_alphabet_to_ja(sent[1], med_dic))
        tmp.append(convert_alphabet_to_ja(sent[2], med_dic))
        if tmp[1] != sent[1]:
            change_word.append('\t'.join([sent[1], tmp[1], sent[0]]))

        output.append('\t'.join(tmp))

    with open('resource/convert_alpha.txt', 'w') as f:
        f.write('\n'.join(change_word))

    with open('datasets/test_convert_alphabet.txt', 'w') as f:
        f.write('\n'.join(output))

    with open('resource/med_dic.pkl', 'wb') as f:
        pickle.dump(med_dic, f)

    print(convert_alphabet_to_ja("高Ca血症", med_dic))
