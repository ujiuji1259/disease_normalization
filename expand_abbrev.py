import re
import mojimoji
import pickle

class Converter(object):
    def __init__(self, med_dic, convert_type):
        self.med_dic = med_dic
        self.convert_type = convert_type

    def convert(self, com):
        if self.convert_type == 'all':
            return self.convert_alphabet_to_ja_allpath(com)
        else:
            return self.convert_alphabet_to_ja(com)

    def convert_alphabet_to_ja(self, sent):
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

            if word in self.med_dic:
                s_word = self.med_dic[word]
            elif word.lower() in self.med_dic:
                s_word = self.med_dic[word.lower()]
            else:
                s_word = word

            if s_word == '':
                s_word = word

            output_word += s_word
            pos = e_pos
        while pos < len(sent):
            output_word += sent[pos]
            pos += 1

        return [mojimoji.han_to_zen(output_word)]

    def convert_alphabet_to_ja_allpath(self, sent):
        sent = mojimoji.zen_to_han(sent, kana=False, digit=False)
        iters = re.finditer(r'([a-zA-Z][a-zA-Z\s]*)', sent)
        output_word = [""]
        pos = 0
        for i in iters:
            s_pos, e_pos = i.span()
            word = i.groups()[0]
            word = re.sub('^\s', r'', word)
            word = re.sub('\s$', r'', word)
            s_word = ""

            while pos < s_pos:
                output_word = [token + sent[pos] for token in output_word]
                pos += 1

            if word in self.med_dic:
                s_word = self.med_dic[word]
            elif word.lower() in self.med_dic:
                s_word = self.med_dic[word.lower()]
            else:
                s_word = []

            if s_word == ['']:
                s_word = []

            s_word = list(set([word] + s_word))
            tmp = [output_word for i in range(len(s_word))]
            for i in range(len(tmp)):
                tmp[i] = [t + s_word[i] for t in tmp[i]]
            
            output_word = sum(tmp, [])

            pos = e_pos
        while pos < len(sent):
            output_word = [token + sent[pos] for token in output_word]
            pos += 1

        output_word = [mojimoji.han_to_zen(t) for t in output_word]

        return output_word


if __name__ == "__main__":
    with open('resource/freq.csv', 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']

    med_dic = {}
    for line in lines:
        if len(line.split(',')) <= 3:
            abbrev, word, freq = [re.sub(r'(^\s|\s$)', r'', l) for l in line.split(',')]
            freq = int(freq)
            #if (abbrev in med_dic and med_dic[abbrev][1] < freq) or abbrev not in med_dic:
                #med_dic[abbrev] = [word, freq]
            if abbrev in med_dic:
                med_dic[abbrev].append(word)
            else:
                med_dic[abbrev] = [word]

    sent = "高Ｃａ血症 AML WAS"

    """
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

    """
    with open('resource/med_dic_all.pkl', 'wb') as f:
        pickle.dump(med_dic, f)

    print(convert_alphabet_to_ja_allpath("高Ca血症K", med_dic))
