"""
This file contains auxiliary functions common across main.py and eval.py
"""
import numpy as np

class GloveVec:
    def __init__(self, file_name):
        self.vect_dict = {}
        self.vect_size = None
        print ('Loading GloVe vectors from %s...' % (file_name)),
        f = open(file_name, 'r')
        for l in f.readlines():
            line = l.strip().split()
            word = line[0]
            vec = np.array([ float(x) for x in line[1:] ], dtype=np.float64)
            self.vect_dict[word] = vec
            if self.vect_size is None:
                self.vect_size = len(line[1:])
        print ('done')

    def is_present(self, word):
        return (word in self.vect_dict)

    def get_vector(self, word):
        if self.is_present(word):
            return self.vect_dict[word]
        return np.random.randn((self.vect_size))

def get_sentence_from_np(tokens, loader, src=True):
    sent_strings = []
    if src:
        eos_tok = loader.vocab['<eos>']
        word_dict = loader.inv_vocab
    else:
        eos_tok = loader.pvocab['<eos>']
        word_dict = loader.inv_pvocab
    for sent in tokens:
        l = []
        for word_idx in sent:
            if word_idx == eos_tok:
                break
            l.append(word_dict[word_idx])
        sent_strings.append(' '.join(l))
    return sent_strings


