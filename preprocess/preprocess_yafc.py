import os
import re
import glob
import math
import string
import random
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
        if is_present(word):
            return self.vect_dict[word]
        return np.random.randn((self.vect_size), dtype=np.float64)

data_dir = './yafc/'
data_ext = 'snt.aligned'
vocab_length = 996
glove_vec_file = './glove/glove.twitter.27B.25d.txt'

train_file_s1 = 'yafc_informal.train.txt'
val_file_s1 = 'yafc_informal.valid.txt'
test_file_s1 = 'yafc_informal.test.txt'

train_file_s2 = 'yafc_formal.train.txt'
val_file_s2 = 'yafc_formal.valid.txt'
test_file_s2 = 'yafc_formal.test.txt'

files = list(glob.iglob(os.path.join(data_dir, '**/*' + data_ext), recursive=True))
files.sort(key=lambda x: os.path.basename(x)[:-(len(data_ext)+1)])
transtab = str.maketrans('éè—üəāöóŕ–ƒùšïĕ§†ûàäñıáí', 'ee ueaoor fusiestuaaniai', string.punctuation + '¢®£©´™“’”¿¨…\u200b綠嘉豆加因人為義薏仁»œ►•·ºس恭¹♡¡λ發‘˝◄½م喜♥☺æها財')
WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
EMAIL_REGEX = r"""[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"""
NUMBER_REGEX = r"""[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"""
vocabulary_s1 = {}
vocabulary_s2 = {}
num_sentences = 0
sentences_s1 = []
sentences_s2 = []

# Calculates the number of sentences
for filepath in files:
    if os.path.basename(filepath).find('_informal') == -1:
        continue
    lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        num_sentences += len(lines)

ntrain = int(math.floor(0.75 * num_sentences))
nval = int(math.floor(0.05 * num_sentences))
ntest = num_sentences - ntrain - nval
perm = list(range(num_sentences))
random.shuffle(perm)

print ('Number of sentences:', num_sentences)
print ('ntrain\tnval\tntest')
print ('%d\t%d\t%d' % (ntrain, nval, ntest))

tot_words_s1 = 0
# reading sentences in first style
for filepath in files:
    if os.path.basename(filepath).find('_informal') == -1:
        continue
    lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for s in lines:
        web_sites = re.findall(WEB_URL_REGEX, s)
        for w in web_sites:
            s = s.replace(w, '<unk>')
        emails = re.findall(EMAIL_REGEX, s)
        for e in emails:
            s = s.replace(e, '<unk>')
        numbers = re.findall(NUMBER_REGEX, s)
        for n in numbers:
            s = s.replace(n, ' numtok ')
        txt = str(s).lower().translate(transtab).strip().split()
        sentences_s1.append(txt)
        tot_words_s1 += len(txt)

sentences_s1 = [sentences_s1[i] for i in perm]
unprocessed_train_set_s1 = sentences_s1[:ntrain]

# create dictionary for S1
for sent in unprocessed_train_set_s1:
    for word in sent:
        if word != "unk":
            vocabulary_s1[word] = vocabulary_s1.get(word, 0) + 1

tmp_vocab_s1 = {}
for sent in sentences_s1:
    for word in sent:
        if word != "unk":
            tmp_vocab_s1[word] = tmp_vocab_s1.get(word, 0) + 1

print ("Number of words in vocab of training set for S1:", len(vocabulary_s1))
print ("Number of words in vocab for S1", len(tmp_vocab_s1))
print ("Number of total words in S1", tot_words_s1)

# # printing unknown characters
# char_set = []
# unk_words = []
# for word in vocabulary_s1.keys():
#     for c in word:
#         if not (ord(c) >= 97 and ord(c) <= 122):
#             char_set.append(c)
#             unk_words.append(word)
# print (set(char_set))
# print (unk_words)

vocab_list_s1 = list(sorted(vocabulary_s1, key=vocabulary_s1.get, reverse=True))[:vocab_length]

# printing words not present in the glove vector file
glove_reader = GloveVec(glove_vec_file)
not_present_s1 = []
for word in vocab_list_s1:
    if not glove_reader.is_present(word):
        not_present_s1.append(word)
print (not_present_s1)

processed_dataset_s1 = []

for sent in sentences_s1:
    sent = [word if word in vocab_list_s1 else '<unk>' for word in sent]
    processed_dataset_s1.append(' '.join(sent))

train_set_s1 = processed_dataset_s1[:ntrain]
val_set_s1 = processed_dataset_s1[ntrain:(ntrain+nval)]
test_set_s1 = processed_dataset_s1[(ntrain+nval):]

print ('Writing:', train_file_s1)
with open(train_file_s1, 'w') as f:
    for l in train_set_s1:
        f.write("%s\n" % l)

print ('Writing:', val_file_s1)
with open(val_file_s1, 'w') as f:
    for l in val_set_s1:
        f.write("%s\n" % l)

print ('Writing:', test_file_s1)
with open(test_file_s1, 'w') as f:
    for l in test_set_s1:
        f.write("%s\n" % l)

print ()

tot_words_s2 = 0
# reading sentences in second style
for filepath in files:
    if os.path.basename(filepath).find('_formal') == -1:
        continue
    lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for s in lines:
        web_sites = re.findall(WEB_URL_REGEX, s)
        for w in web_sites:
            s = s.replace(w, '<unk>')
        emails = re.findall(EMAIL_REGEX, s)
        for e in emails:
            s = s.replace(e, '<unk>')
        numbers = re.findall(NUMBER_REGEX, s)
        for n in numbers:
            s = s.replace(n, ' numtok ')
        txt = str(s).lower().translate(transtab).strip().split()
        sentences_s2.append(txt)
        tot_words_s2 += len(txt)

sentences_s2 = [sentences_s2[i] for i in perm]
unprocessed_train_set_s2 = sentences_s2[:ntrain]

# create dictionary for S2
for sent in unprocessed_train_set_s2:
    for word in sent:
        if word != "unk":
            vocabulary_s2[word] = vocabulary_s2.get(word, 0) + 1

tmp_vocab_s2 = {}
for sent in sentences_s2:
    for word in sent:
        if word != "unk":
            tmp_vocab_s2[word] = tmp_vocab_s2.get(word, 0) + 1

print ("Number of words in vocab for training set for S2:", len(vocabulary_s2))
print ("Number of words in vocab for S2:", len(tmp_vocab_s2))
print ("Number of total words in S2:", tot_words_s2)

vocab_list_s2 = list(sorted(vocabulary_s2, key=vocabulary_s2.get, reverse=True))[:vocab_length]

# printing words not present in the glove vector file
not_present_s2 = []
for word in vocab_list_s1:
    if not glove_reader.is_present(word):
        not_present_s2.append(word)
print (not_present_s2)

processed_dataset_s2 = []

for sent in sentences_s2:
    sent = [word if word in vocab_list_s2 else '<unk>' for word in sent]
    processed_dataset_s2.append(' '.join(sent))

train_set_s2 = processed_dataset_s2[:ntrain]
val_set_s2 = processed_dataset_s2[ntrain:(ntrain+nval)]
test_set_s2 = processed_dataset_s2[(ntrain+nval):]

print ('Writing:', train_file_s2)
with open(train_file_s2, 'w') as f:
    for l in train_set_s2:
        f.write("%s\n" % l)

print ('Writing:', val_file_s2)
with open(val_file_s2, 'w') as f:
    for l in val_set_s2:
        f.write("%s\n" % l)

print ('Writing:', test_file_s2)
with open(test_file_s2, 'w') as f:
    for l in test_set_s2:
        f.write("%s\n" % l)
