import os
import re
import glob
import math
import string
import random
import numpy as np

data_dir_s1 = './bible/DARBY'
data_dir_s2 = './bible/YLT'
data_ext = 'txt'
vocab_length = 996

train_file_s1 = 'bible_darby.train.txt'
val_file_s1 = 'bible_darby.valid.txt'
test_file_s1 = 'bible_darby.test.txt'

train_file_s2 = 'bible_ylt.train.txt'
val_file_s2 = 'bible_ylt.valid.txt'
test_file_s2 = 'bible_ylt.test.txt'

files_s1 = list(glob.iglob(os.path.join(data_dir_s1, '**/*' + data_ext), recursive=True))
files_s1.sort(key=lambda x: os.path.basename(x)[:-(len(data_ext)+1)])
files_s2 = list(glob.iglob(os.path.join(data_dir_s2, '**/*' + data_ext), recursive=True))
files_s2.sort(key=lambda x: os.path.basename(x)[:-(len(data_ext)+1)])

transtab = str.maketrans('—', ' ', string.punctuation + '£…')
# WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
# EMAIL_REGEX = r"""[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"""
NUMBER_REGEX = r"""[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"""
vocabulary_s1 = {}
vocabulary_s2 = {}
num_sentences = 0
sentences_s1 = []
sentences_s2 = []

# Calculates the number of sentences
for filepath in files_s1:
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
for filepath in files_s1:
    lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for s in lines:
        # web_sites = re.findall(WEB_URL_REGEX, s)
        # for w in web_sites:
        #     s = s.replace(w, '<unk>')
        # emails = re.findall(EMAIL_REGEX, s)
        # for e in emails:
        #     s = s.replace(e, '<unk>')
        numbers = re.findall(NUMBER_REGEX, s)
        for n in numbers:
            s = s.replace(n, ' numtok ')
        txt = str(s).lower().translate(transtab).strip().split()
        sentences_s1.append(txt[1:])
        tot_words_s1 += len(txt) - 1

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

# printing unknown characters
char_set = []
unk_words = []
for word in vocabulary_s1.keys():
    for c in word:
        if not (ord(c) >= 97 and ord(c) <= 122):
            char_set.append(c)
            unk_words.append(word)
print (set(char_set))
print (unk_words)

vocab_list_s1 = list(sorted(vocabulary_s1, key=vocabulary_s1.get, reverse=True))[:vocab_length]

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
for filepath in files_s2:
    lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for s in lines:
        # web_sites = re.findall(WEB_URL_REGEX, s)
        # for w in web_sites:
        #     s = s.replace(w, '<unk>')
        # emails = re.findall(EMAIL_REGEX, s)
        # for e in emails:
        #     s = s.replace(e, '<unk>')
        numbers = re.findall(NUMBER_REGEX, s)
        for n in numbers:
            s = s.replace(n, ' numtok ')
        txt = str(s).lower().translate(transtab).strip().split()
        sentences_s2.append(txt[1:])
        tot_words_s2 += len(txt) - 1

print (len(sentences_s2))
print (len(perm))
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
