import h5py
import numpy as np
import json

class PreprocessH5(object):
  def __init__(self, train_filename, test_filename, val_filename, target_filename, metadata_filename, max_length):
    self.train_filename = train_filename
    self.test_filename = test_filename
    self.val_filename = val_filename
    self.target_filename = target_filename
    self.max_length = max_length
    self.metadata_filename = metadata_filename
    self.vocab = {'<sos>' : 0, '<eos>' : 1, '<pad>' : 2}
    self.vocab_size = len(self.vocab)

  def preprocess_h5_file(self, mode, filename, max_length):
    f = open(filename, 'r')
    sentences_txt = f.readlines()
    f.close()

    # Initialize numpy array with (batch_size, max_length)
    sentences = np.full((len(sentences_txt), max_length), self.vocab['<pad>'], dtype = np.int)
    mask = np.full((len(sentences_txt), max_length), 0, dtype = np.int)
    s_idx = 0

    for sentence in sentences_txt:
      
      words = sentence.strip('\n').split()
      words = words[:max_length - 1]
      
      for i in range(len(words)):
        
        if words[i] not in self.vocab.keys():
          if mode == 'train':
            self.vocab[words[i]] = self.vocab_size
            self.vocab_size += 1
          else:
            words[i] = '<unk>'

        sentences[s_idx][i] = self.vocab[words[i]]
        mask[s_idx][i] = 1
      # Add end of sentence tag
      sentences[s_idx][i + 1] = self.vocab['<eos>']
      mask[s_idx][i + 1] = 1
      s_idx += 1
    # print sentences_txt
    return sentences[:], mask[:]

  def preprocess_h5(self):

    train_sentences, train_mask = self.preprocess_h5_file('train', self.train_filename, self.max_length)
    test_sentences, test_mask = self.preprocess_h5_file('test', self.test_filename, self.max_length)
    val_sentences, val_mask = self.preprocess_h5_file('val', self.val_filename, self.max_length)

    # print train_sentences, test_sentences, val_sentences
    # print train_mask, test_mask, val_mask

    # Add embedding vector to hdf5 file  
    f = h5py.File(self.target_filename, 'w')
    f.create_dataset('train', data = train_sentences, dtype = int)
    f.create_dataset('test', data = test_sentences, dtype = int)
    f.create_dataset('val', data = val_sentences, dtype = int)

    f.create_dataset('train_mask', data = train_mask, dtype = int)
    f.create_dataset('test_mask', data = test_mask, dtype = int)
    f.create_dataset('val_mask', data = val_mask, dtype = int)

    self.createMetaDataJSON(self.metadata_filename)

    f.close()

  def createMetaDataJSON(self, filename):

    temp_data = {'args' : [self.train_filename, self.test_filename, self.val_filename, self.target_filename, str(self.max_length)], 
                  'vocabulary' : self.vocab,
                  'vocabulary_size' : self.vocab_size,
                  }
    with open(filename, 'w') as outfile:
      json.dump(temp_data, outfile)
    
# Sample invocation  

# x = PreprocessH5('../preprocess/yafc_informal.train.txt', '../preprocess/yafc_informal.test.txt', '../preprocess/yafc_informal.valid.txt', 'yafc_informal.h5', 'yafc_informal.json', 20)
# x.preprocess_h5()
# x = PreprocessH5('../preprocess/yafc_formal.train.txt', '../preprocess/yafc_formal.test.txt', '../preprocess/yafc_formal.valid.txt', 'yafc_formal.h5', 'yafc_formal.json', 20)
# x.preprocess_h5()

# x = PreprocessH5('../preprocess/bible_darby.train.txt', '../preprocess/bible_darby.test.txt', '../preprocess/bible_darby.valid.txt', 'bible_darby.h5', 'bible_darby.json', 20)
# x.preprocess_h5()
# x = PreprocessH5('../preprocess/bible_ylt.train.txt', '../preprocess/bible_ylt.test.txt', '../preprocess/bible_ylt.valid.txt', 'bible_ylt.h5', 'bible_ylt.json', 20)
# x.preprocess_h5()

x = PreprocessH5('../preprocess/shakespeare_modern.train.txt', '../preprocess/shakespeare_modern.test.txt', '../preprocess/shakespeare_modern.valid.txt', 'shakespeare_modern.h5', 'shakespeare_modern.json', 20)
x.preprocess_h5()
x = PreprocessH5('../preprocess/shakespeare_original.train.txt', '../preprocess/shakespeare_original.test.txt', '../preprocess/shakespeare_original.valid.txt', 'shakespeare_original.h5', 'shakespeare_original.json', 20)
x.preprocess_h5()
