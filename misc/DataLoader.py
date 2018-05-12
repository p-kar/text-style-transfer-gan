import h5py
import random
import numpy as np
import pdb
import json
import argparse

class DataLoader(object):

  def __init__(self, opts):
    self.h5_file = h5py.File(opts.dataFile, 'r')

    self.data = {}

    # Sentences
    self.data['train'] = np.array(self.h5_file['train'])
    self.data['val'] = np.array(self.h5_file['val'])
    self.data['test'] = np.array(self.h5_file['test'])

    # Masks
    self.data['train_mask'] = np.array(self.h5_file['train_mask'])
    self.data['val_mask'] = np.array(self.h5_file['val_mask'])
    self.data['test_mask'] = np.array(self.h5_file['test_mask'])

    # Dataset statistics
    self.train_count = self.h5_file['train'].shape[0]
    self.val_count = self.h5_file['val'].shape[0]
    self.test_count = self.h5_file['test'].shape[0]

    json_data = {}
    with open(opts.jsonFile, 'r') as firstFile:
      json_data = json.load(firstFile)
    self.args = json_data['args']
    self.vocab = json_data['vocabulary']
    self.inv_vocab = {v: k for k, v in self.vocab.iteritems()}
    self.vocab_size = json_data['vocabulary_size']

    # Same stats for parallel corpus
    self.ph5_file = h5py.File(opts.pdataFile, 'r')

    self.pdata = {}

    # Sentences
    self.pdata['train'] = np.array(self.ph5_file['train'])
    self.pdata['val'] = np.array(self.ph5_file['val'])
    self.pdata['test'] = np.array(self.ph5_file['test'])

    # Masks
    self.pdata['train_mask'] = np.array(self.ph5_file['train_mask'])
    self.pdata['val_mask'] = np.array(self.ph5_file['val_mask'])
    self.pdata['test_mask'] = np.array(self.ph5_file['test_mask'])

    # Dataset statistics
    self.ptrain_count = self.ph5_file['train'].shape[0]
    self.pval_count = self.ph5_file['val'].shape[0]
    self.ptest_count = self.ph5_file['test'].shape[0]

    json_data = {}
    with open(opts.pjsonFile, 'r') as secondFile:
      json_data = json.load(secondFile)
    self.pargs = json_data['args']
    self.pvocab = json_data['vocabulary']
    self.inv_pvocab = {v: k for k, v in self.pvocab.iteritems()}
    self.pvocab_size = json_data['vocabulary_size']

    # Iteration tracker 
    self.train_idx = 0
    self.val_idx = 0
    self.test_idx = 0 

    self.h5_file.close()

    self.batch_size = opts.batch_size

    # Shuffle the training data indices and access them in the shuffled order

    self.shuffle = opts.shuffle

    self.shuffled_idx = list(range(self.data['train'].shape[0]))

    if self.shuffle:

      random.shuffle(self.shuffled_idx)

    
  # Returns the next training batch (indexed by self.shuffled_idx and starting at self.train_idx)
  def next_batch_train(self):
    
    # Batch size to extract
    batch_size = min(self.batch_size, self.train_count - self.train_idx)

    # Extract batch from dataset
    out = np.array(self.data['train'][self.shuffled_idx[self.train_idx:(self.train_idx + batch_size)]])
    out_mask = np.array(self.data['train_mask'][self.shuffled_idx[self.train_idx:(self.train_idx + batch_size)]])

    #Extract batch from parallel dataset
    pout = np.array(self.pdata['train'][self.shuffled_idx[self.train_idx:(self.train_idx + batch_size)]])
    pout_mask = np.array(self.pdata['train_mask'][self.shuffled_idx[self.train_idx:(self.train_idx + batch_size)]])

    if self.train_idx + batch_size >= self.train_count:

      depleted = True
      self.train_idx = 0

    else:

      depleted = False
      self.train_idx = self.train_idx + batch_size

    return out, out_mask, pout, pout_mask, depleted

  # Returns the next training batch (indexed by self.shuffled_idx and starting at self.train_idx)
  def next_batch_val(self):
    
    # Batch size to extract
    batch_size = min(self.batch_size, self.val_count - self.val_idx)

    # Extract batch from dataset
    out = np.array(self.data['val'][self.val_idx:(self.val_idx + batch_size)])
    out_mask = np.array(self.data['val_mask'][self.val_idx:(self.val_idx + batch_size)])

    #Extract batch from parallel dataset
    pout = np.array(self.pdata['val'][self.val_idx:(self.val_idx + batch_size)])
    pout_mask = np.array(self.pdata['val_mask'][self.val_idx:(self.val_idx+batch_size)])

    if self.val_idx + batch_size >= self.val_count:

      depleted = True
      self.val_idx = 0

    else:

      depleted = False
      self.val_idx = self.val_idx + batch_size

    return out, out_mask, pout, pout_mask, depleted

  # Returns the next training batch (indexed by self.shuffled_idx and starting at self.train_idx)
  def next_batch_test(self):
    
    # Batch size to extract
    batch_size = min(self.batch_size, self.test_count - self.test_idx)

    # Extract batch from dataset
    out = np.array(self.data['test'][self.test_idx:(self.test_idx + batch_size)])
    out_mask = np.array(self.data['test_mask'][self.test_idx:(self.test_idx + batch_size)])

    #Extract batch from parallel dataset
    pout = np.array(self.pdata['test'][self.test_idx:(self.test_idx + batch_size)])
    pout_mask = np.array(self.pdata['test_mask'][self.test_idx:(self.test_idx+batch_size)])

    if self.test_idx + batch_size >= self.test_count:

      depleted = True
      self.test_idx = 0

    else:

      depleted = False
      self.test_idx = self.test_idx + batch_size

    return out, out_mask, pout, pout_mask, depleted

# # Add Arguments
# parser = argparse.ArgumentParser()

# parser.add_argument("-d", "--dataFile", help='File path for dataset 1', required=True, type=str)
# parser.add_argument("-j", "--jsonFile", help='JSON file path for dataset 1', required=True, type=str)
# parser.add_argument("-pd", "--pdataFile", help='File path for dataset 2', required=True, type=str)
# parser.add_argument("-pj", "--pjsonFile", help='JSON file path for dataset 2', required=True, type=str)
# parser.add_argument("-b", "--batch_size", help='Batch size', required=True, type=int)
# parser.add_argument("-s", "--shuffle", help='True or False for shuffling training set across epochs', required=True, type=bool)
# opts = parser.parse_args()

# x = DataLoader(opts)

# for _ in xrange(5):
#   out, out_mask, pout, pout_mask, depleted = x.next_batch_test()
#   print out.shape, pout.shape
#   # print out, out_mask
#   # print pout, pout_mask
#   print x.test_idx