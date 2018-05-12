import os
import sys
import pdb
import json
import h5py
import time
import random
import argparse
import tensorboardX
from tensorboardX import SummaryWriter

import torch
import torchvision
import torch.optim as optim
import torchvision.utils as vutils
from torch.optim import lr_scheduler

sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), 'misc/'))
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), 'coco-caption/pycocoevalcap/bleu/'))

from loss import *
from utils import *
from models import * 
from DataLoader import * 
from bleu_scorer import BleuScorer

use_cuda = torch.cuda.is_available()

def str2bool(t):
    if t.lower() in ['true', 't', '1']:
        return True
    else:
        return False

def set_random_seeds(seed):
    """
    Sets the random seeds for numpy, python, pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_stylize(G12, G21, loader, opts, split='test'):
    """
    Evaluates sentence generation from both the generators using
    BLEU, CIDER, METEOR, ROUGE-L, SPICE
    """
    depleted = False
    sents_s1_all = [] # GT s1
    sents_s2_hat_all = [] # GT s1 -> s2 hat
    sents_s2_all = [] # GT s2
    sents_s1_hat_all = [] # GT s2 -> s1 hat
    while not depleted:
        # Sents: batch_size x max_length [w1, w2, ..., <eos>, <pad>, <pad>, ...]
        # Masks: batch_size x max_length [ 1,  1, ...,     1,     0,     0, ...]
        if split == 'train':
            sents_s1, masks_s1, sents_s2, masks_s2, depleted = loader.next_batch_train()
        elif split == 'val':
            sents_s1, masks_s1, sents_s2, masks_s2, depleted = loader.next_batch_val() 
        elif split == 'test':
            sents_s1, masks_s1, sents_s2, masks_s2, depleted = loader.next_batch_test()
        
        batch_size = sents_s1.shape[0]
        # Assuming both styles have same max_length
        max_length = sents_s1.shape[1]
        # source input must not contain a start token
        input_sents_s1 = torch.LongTensor(sents_s1)
        input_sents_s2 = torch.LongTensor(sents_s2)
        if use_cuda:
            input_sents_s1 = input_sents_s1.cuda()
            input_sents_s2 = input_sents_s2.cuda()
        input_sents_s1 = Variable(input_sents_s1)
        input_sents_s2 = Variable(input_sents_s2)
        # encode the input source sentence
        input_sents_s1_encoded, hidden_12 = G12.encode(input_sents_s1)
        input_sents_s2_encoded, hidden_21 = G21.encode(input_sents_s2)
        # generate the predicted target
        # initial input must be the start token
        decoder_input_s2_hat = Variable(torch.LongTensor(np.ones((batch_size, 1))*opts.start_idx_s2))
        decoder_input_s1_hat = Variable(torch.LongTensor(np.ones((batch_size, 1))*opts.start_idx_s1))
        if use_cuda:
            decoder_input_s2_hat = decoder_input_s2_hat.cuda()
            decoder_input_s1_hat = decoder_input_s1_hat.cuda()
        rollouts_s2_hat, _ = G12.decoder_rollout(max_length, decoder_input_s2_hat, hidden_12, input_sents_s1_encoded, opts.alpha)
        rollouts_s1_hat, _ = G21.decoder_rollout(max_length, decoder_input_s1_hat, hidden_21, input_sents_s2_encoded, opts.alpha)
        sents_s2_hat = rollouts_s2_hat.data.cpu().numpy().astype(int)
        sents_s1_hat = rollouts_s1_hat.data.cpu().numpy().astype(int)
        
        # computing the string sentences
        sents_s1_all.extend(get_sentence_from_np(sents_s1, loader, src=True))
        sents_s1_hat_all.extend(get_sentence_from_np(sents_s1_hat, loader, src=True))
        sents_s2_all.extend(get_sentence_from_np(sents_s2, loader, src=False))
        sents_s2_hat_all.extend(get_sentence_from_np(sents_s2_hat, loader, src=False))

    # Compute BLEU scores
    bleu_scorer_G21 = BleuScorer(n=4)
    bleu_scorer_G12 = BleuScorer(n=4)
    for i in range(len(sents_s1_all)):
        bleu_scorer_G21 += (sents_s1_hat_all[i], [sents_s1_all[i]])
        bleu_scorer_G12 += (sents_s2_hat_all[i], [sents_s2_all[i]])
    bleu_G21, _ = bleu_scorer_G21.compute_score(option='closest')
    bleu_G12, _ = bleu_scorer_G12.compute_score(option='closest')

    print('BLEU scores for Style 1 to 2 ===> B1: %.3f  B2: %.3f B3: %.3f B4: %.3f'%(bleu_G12[0], bleu_G12[1], bleu_G12[2], bleu_G12[3]))
    print('BLEU scores for Style 2 to 1 ===> B1: %.3f  B2: %.3f B3: %.3f B4: %.3f'%(bleu_G21[0], bleu_G21[1], bleu_G21[2], bleu_G21[3]))

def main(opts):
    loader = DataLoader(opts)

    opts.max_length = int(loader.args[4]) # assuming max length is the same for both sets
    # for source S1
    opts.start_idx_s1 = loader.vocab['<sos>']
    opts.padding_idx_s1 = loader.vocab['<pad>']
    opts.vocab_size_s1 = loader.vocab_size
    # for target S2
    opts.start_idx_s2 = loader.pvocab['<sos>']
    opts.padding_idx_s2 = loader.pvocab['<pad>']
    opts.vocab_size_s2 = loader.pvocab_size

    # for decoder skip connection weight
    opts.alpha = 0
    
    embedding_s1 = EmbeddingLayer(opts.vocab_size_s1, opts.embedding_size, opts.padding_idx_s1)
    embedding_s2 = EmbeddingLayer(opts.vocab_size_s2, opts.embedding_size, opts.padding_idx_s2)

    G12 = Generator(embedding_s1, embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length, opts.use_attention)
    G21 = Generator(embedding_s2, embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length, opts.use_attention)

    loaded_states = torch.load(opts.model_path)
    
    G12.load_state_dict(loaded_states['G12_state_dict'])
    G21.load_state_dict(loaded_states['G21_state_dict'])
    
    if use_cuda:
        G12.cuda()
        G21.cuda()
    G12.eval()
    G21.eval()
    if opts.split == 'all':
        print ('BLEU score on the training set:')
        evaluate_stylize(G12, G21, loader, opts, split='train')
        print ('')
        print ('BLEU score on the validation set:')
        evaluate_stylize(G12, G21, loader, opts, split='val')
        print ('')
        print ('BLEU score on the test set:')
        evaluate_stylize(G12, G21, loader, opts, split='test')
        print ('')
    elif opts.split == 'val_and_test':
        print ('BLEU score on the validation set:')
        evaluate_stylize(G12, G21, loader, opts, split='val')
        print ('')
        print ('BLEU score on the test set:')
        evaluate_stylize(G12, G21, loader, opts, split='test')
        print ('')
    else:
        evaluate_stylize(G12, G21, loader, opts, split=opts.split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # DataLoader
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for training and testing')
    parser.add_argument('--dataFile', type=str, default='data/yafc_formal.h5', help='Preprocessed data stored in h5 format for S1')
    parser.add_argument('--jsonFile', type=str, default='data/yafc_formal.json', help='json file containing vocabulary for S1')
    parser.add_argument('--pdataFile', type=str, default='data/yafc_informal.h5', help='Preprocessed data stored in h5 format for S2')
    parser.add_argument('--pjsonFile', type=str, default='data/yafc_informal.json', help='json file containing vocabulary for S2')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='Shuffle training data?')
    
    # Models
    parser.add_argument('--embedding_size', type=int, default=300, help='Word embedding size')
    parser.add_argument('--hidden_size', type=int, default=350, help='Hidden size for RNN')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='Number of hidden layers in RNN')
    parser.add_argument('--use_lstm', type=str2bool, default=True, help='Use LSTM or GRU?')
    parser.add_argument('--use_attention', type=str2bool, default=True, help='Use attention decoder in the generator')

    # Optimization parameters
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Dropout probability')

    # Model load parameters
    parser.add_argument('--model_path', type=str, default='models/cycle_train/model_best.net', help='Saved model from CycleGAN training')
    parser.add_argument('--split', type=str, default='all', help='split to evaluate on [train | val | test | all | val_and_test].')

    opts = parser.parse_args()
    
    set_random_seeds(123)
    main(opts)
