import os
import sys
import pdb
import json
import h5py
import math
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
from loss import *
from models import * 
from utils import *
from DataLoader import * 

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

def perplexity(input, masks=None):
    # input: B x L array
    if masks is None:
        masks = np.ones_like(input)
    return np.exp(-np.sum(input * masks) / np.sum(masks))

def generate_sample_sentences(lang_model, loader, num_sent=3):
    lang_model.eval()
    input_tokens = np.zeros((num_sent, 1)).astype(int)
    input_tokens[:, :] = loader.vocab['<sos>']
    input_tokens = torch.LongTensor(input_tokens)
    if use_cuda:
            input_tokens = input_tokens.cuda()
    input_tokens = Variable(input_tokens)
    sents = lang_model.sample(input_tokens, max_length=opts.max_length).data.cpu().numpy().astype(int)
    sent_strings = []
    for sent in sents:
        l = []
        for word_idx in sent:
            if word_idx == loader.vocab['<eos>']:
                break
            l.append(loader.inv_vocab[word_idx])
        sent_strings.append(' '.join(l))
    lang_model.train()
    return sent_strings

def evaluate_language_model(lang_model, loader, opts, split='val'):
    lang_model.eval()
    depleted = False
    probs_correct = []
    masks_correct = []
    while not depleted:
        # Sents: batch_size x max_length [w1, w2, ..., <eos>, <pad>, <pad>, ...]
        # Masks: batch_size x max_length [ 1,  1, ...,     1,     0,     0, ...]
        if split == 'train':
            sents, masks, _, _, depleted = loader.next_batch_train()
        elif split == 'val':
            sents, masks, _, _, depleted = loader.next_batch_val() 
        elif split == 'test':
            sents, masks, _, _, depleted = loader.next_batch_test()
        batch_size = sents.shape[0]
        # Input must contain a start token
        input_sents = np.zeros((sents.shape[0], sents.shape[1])).astype(sents.dtype)
        input_sents[:, 0] = opts.start_idx 
        input_sents[:, 1:] = np.copy(sents)[:, :-1] 
        input_sents = torch.LongTensor(input_sents)
        target_sents = np.copy(sents)
        if use_cuda:
            input_sents = input_sents.cuda()
        input_sents = Variable(input_sents)
        # log_probs - batch_size x max_length x vocab_size
        log_probs = lang_model(input_sents).data.cpu().numpy()
        # advanced indexing to compute log probs only at true words
        true_log_probs = log_probs.reshape(-1, opts.vocab_size)[range(batch_size * opts.max_length), target_sents.reshape(-1)]
        true_log_probs = true_log_probs.reshape(batch_size, opts.max_length)
        probs_correct.append(true_log_probs)
        masks_correct.append(masks)

    probs_correct = np.concatenate(probs_correct, axis=0)
    masks_correct = np.concatenate(masks_correct, axis=0)
    
    # reset the model back to train
    lang_model.train()
    return perplexity(probs_correct, masks_correct)

def load_embedding_from_glove(opts, loader, embedding):
    glove_reader = GloveVec(opts.pretrained_glove_vector_path)
    assert (glove_reader.vect_size == opts.embedding_size), "Embedding size mismatch"
    embed_data = np.zeros((loader.vocab_size, opts.embedding_size), dtype=np.float64)
    for word in loader.vocab.keys():
        embed_idx = loader.vocab[word]
        embed_data[embed_idx, :] = glove_reader.get_vector(word)
    embedding.embedding.weight.data.copy_(torch.from_numpy(embed_data))
    del glove_reader

def pretrain_language_model(opts):
    print ("############## Language Model Pretraining ##############")
    print ("########################################################")
    print ('')
    loader = DataLoader(opts)

    opts.start_idx = loader.vocab['<sos>']
    opts.padding_idx = loader.vocab['<pad>']
    opts.max_length = int(loader.args[4])
    opts.vocab_size = loader.vocab_size

    # creating the models (EmbeddingLayer, LanguageModel)
    embedding = EmbeddingLayer(loader.vocab_size, opts.embedding_size, opts.padding_idx)
    if opts.use_glove_embeddings:
        load_embedding_from_glove(opts, loader, embedding)
    lang_model = LanguageModel(embedding, opts.hidden_size, num_rnn_layers=opts.num_rnn_layers, use_lstm=opts.use_lstm, dropout_p=opts.dropout_p)
    # set mode to train
    embedding.train()
    lang_model.train()
    # put parameters to cuda if cuda is available
    if use_cuda:
        embedding.cuda()
        lang_model.cuda()
    # define loss criterion
    criterion = nn.NLLLoss(reduce=False)
    # define optimizer
    optimizer = optim.Adam(lang_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    # for logging
    n_iter = 0
    writer = SummaryWriter(log_dir=opts.log_dir)
    # for choosing best model
    best_val_perplexity = float('inf')

    for _epoch in range(opts.epochs):
        depleted = False
        total_loss = 0.0
        num_sentences = 0.0
        time_start = time.time()

        while not depleted:
            # Sents: batch_size x max_length [w1, w2, ..., <eos>, <pad>, <pad>, ..]
            # Masks: batch_size x max_length [ 1,  1, ...,     1,     0,     0, ...]
            sents, masks, _, _, depleted = loader.next_batch_train()
            batch_size = sents.shape[0]
            # Input must contain a start token
            input_sents = np.zeros((sents.shape[0], sents.shape[1])).astype(sents.dtype)
            input_sents[:, 0] = opts.start_idx
            input_sents[:, 1:] = np.copy(sents)[:, :-1]
            input_sents = torch.LongTensor(input_sents) 
            target_sents = torch.LongTensor(sents)
            masks = torch.Tensor(masks)

            if use_cuda:
                input_sents = input_sents.cuda()
                target_sents = target_sents.cuda()
                masks = masks.cuda()

            input_sents = Variable(input_sents)
            target_sents = Variable(target_sents)
            masks = Variable(masks)
            # Forward pass
            # log_probs - (batch_size, max_length, vocab_size) variable
            log_probs = lang_model(input_sents) 
            # nll_loss_unmasked_batch: (B) Variable

            nll_loss_unmasked_batch = criterion(log_probs.view(-1, opts.vocab_size), target_sents.view(-1))
            nll_loss_batch = (nll_loss_unmasked_batch * masks.view(-1)).sum()
            nll_loss = nll_loss_batch / batch_size

            # For logging
            total_loss += nll_loss.data[0]
            num_sentences += batch_size

            # Backward pass
            optimizer.zero_grad()
            nll_loss.backward()

            # Update parameters
            nn.utils.clip_grad_norm(lang_model.parameters(), opts.max_norm)
            optimizer.step()
            n_iter += 1
        
        time_end = time.time()
        time_taken = time_end - time_start
        avg_train_loss = total_loss / num_sentences
        val_perplexity = evaluate_language_model(lang_model, loader, opts)
        sample_sents = generate_sample_sentences(lang_model, loader, num_sent=opts.num_sample_sents)
        # printing statistics on console
        print ("epoch: %d, updates: %d, time taken: %.2fs, avg. train loss: %.5f, validation perplexity: %.2f." % (_epoch, \
            n_iter, time_taken, avg_train_loss, val_perplexity))
        print ("******************* Sample Sentences *******************")
        print ('\n'.join(sample_sents))
        print ("********************************************************")
        print ('')
        # writing to TensorboardX
        writer.add_scalar('avg_train_loss', avg_train_loss, _epoch)
        writer.add_scalar('val_perplexity', val_perplexity, _epoch)
        # saving the model to disk
        if val_perplexity <= best_val_perplexity:
            best_val_perplexity = val_perplexity
            save_state = {
                'epoch': _epoch,
                'state_dict': lang_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'opts': opts, 
                'val_perplexity': val_perplexity
            }
            model_name = opts.model_name + '_best.net'
            torch.save(save_state, os.path.join(opts.save_path, model_name))

        save_state = {
            'epoch': _epoch,
            'state_dict': lang_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'opts': opts, 
            'val_perplexity': val_perplexity
        }
        model_name = opts.model_name + '_latest.net'
        torch.save(save_state, os.path.join(opts.save_path, model_name)) 
    
    # cleanup TensorboardX summary writer
    writer.close()

def load_embedding_from_lm(embedding, lm_state_dict):
    lm_state_dict = lm_state_dict.copy()
    lm_state_dict['embedding.weight'] = lm_state_dict.pop('embedding.embedding.weight')
    model_dict = embedding.state_dict()
    # filter out unecessary keys
    lm_state_dict = {k: v for k, v in lm_state_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(lm_state_dict)
    # load the new state_dict
    embedding.load_state_dict(model_dict)

def load_encoder_from_lm(G, lm_state_dict):
    if opts.num_rnn_layers > 1:
        raise NotImplementedError('multiple RNN layers not supported yet')
    
    lm_state_dict = lm_state_dict.copy()
    lm_state_dict.pop('embedding.embedding.weight')
    lm_state_dict.pop('out.weight')
    lm_state_dict.pop('out.bias')
    lm_state_dict['encoder.rnn.weight_ih_l0'] = lm_state_dict.pop('rnn.weight_ih_l0')
    lm_state_dict['encoder.rnn.weight_hh_l0'] = lm_state_dict.pop('rnn.weight_hh_l0')
    lm_state_dict['encoder.rnn.bias_ih_l0'] = lm_state_dict.pop('rnn.bias_ih_l0')
    lm_state_dict['encoder.rnn.bias_hh_l0'] = lm_state_dict.pop('rnn.bias_hh_l0')
    model_dict = G.state_dict()
    # filter out unecessary keys
    lm_state_dict = {k: v for k, v in lm_state_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(lm_state_dict)
    # load the new state_dict
    G.load_state_dict(model_dict)

def load_decoder_from_lm(G, lm_state_dict):
    if opts.num_rnn_layers > 1:
        raise NotImplementedError('multiple RNN layers not supported yet')
    
    lm_state_dict = lm_state_dict.copy()
    lm_state_dict.pop('embedding.embedding.weight')
    lm_state_dict.pop('out.weight')
    lm_state_dict.pop('out.bias')
    lm_state_dict['decoder.rnn.weight_ih_l0'] = lm_state_dict.pop('rnn.weight_ih_l0')
    lm_state_dict['decoder.rnn.weight_hh_l0'] = lm_state_dict.pop('rnn.weight_hh_l0')
    lm_state_dict['decoder.rnn.bias_ih_l0'] = lm_state_dict.pop('rnn.bias_ih_l0')
    lm_state_dict['decoder.rnn.bias_hh_l0'] = lm_state_dict.pop('rnn.bias_hh_l0')
    model_dict = G.state_dict()
    # filter out unecessary keys
    lm_state_dict = {k: v for k, v in lm_state_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(lm_state_dict)
    # load the new state_dict
    G.load_state_dict(model_dict)

def load_discrimator_from_lm(D, lm_state_dict):
    lm_state_dict = lm_state_dict.copy()
    lm_state_dict.pop('embedding.embedding.weight')
    lm_state_dict.pop('out.weight')
    lm_state_dict.pop('out.bias')
    model_dict = D.state_dict()
    # filter out unecessary keys
    lm_state_dict = {k: v for k, v in lm_state_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(lm_state_dict)
    # load the new state_dict
    D.load_state_dict(model_dict)

def estimate_returns(opts, prev_tokens, num_steps, input, hidden, encoder_outputs, G, D):
    # make sure that prev_tokens do not contain <sos> token and it is a variable
    """Runs Monte Carlo search on the decoder to get returns for the intermediate steps
    Args:
        opts: needed for num_searches and discount_factor 
        prev_tokens: partial sentence (without <sos> token), shape: batch_size x (max_length - num_steps)
        num_steps: (int) number of steps for decoder rollout
        input: current words, shape: batch_size x 1
        hidden: hidden layer for the decoder, shape: num_rnn_layers x batch_size x hidden_size
        encoder_outputs: encoder intermediate hidden states, shape: max_length x batch_size x hidden_size
    Returns:
        returns: torch Tensor containing the return for the batch, shape: batch_size x 1
    """
    batch_size = input.shape[0]
    returns = torch.zeros(batch_size, 1)
    if use_cuda:
        returns = returns.cuda()

    if num_steps == 0:
        # If number of steps is zero, do not generate. Just compute the returns for the full sentence.
        Dout = D(prev_tokens)
        return Dout.data.view(batch_size, 1)

    for _step in range(opts.num_searches):
        rollouts, _ = G.decoder_rollout(num_steps, input, hidden, encoder_outputs, opts.alpha)
        sents = torch.cat((prev_tokens, rollouts), dim=1)
        Dout = D(sents)     # returns scalar value
        # b x 1
        # Discount the return
        returns = returns + float(np.power(opts.discount_factor, num_steps)) * Dout.data.view(batch_size, 1)
    returns = returns / opts.num_searches
    return returns

def batch_train_gan(sents_src, sents_tgt, masks_tgt, vocab_size_tgt, G, D, criterion, start_tok_src, start_tok_tgt, direc, opts):
    """
        sents_src: Input sentences from source style to be conditioned on
        sents_tgt: Positive samples of sentences from target style
        masks_tgt: masks for the target sentences
   vocab_size_tgt: vocab size for the target
                G: Generator
                D: Discriminator
    start_tok_src: start token for source style
    start_tok_tgt: start token for target style
            direc: direction required for baseline
    """
    batch_size = sents_src.shape[0]
    max_length = sents_src.shape[1]
    # source: input must not contain a start token
    input_sents_src = torch.LongTensor(sents_src)
    target_sents = torch.LongTensor(sents_tgt)
    masks = torch.Tensor(masks_tgt)
    if use_cuda:
        input_sents_src = input_sents_src.cuda()
        target_sents = target_sents.cuda()
        masks = masks.cuda()
    input_sents_src = Variable(input_sents_src)
    input_sents_tgt = Variable(target_sents)
    # Encode the input source sentence
    input_sents_src_encoded, hidden = G.encode(input_sents_src)
    # Generate the predicted target
    decoder_input = Variable(torch.LongTensor(np.ones((batch_size, 1))*start_tok_tgt))
    prev_tokens = None
    if use_cuda:
        decoder_input = decoder_input.cuda()
    q_values_accumulated = None
    log_probs_accumulated = None
    log_probs_all = None

    ########### REMOVE THIS ##############
    ## TODO: What is "REMOVE THIS" for? - Santhosh
    tot_returns = 0.0
     
    for l in range(opts.max_length):
        tokens, log_probs_, hidden = G.decoder_step(decoder_input, hidden, input_sents_src_encoded, opts.alpha)
        # Use advanced indexing to access log_probs of actions
        log_probs = log_probs_[range(batch_size), tokens[:, 0]].view(batch_size, 1)

        if prev_tokens is None:
            prev_tokens = tokens
        else:
            prev_tokens = torch.cat((prev_tokens, tokens), dim=1)
        returns = estimate_returns(opts, prev_tokens, opts.max_length-l-1, tokens, \
                                   hidden, input_sents_src_encoded, G, D)
        if q_values_accumulated is None:
            q_values_accumulated = returns
            log_probs_accumulated = log_probs
            log_probs_all = log_probs_
        else:
            q_values_accumulated = torch.cat((q_values_accumulated, returns), dim=1)
            log_probs_accumulated = torch.cat((log_probs_accumulated, log_probs), dim=1)
            log_probs_all = torch.cat((log_probs_all, log_probs_), dim=1)
        
        if direc == 0:
            opts.baseline12 = opts.baseline12 * 0.99 + torch.mean(returns) * 0.01
        elif direc == 1:
            opts.baseline21 = opts.baseline21 * 0.99 + torch.mean(returns) * 0.01

        tot_returns += torch.mean(returns)
    # To be returned 
    predicted_tokens = prev_tokens

    # Compute RL loss for generator
    if direc == 0:
        loss_term_generator = rl_loss(q_values_accumulated, log_probs_accumulated, opts.baseline12)
    elif direc == 1:
        loss_term_generator = rl_loss(q_values_accumulated, log_probs_accumulated, opts.baseline21)
    
    # Discriminator update
    # positive target: discriminator input must NOT contain a start token
    # Forward pass
    Dout_real = D(input_sents_tgt)
    Dout_fake = D(predicted_tokens)
    # Compute discriminator loss
    loss_term_discriminator = torch.mean(Dout_fake) - torch.mean(Dout_real)

    # Compute the supervised generator loss
    # NOTE: This must be a teacher forcing loss. Currently, this is a sampling + Cross Entropy loss which
    # does not necessarily make sense. 
    target_sents = Variable(target_sents)
    masks = Variable(masks)
    nll_loss_unmasked_batch = criterion(log_probs_all.view(-1, vocab_size_tgt), target_sents.view(-1))
    nll_loss_batch = (nll_loss_unmasked_batch * masks.view(-1)).sum()
    loss_term_supervised_G = nll_loss_batch / batch_size

    return loss_term_generator, loss_term_discriminator, loss_term_supervised_G, predicted_tokens, log_probs_accumulated, tot_returns / opts.max_length

def batch_cyclic_loss(sents_src, sents_tgt_hat, sents_tgt, masks_src, masks_tgt, G_tgt2src, start_tok_src, criterion, cos_criterion, opts):
    """
    Computes the cyclic reconstruction loss
    Inputs:
        sents_src: Input sentences from source style to be conditioned on (B x max_length np array)
        sents_tgt_hat: Generated samples of sentences from target style (B x max_length Variable)
        sents_tgt: Ground truth samples of sentences from target style (B x max_length np array)
        masks_src: Masks for input source style sentences
        G_tgt2src: Generator that converts to source style conditioned on target style
        start_tok_src: Start token in source style
    Returns:
        reconstruction_loss_batch - reconstruction loss for the current source sentences in batch
    """
    batch_size = sents_src.shape[0]
    max_length = sents_src.shape[1]
    # Encode the generated target sentence
    encoder_outputs, hidden = G_tgt2src.encode(sents_tgt_hat)
    # Encode the groung truth target sentence
    if use_cuda:
        sents_tgt = Variable(torch.LongTensor(sents_tgt)).cuda()
    else:
        sents_tgt = Variable(torch.LongTensor(sents_tgt))
    encoder_outputs_ground_truth,_ = G_tgt2src.encode(sents_tgt)
    # Decode sentence in the source style
    decoder_input = np.zeros((batch_size, max_length)).astype(sents_src.dtype)
    decoder_input[:, 0] = start_tok_src
    decoder_input[:, 1:] = np.copy(sents_src)[:, :-1]
    decoder_input = torch.LongTensor(decoder_input)
    if use_cuda:
        decoder_input = decoder_input.cuda()
    decoder_input = Variable(decoder_input)
     
    predicted_tokens, log_probs = G_tgt2src.decode(opts.max_length, decoder_input, hidden, encoder_outputs, opts.alpha)
    # Compute the reconstruction loss
    vocab_size_src = log_probs.size(2)
    sents_src_labels = torch.LongTensor(sents_src)
    sents_masks = torch.FloatTensor(masks_src)
    if use_cuda:
        sents_src_labels = sents_src_labels.cuda()
        sents_masks = sents_masks.cuda()
    sents_src_labels = Variable(sents_src_labels)
    sents_masks = Variable(sents_masks)

    loss_unmasked = criterion(log_probs.view(-1, vocab_size_src), sents_src_labels.view(-1))
    reconstruction_loss_batch = (loss_unmasked * sents_masks.view(-1)).sum() / batch_size

    # Advanced indexing to find the encoding of the <EOS> tag
    sentence_indexes = np.sum(masks_tgt, axis=1).astype(np.int32) - 1
    encoder_outputs = torch.transpose(encoder_outputs.data.cpu(), 0, 1)[range(batch_size), sentence_indexes]
    encoder_outputs_ground_truth = torch.transpose(encoder_outputs_ground_truth.data.cpu(), 0, 1)[range(batch_size), sentence_indexes]
    # Calculate supervised encoder consine similarity    
    encoder_sup_loss = cos_criterion(Variable(encoder_outputs), Variable(encoder_outputs_ground_truth))

    return reconstruction_loss_batch, predicted_tokens, encoder_sup_loss

def evaluate_generators_src2tgt(G12, G21, criterion, loader, opts, split='val'):
    depleted = False
    total_loss_term_rec11 = 0.0
    total_loss_term_rec22 = 0.0
    num_sentences = 0.0
    sents_s1_str = []
    sents_s2_hat_str = []
    sents_s11_hat_str = []
    sents_s2_str = []
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
        max_length = sents_s1.shape[1]
        num_sentences += batch_size
        # source input must not contain a start token
        input_sents_s1 = torch.LongTensor(sents_s1)
        if use_cuda:
            input_sents_s1 = input_sents_s1.cuda()
        input_sents_s1 = Variable(input_sents_s1)
        # encode the input source sentence
        input_sents_s1_encoded, hidden = G12.encode(input_sents_s1)
        # generate the predicted target
        decoder_input = Variable(torch.LongTensor(np.ones((batch_size, 1))*opts.start_idx_s2))
        if use_cuda:
            decoder_input = decoder_input.cuda()
        rollouts_s2_hat, _ = G12.decoder_rollout(max_length, decoder_input, hidden, input_sents_s1_encoded, opts.alpha)
        sents_s2_hat = rollouts_s2_hat.data.cpu().numpy().astype(int)
        # encode the generated target sentence
        gen_sents_s2_hat_encoded, hidden = G21.encode(rollouts_s2_hat)
        # generate the predicted source
        decoder_input = Variable(torch.LongTensor(np.ones((batch_size, 1))*opts.start_idx_s1))
        if use_cuda:
            decoder_input = decoder_input.cuda()
        rollouts_s11_hat, _ = G21.decoder_rollout(max_length, decoder_input, hidden, gen_sents_s2_hat_encoded, opts.alpha)
        sents_s11_hat = rollouts_s11_hat.data.cpu().numpy().astype(int)
        # use teacher forcing to generate the cross entropy score
        decoder_input = np.zeros((batch_size, max_length)).astype(sents_s1.dtype)
        decoder_input[:, 0] = opts.start_idx_s1
        decoder_input[:, 1:] = np.copy(sents_s1)[:, :-1]
        decoder_input = Variable(torch.LongTensor(decoder_input))
        if use_cuda:
            decoder_input = decoder_input.cuda()
        
        _, log_probs = G21.decode(opts.max_length, decoder_input, hidden, gen_sents_s2_hat_encoded, opts.alpha)
        # Compute the reconstruction loss
        vocab_size_s1 = log_probs.size(2)
        sents_masks_s1 = torch.FloatTensor(masks_s1)
        if use_cuda:
            sents_masks_s1 = sents_masks_s1.cuda()
        sents_masks_s1 = Variable(sents_masks_s1)

        loss_unmasked_rec11 = criterion(log_probs.view(-1, vocab_size_s1), input_sents_s1.view(-1))
        total_loss_term_rec11 = total_loss_term_rec11 + (loss_unmasked_rec11 * sents_masks_s1.view(-1)).sum().data.cpu()[0]

        # computing the string sentences
        sents_s1_str.extend(get_sentence_from_np(sents_s1, loader, src=True))
        sents_s2_hat_str.extend(get_sentence_from_np(sents_s2_hat, loader, src=False))
        sents_s11_hat_str.extend(get_sentence_from_np(sents_s11_hat, loader, src=True))
        sents_s2_str.extend(get_sentence_from_np(sents_s2, loader, src=False))

    n_sents = min(opts.num_sample_sents, len(sents_s1))

    return ((total_loss_term_rec11 / num_sentences), \
    sents_s1_str[:n_sents], sents_s2_hat_str[:n_sents], sents_s11_hat_str[:n_sents], sents_s2_str[:n_sents])

def train_cyclegan_run_iterations(opts, embedding_s1, embedding_s2, G12, G21, D1, D2, loader):
    # freeze embeddings
    if opts.freeze_embeddings:
        for param in embedding_s1.parameters():
            param.requires_grad = False
        for param in embedding_s2.parameters():
            param.requires_grad = False

    # keep track of number of generator iterations
    g_update_step_diff = opts.g_update_step_diff
    gen_train_mode = 'fast'

    # set mode to train
    embedding_s1.train()
    embedding_s2.train()
    G12.train()
    G21.train()
    D1.train()
    D2.train()
    
    # create average return baseline
    opts.baseline12 = 0.0
    opts.baseline21 = 0.0
    opts.baseline_r1 = 0.0
    opts.baseline_r2 = 0.0
    opts.baseline_c11 = 0.0
    opts.baseline_c22 = 0.0

    # loss criterion
    criterion = nn.NLLLoss(reduce=False)
    cos_criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    # Reconstruction RL Loss
    discount_factor_tensor = torch.FloatTensor(np.array([pow(opts.discount_factor, i) for i in reversed(range(opts.max_length))])).view(1, -1)

    # put parameter to cuda if cuda is available
    if use_cuda:
        embedding_s1.cuda()
        embedding_s2.cuda()
        G12.cuda()
        G21.cuda()
        D1.cuda()
        D2.cuda()
        criterion.cuda()
        discount_factor_tensor = discount_factor_tensor.cuda()

    # find unique params for the whole network
    net_params_G = ( set(embedding_s1.parameters()) | set(embedding_s2.parameters()) \
        | set(G12.parameters()) | set(G21.parameters()) )
    net_params_D = ( set(embedding_s1.parameters()) | set(embedding_s2.parameters()) \
        | set(D1.parameters()) | set(D2.parameters()) )
    # remove params that don't require grad updates
    net_params_G = [p for p in net_params_G if p.requires_grad == True]
    net_params_D = [p for p in net_params_D if p.requires_grad == True]

    # define optimizer
    optimizer_G = optim.RMSprop(net_params_G, lr=opts.lr, weight_decay=opts.weight_decay)
    optimizer_D = optim.RMSprop(net_params_D, lr=opts.lr * opts.lr_ratio_D_by_G, weight_decay=opts.weight_decay)

    # for logging
    n_iter = 0
    writer = SummaryWriter(log_dir=opts.log_dir)
    loss_log = {'G12': 0.0, 'G21': 0.0, 'G11': 0.0, 'G22': 0.0, 'SG12': 0.0, 'SG21': 0.0, 'G12R' : 0.0, 'G21R' : 0.0, 'G12C' : 0.0, 'G21C' : 0.0, 'D1' : 0.0, 'D2' : 0.0}
    num_batches = 0.0
    time_start = time.time()
    # for choosing the best model
    best_val_rec_loss = float('inf')
    # for keeping track of generator updates
    num_disc_updates_since_last_gen_update = 0
    for _epoch in range(opts.epochs):
        depleted = False

        while not depleted:
            n_iter += 1 # for discriminator bias
            # Set the waiting period for generator training
            if n_iter % opts.disc_recalibrate == 0 and gen_train_mode == 'fast':
                # TODO: Adding a condition that checks history of D losses
                g_update_step_diff = opts.g_update_step_diff_recalib
                gen_train_mode = 'slow'
                print('Recalibrating the discriminator')
            # Sents: batch_size x max_length [w1, w2, ..., <eos>, <pad>, <pad>, ..]
            # Masks: batch_size x max_length [ 1,  1, ...,     1,     0,     0, ...]
            sents_s1, masks_s1, sents_s2, masks_s2, depleted = loader.next_batch_train()
            batch_size = sents_s1.shape[0]
            # Compute the different losses
            loss_term_g12, loss_term_d2, loss_term_sg12, sents_s2_hat, log_probs_12, avg_returns_12 = batch_train_gan(sents_s1, sents_s2, masks_s2, opts.vocab_size_s2, G12, D2, criterion,\
                                                                                opts.start_idx_s1, opts.start_idx_s2, 0, opts)
            loss_term_g21, loss_term_d1, loss_term_sg21, sents_s1_hat, log_probs_21, avg_returns_21 = batch_train_gan(sents_s2, sents_s1, masks_s1, opts.vocab_size_s1, G21, D1, criterion,\
                                                                                opts.start_idx_s2, opts.start_idx_s1, 1, opts)
            loss_term_rec11, sents_s11_hat, loss_cosine_enc11 = batch_cyclic_loss(sents_s1, sents_s2_hat, sents_s2, masks_s1, masks_s2, G21, opts.start_idx_s1, criterion, cos_criterion, opts)
            loss_term_rec22, sents_s22_hat, loss_cosine_enc22 = batch_cyclic_loss(sents_s2, sents_s1_hat, sents_s1, masks_s2, masks_s1, G12, opts.start_idx_s2, criterion, cos_criterion, opts)

            # RL reward for reconstruction loss
            rec_reward_11 = -1.0 * loss_term_rec11.data.cpu()[0]
            rec_reward_22 = -1.0 * loss_term_rec22.data.cpu()[0]
            # baseline rewards for reconstruction RL reward
            opts.baseline_r1 = opts.baseline_r1 * 0.99 + rec_reward_11 * 0.01
            opts.baseline_r2 = opts.baseline_r2 * 0.99 + rec_reward_22 * 0.01
            # compute the reconstruction RL loss
            loss_term_rec12 = rl_loss(discount_factor_tensor.expand(batch_size, opts.max_length) * rec_reward_11, log_probs_12, opts.baseline_r1)
            loss_term_rec21 = rl_loss(discount_factor_tensor.expand(batch_size, opts.max_length) * rec_reward_22, log_probs_21, opts.baseline_r2)

            # RL reward for encoder cosine loss
            cos_reward_11 = loss_cosine_enc11.data.cpu()[0]
            cos_reward_22 = loss_cosine_enc22.data.cpu()[0]
            # baseline rewards for cosine RL reward
            opts.baseline_c11 = opts.baseline_c11 * 0.99 + cos_reward_11 * 0.01
            opts.baseline_c22 = opts.baseline_c22 * 0.99 + cos_reward_22 * 0.01
            # compute the cosine RL loss
            loss_term_cos12 = rl_loss(discount_factor_tensor.expand(batch_size, opts.max_length) * cos_reward_11, log_probs_12, opts.baseline_c11)
            loss_term_cos21 = rl_loss(discount_factor_tensor.expand(batch_size, opts.max_length) * cos_reward_22, log_probs_21, opts.baseline_c22)

            # Optimize the generators
            if _epoch >= opts.d_pretrain_num_epochs and num_disc_updates_since_last_gen_update >= g_update_step_diff:
                optimizer_G.zero_grad()
                overall_loss_G = opts.lamda_rl * (loss_term_g12 + loss_term_g21) + opts.lamda_rec_ij * (loss_term_rec12 + loss_term_rec21) + opts.lamda_cos_ij * (loss_term_cos12 + loss_term_cos21)
                overall_loss_G.backward()
                nn.utils.clip_grad_norm(net_params_G, opts.max_norm)
                optimizer_G.step()
                if gen_train_mode == 'slow':
                    gen_train_mode = 'fast'
                    g_update_step_diff = opts.g_update_step_diff
                num_disc_updates_since_last_gen_update = 0
            # Optimize the discriminators
            if n_iter % opts.d_update_step_diff == 0:
                optimizer_D.zero_grad()
                overall_loss_D = loss_term_d1 + loss_term_d2
                overall_loss_D.backward()
                nn.utils.clip_grad_norm(net_params_D, opts.max_norm)
                optimizer_D.step()
                # clipping weights
                for p in net_params_D:
                    p.data.clamp_(opts.clamp_lower, opts.clamp_upper)
                num_disc_updates_since_last_gen_update += 1 
            # Log the losses
            loss_log['G12'] += loss_term_g12.data.cpu()[0]
            loss_log['G21'] += loss_term_g21.data.cpu()[0]
            loss_log['G11'] += loss_term_rec11.data.cpu()[0]
            loss_log['G22'] += loss_term_rec22.data.cpu()[0]
            loss_log['G12R'] += loss_term_rec12.data.cpu()[0]
            loss_log['G21R'] += loss_term_rec21.data.cpu()[0]
            loss_log['G12C'] += loss_term_cos12.data.cpu()[0]
            loss_log['G21C'] += loss_term_cos21.data.cpu()[0]
            loss_log['SG12'] += loss_term_sg12.data.cpu()[0]
            loss_log['SG21'] += loss_term_sg21.data.cpu()[0]
            loss_log['D1']  += loss_term_d1.data.cpu()[0]
            loss_log['D2']  += loss_term_d2.data.cpu()[0]
            num_batches += 1

            # update skip connection weight
            opts.alpha = opts.alpha * opts.skip_weight_decay
            
            # display message on output
            if num_batches != 0 and n_iter % opts.log_iter == 0:
                time_end = time.time()
                time_taken = time_end - time_start
                avg_loss_G12 = loss_log['G12'] / num_batches
                avg_loss_G21 = loss_log['G21'] / num_batches
                avg_loss_G11 = loss_log['G11'] / num_batches
                avg_loss_G22 = loss_log['G22'] / num_batches
                avg_loss_G12R = loss_log['G12R'] / num_batches
                avg_loss_G21R = loss_log['G21R'] / num_batches
                avg_loss_G12C = loss_log['G12C'] / num_batches
                avg_loss_G21C = loss_log['G21C'] / num_batches
                avg_loss_SG12 = loss_log['SG12'] / num_batches
                avg_loss_SG21 = loss_log['SG21'] / num_batches
                avg_loss_D1 = loss_log['D1'] / num_batches
                avg_loss_D2 = loss_log['D2'] / num_batches

                # printing statistics on console
                print ("epoch: %d updates: %d time: %.1fs G12: %.4f G21: %.4f SG12: %.2f SG21: %.2f G11: %.2f G22: %.2f G12R: %.1f G21R: %.1f G12C: %.3f G21C: %.3f D1: %.4f D2: %.4f G_upd_last: %d" % (_epoch, \
                    n_iter, time_taken, avg_loss_G12, avg_loss_G21, avg_loss_SG12, avg_loss_SG21, avg_loss_G11, avg_loss_G22, \
                    avg_loss_G12R, avg_loss_G21R, avg_loss_G12C, avg_loss_G21C, avg_loss_D1, avg_loss_D2, num_disc_updates_since_last_gen_update))
                # Writing values to SummaryWriter
                writer.add_scalars('train/G_rec_losses', {'G11': avg_loss_G11, 'G22': avg_loss_G22}, n_iter)
                writer.add_scalars('train/G_rec_rl_losses', {'G12R': avg_loss_G12R, 'G21R': avg_loss_G21R}, n_iter)
                writer.add_scalars('train/G_cos_rl_losses', {'G12C': avg_loss_G12C, 'G21C': avg_loss_G21C}, n_iter)
                writer.add_scalars('train/G_rl_losses', {'G12': avg_loss_G12, 'G21': avg_loss_G21}, n_iter)
                writer.add_scalars('train/G_sup_losses', {'SG12': avg_loss_SG12, 'SG21': avg_loss_SG21}, n_iter)
                writer.add_scalars('train/D_losses', {'D1': avg_loss_D1, 'D2': avg_loss_D2}, n_iter)
                writer.add_scalars('train/baseline_D', {'B12': opts.baseline12, 'B21': opts.baseline21}, n_iter)
                writer.add_scalars('train/baseline_rec', {'BR1': opts.baseline_r1, 'BR2': opts.baseline_r2}, n_iter)
                writer.add_scalars('train/baseline_cos', {'BC1': opts.baseline_c11, 'BC2': opts.baseline_c22}, n_iter)
                writer.add_scalars('train/returns', {'R12': avg_returns_12, 'R21': avg_returns_21}, n_iter)
                # reset values back
                loss_log = {'G12': 0.0, 'G21': 0.0, 'G11': 0.0, 'G22': 0.0, 'SG12': 0.0, 'SG21': 0.0, 'G12R' : 0.0, 'G21R' : 0.0, 'G12C' : 0.0, 'G21C' : 0.0, 'D1' : 0.0, 'D2' : 0.0}
                num_batches = 0.0
                time_start = time.time()
            # print sample sentences on output  
            if n_iter % opts.sent_sample_iter == 0:
                n_sents = min(opts.num_sample_sents, len(sents_s1))
                sents_s11_hat = sents_s11_hat.data.cpu().numpy().astype(int)
                sents_s2_hat = sents_s2_hat.data.cpu().numpy().astype(int)
                
                sents_s1_str = get_sentence_from_np(sents_s1, loader, src=True)[:n_sents]
                sents_s2_hat_str = get_sentence_from_np(sents_s2_hat, loader, src=False)[:n_sents]
                sents_s11_hat_str = get_sentence_from_np(sents_s11_hat, loader, src=True)[:n_sents]
                sents_s2_str = get_sentence_from_np(sents_s2, loader, src=False)[:n_sents]
                print ('')
                print ("******************* Sample Sentences *******************")
                for _n_sent in range(n_sents):
                    print ('Sentence %d' % (_n_sent + 1))
                    print ('S1     : ' + sents_s1_str[_n_sent])
                    print ('S2_hat : ' + sents_s2_hat_str[_n_sent])
                    print ('S1_hat : ' + sents_s11_hat_str[_n_sent])
                    print ('S2     : ' + sents_s2_str[_n_sent])
                    print ('')
                print ("********************************************************")
                print ('')

        # Evaluating model on the validation set
        val_rec11_loss, val_sents_s1_str, val_sents_s2_hat_str, val_sents_s11_hat_str, val_sents_s2_str = evaluate_generators_src2tgt(G12, \
                                                                                                            G21, criterion, loader, opts, split='val')
        print ("#################### Validation Step ####################")
        print ("#########################################################")
        print ("epoch: %d, updates: %d, validation loss G11: %.5f." % (_epoch, n_iter, val_rec11_loss))
        # Writing values to SummaryWriter
        writer.add_scalar('val/G11_loss', val_rec11_loss, n_iter)
        print ('')
        print ("************** Validation Sample Sentences **************")
        for _n_sent in range(len(val_sents_s1_str)):
            print ('Sentence %d' % (_n_sent + 1))
            print ('S1     : ' + val_sents_s1_str[_n_sent])
            print ('S2_hat : ' + val_sents_s2_hat_str[_n_sent])
            print ('S1_hat : ' + val_sents_s11_hat_str[_n_sent])
            print ('S2     : ' + val_sents_s2_str[_n_sent])
            print ('')
        print ("*********************************************************")
        print ('')

        # saving the model to disk
        if val_rec11_loss <= best_val_rec_loss:
            best_val_rec_loss = val_rec11_loss
            save_state = {
                'epoch': _epoch,
                'G12_state_dict': G12.state_dict(),
                'G21_state_dict': G21.state_dict(),
                'D1_state_dict': D1.state_dict(),
                'D2_state_dict': D2.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'opts': opts, 
                'rec_loss': val_rec11_loss
            }
            model_name = opts.model_name + '_best.net'
            torch.save(save_state, os.path.join(opts.save_path, model_name))

        save_state = {
            'epoch': _epoch,
            'G12_state_dict': G12.state_dict(),
            'G21_state_dict': G21.state_dict(),
            'D1_state_dict': D1.state_dict(),
            'D2_state_dict': D2.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'opts': opts, 
            'rec_loss': val_rec11_loss
        }
        model_name = opts.model_name + '_latest.net'
        torch.save(save_state, os.path.join(opts.save_path, model_name)) 

def train_cyclegan(opts):
    print ("################### Cyclegan Training ###################")
    print ("#########################################################")
    print ('')
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
    opts.alpha = 1.0
    
    # Loading pretrained language models state_dict
    lm1_state_dict = torch.load(opts.pretrained_lm1_model_path)['state_dict']
    lm2_state_dict = torch.load(opts.pretrained_lm2_model_path)['state_dict']

    # Initialising embeddings with pretrained language models
    embedding_s1 = EmbeddingLayer(opts.vocab_size_s1, opts.embedding_size, opts.padding_idx_s1)
    embedding_s2 = EmbeddingLayer(opts.vocab_size_s2, opts.embedding_size, opts.padding_idx_s2)
    load_embedding_from_lm(embedding_s1, lm1_state_dict)
    load_embedding_from_lm(embedding_s2, lm2_state_dict)

    # Initialising generators with pretrained language models
    G12 = Generator(embedding_s1, embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length)
    G21 = Generator(embedding_s2, embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length)

    # TODO: need to figure out how to initialise the decoder
    load_encoder_from_lm(G12, lm1_state_dict)
    load_decoder_from_lm(G12, lm2_state_dict)
    load_encoder_from_lm(G21, lm2_state_dict)
    load_decoder_from_lm(G21, lm1_state_dict)

    # Initialising discriminators with pretrained language models
    D1 = Discriminator(embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p)
    D2 = Discriminator(embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p)

    load_discrimator_from_lm(D1, lm1_state_dict)
    load_discrimator_from_lm(D2, lm2_state_dict)
    
    train_cyclegan_run_iterations(opts, embedding_s1, embedding_s2, G12, G21, D1, D2, loader)

def evaluate_seqseq_src2tgt(G12, G21, criterion, loader, opts, split='val'):
    depleted = False
    total_loss_G12 = 0.0
    total_loss_G21 = 0.0
    num_sentences = 0.0
    sents_s1_str = []
    sents_s2_hat_str = []
    sents_s1_hat_str = []
    sents_s2_str = []
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
        max_length = sents_s1.shape[1]
        num_sentences += batch_size
        # source input must not contain a start token
        input_sents_s1 = torch.LongTensor(sents_s1)
        input_sents_s2 = torch.LongTensor(sents_s2)
        if use_cuda:
            input_sents_s1 = input_sents_s1.cuda()
            input_sents_s2 = input_sents_s2.cuda()
        input_sents_s1 = Variable(input_sents_s1)
        input_sents_s2 = Variable(input_sents_s2)
        
        # encode the input source sentence
        input_sents_s1_encoded, hidden = G12.encode(input_sents_s1)
        # generate the predicted target
        decoder_input_12 = np.zeros((batch_size, max_length)).astype(sents_s2.dtype)
        decoder_input_12[:, 0] = opts.start_idx_s2
        decoder_input_12[:, 1:] = np.copy(sents_s2)[:, :-1]
        decoder_input_12 = Variable(torch.LongTensor(decoder_input_12))
        if use_cuda:
            decoder_input_12 = decoder_input_12.cuda()
        rollouts_s2_hat, log_probs_12 = G12.decode(max_length, decoder_input_12, hidden, input_sents_s1_encoded, opts.alpha)
        sents_s2_hat = rollouts_s2_hat.data.cpu().numpy().astype(int)

        # encode the input source sentence
        input_sents_s2_encoded, hidden = G21.encode(input_sents_s2)
        # generate the predicted target
        decoder_input_21 = np.zeros((batch_size, max_length)).astype(sents_s1.dtype)
        decoder_input_21[:, 0] = opts.start_idx_s1
        decoder_input_21[:, 1:] = np.copy(sents_s1)[:, :-1]
        decoder_input_21 = Variable(torch.LongTensor(decoder_input_21))
        if use_cuda:
            decoder_input_21 = decoder_input_21.cuda()
        rollouts_s1_hat, log_probs_21 = G21.decode(max_length, decoder_input_21, hidden, input_sents_s2_encoded, opts.alpha)
        sents_s1_hat = rollouts_s1_hat.data.cpu().numpy().astype(int)

        # Compute the cross entropy loss
        vocab_size_s2 = log_probs_12.size(2)
        sents_masks_s2 = torch.FloatTensor(masks_s2)
        if use_cuda:
            sents_masks_s2 = sents_masks_s2.cuda()
        sents_masks_s2 = Variable(sents_masks_s2)
        loss_unmasked_G12 = criterion(log_probs_12.view(-1, vocab_size_s2), input_sents_s2.view(-1))
        total_loss_G12 = total_loss_G12 + (loss_unmasked_G12 * sents_masks_s2.view(-1)).sum().data.cpu()[0]
        
        # Compute the cross entropy loss
        vocab_size_s1 = log_probs_21.size(2)
        sents_masks_s1 = torch.FloatTensor(masks_s1)
        if use_cuda:
            sents_masks_s1 = sents_masks_s1.cuda()
        sents_masks_s1 = Variable(sents_masks_s1)
        loss_unmasked_G21 = criterion(log_probs_21.view(-1, vocab_size_s1), input_sents_s1.view(-1))
        total_loss_G21 = total_loss_G21 + (loss_unmasked_G21 * sents_masks_s1.view(-1)).sum().data.cpu()[0]

        # computing the string sentences
        sents_s1_str.extend(get_sentence_from_np(sents_s1, loader, src=True))
        sents_s2_hat_str.extend(get_sentence_from_np(sents_s2_hat, loader, src=False))
        sents_s1_hat_str.extend(get_sentence_from_np(sents_s1_hat, loader, src=True))
        sents_s2_str.extend(get_sentence_from_np(sents_s2, loader, src=False))

    n_sents = min(opts.num_sample_sents, len(sents_s1))

    return ((total_loss_G12 / num_sentences), (total_loss_G21 / num_sentences), \
    sents_s1_str[:n_sents], sents_s2_hat_str[:n_sents], sents_s1_hat_str[:n_sents], sents_s2_str[:n_sents])

def batch_train_seq2seq(sents_src, sents_tgt, masks_tgt, vocab_size_tgt, G, criterion, start_tok_tgt, alpha, schedule_eps=None):
    """
        sents_src: Input sentences from source style to be conditioned on
        sents_tgt: Paired samples of sentences from target style
        masks_tgt: masks for the target sentences
   vocab_size_tgt: vocab size for the target
                G: Generator
        criterion: NLLLoss criterion
    start_tok_tgt: start token for target style
            alpha: alpha needed for decoder residual connection
     schedule_eps: epsilon for scheduled sampling (check decode() for details)
    """
    batch_size = sents_src.shape[0]
    max_length = sents_src.shape[1]
    # source: input must not contain a start token
    input_sents_src = torch.LongTensor(sents_src)
    target_sents = torch.LongTensor(sents_tgt)
    masks = torch.Tensor(masks_tgt)
    if use_cuda:
        input_sents_src = input_sents_src.cuda()
        target_sents = target_sents.cuda()
        masks = masks.cuda()
    input_sents_src = Variable(input_sents_src)
    # target: decoder input must contain a start token
    output_sents_tgt = np.zeros((batch_size, max_length)).astype(sents_tgt.dtype)
    output_sents_tgt[:, 0] = start_tok_tgt
    output_sents_tgt[:, 1:] = np.copy(sents_tgt)[:, :-1]
    output_sents_tgt = torch.LongTensor(output_sents_tgt)
    if use_cuda:
        output_sents_tgt = output_sents_tgt.cuda()
    output_sents_tgt = Variable(output_sents_tgt)

    # Forward pass through the generator
    input_sents_src_encoded, hidden = G.encode(input_sents_src)
    rollouts, log_probs = G.decode(max_length, output_sents_tgt, hidden, input_sents_src_encoded, opts.alpha, schedule_eps)

    target_sents = Variable(target_sents)
    masks = Variable(masks)
    nll_loss_unmasked_batch = criterion(log_probs.view(-1, vocab_size_tgt), target_sents.view(-1))
    nll_loss_batch = (nll_loss_unmasked_batch * masks.view(-1)).sum()
    nll_loss = nll_loss_batch / batch_size

    return rollouts, nll_loss

def train_seq2seq(opts):
    print ("################# seq2seq MLE training #################")
    print ("########################################################")
    print ('')
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
    opts.alpha = 1.0
    
    # scheduled sampling parameters
    if opts.enable_scheduled_sampling:
        opts.scheduled_eps = 1.0
    else:
        opts.scheduled_eps = None

    # Loading pretrained language models state_dict
    lm1_state_dict = torch.load(opts.pretrained_lm1_model_path)['state_dict']
    lm2_state_dict = torch.load(opts.pretrained_lm2_model_path)['state_dict']
    # Initialising embeddings with pretrained language models
    embedding_s1 = EmbeddingLayer(opts.vocab_size_s1, opts.embedding_size, opts.padding_idx_s1)
    embedding_s2 = EmbeddingLayer(opts.vocab_size_s2, opts.embedding_size, opts.padding_idx_s2)
    load_embedding_from_lm(embedding_s1, lm1_state_dict)
    load_embedding_from_lm(embedding_s2, lm2_state_dict)
    # Initialising generators with pretrained language models
    G12 = Generator(embedding_s1, embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length, opts.use_attention)
    G21 = Generator(embedding_s2, embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length, opts.use_attention)
    load_encoder_from_lm(G12, lm1_state_dict)
    load_decoder_from_lm(G12, lm2_state_dict)
    load_encoder_from_lm(G21, lm2_state_dict)
    load_decoder_from_lm(G21, lm1_state_dict)

    # set mode to train
    embedding_s1.train()
    embedding_s2.train()
    G12.train()
    G21.train()

    # loss criterion
    criterion = nn.NLLLoss(reduce=False)

    # put parameter to cuda if cuda is available
    if use_cuda:
        embedding_s1.cuda()
        embedding_s2.cuda()
        G12.cuda()
        G21.cuda()
        criterion.cuda()

    optimizer_G12 = optim.Adam(G12.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    optimizer_G21 = optim.Adam(G21.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    # for logging
    n_iter = 0
    writer = SummaryWriter(log_dir=opts.log_dir)
    loss_log = {'G12': 0.0, 'G21': 0.0}
    num_batches = 0.0
    time_start = time.time()
    # for choosing the best model
    best_val_G12_loss = float('inf')
    best_val_G21_loss = float('inf')
    best_G12_state_dict = G12.state_dict()
    best_G21_state_dict = G21.state_dict()

    for _epoch in range(opts.epochs):
        depleted = False

        while not depleted:
            n_iter += 1
            sents_s1, masks_s1, sents_s2, masks_s2, depleted = loader.next_batch_train()
            batch_size = sents_s1.shape[0]

            sents_s2_hat, loss_G12 = batch_train_seq2seq(sents_s1, sents_s2, masks_s2, opts.vocab_size_s2, G12, criterion, opts.start_idx_s2, opts.alpha, opts.scheduled_eps)
            sents_s1_hat, loss_G21 = batch_train_seq2seq(sents_s2, sents_s1, masks_s1, opts.vocab_size_s1, G21, criterion, opts.start_idx_s1, opts.alpha, opts.scheduled_eps)

            # optimize the generator G12
            optimizer_G12.zero_grad()
            loss_G12.backward()
            nn.utils.clip_grad_norm(G12.parameters(), opts.max_norm)
            optimizer_G12.step()
            # optimize the generator G21
            optimizer_G21.zero_grad()
            loss_G21.backward()
            nn.utils.clip_grad_norm(G21.parameters(), opts.max_norm)
            optimizer_G21.step()

            # Log the losses
            loss_log['G12'] += loss_G12.data.cpu()[0]
            loss_log['G21'] += loss_G21.data.cpu()[0]
            num_batches += 1

            # update skip connection weight
            opts.alpha = opts.alpha * opts.skip_weight_decay

            # display message on output
            if num_batches != 0 and n_iter % opts.log_iter == 0:
                time_end = time.time()
                time_taken = time_end - time_start
                avg_loss_G12 = loss_log['G12'] / num_batches
                avg_loss_G21 = loss_log['G21'] / num_batches
                # printing statistics on console
                if opts.enable_scheduled_sampling:
                    print ("epoch: %d, updates: %d, time: %.2fs, G12: %.5f, G21: %.5f, sched_eps: %.3f" % (_epoch, \
                    n_iter, time_taken, avg_loss_G12, avg_loss_G21, opts.scheduled_eps))
                else:
                    print ("epoch: %d, updates: %d, time: %.2fs, G12: %.5f, G21: %.5f." % (_epoch, \
                    n_iter, time_taken, avg_loss_G12, avg_loss_G21))
                # Writing values to SummaryWriter
                writer.add_scalars('train', {'G12': avg_loss_G12, 'G21': avg_loss_G21}, n_iter)
                # reset values back
                loss_log = {'G12': 0.0, 'G21': 0.0}
                num_batches = 0.0
                time_start = time.time()
            # print sample sentences on output
            if n_iter % opts.sent_sample_iter == 0:
                n_sents = min(opts.num_sample_sents, len(sents_s1))
                sents_s1_hat = sents_s1_hat.data.cpu().numpy().astype(int)
                sents_s2_hat = sents_s2_hat.data.cpu().numpy().astype(int)
                
                sents_s1_str = get_sentence_from_np(sents_s1, loader, src=True)[:n_sents]
                sents_s1_hat_str = get_sentence_from_np(sents_s1_hat, loader, src=True)[:n_sents]
                sents_s2_str = get_sentence_from_np(sents_s2, loader, src=False)[:n_sents]
                sents_s2_hat_str = get_sentence_from_np(sents_s2_hat, loader, src=False)[:n_sents]
                print ('')
                print ("******************* Sample Sentences *******************")
                for _n_sent in range(n_sents):
                    print ('Sentence %d' % (_n_sent + 1))
                    print ('S1     : ' + sents_s1_str[_n_sent])
                    print ('S2_hat : ' + sents_s2_hat_str[_n_sent])
                    print ('S2     : ' + sents_s2_str[_n_sent])
                    print ('S1_hat : ' + sents_s1_hat_str[_n_sent])
                    print ('')
                print ("********************************************************")
                print ('')

        # Evaluating model on the validation set
        val_G12_loss, val_G21_loss, val_sents_s1_str, val_sents_s2_hat_str, val_sents_s1_hat_str, val_sents_s2_str = evaluate_seqseq_src2tgt(G12, \
                                                                                                            G21, criterion, loader, opts, split='val')
        print ("#################### Validation Step ####################")
        print ("#########################################################")
        print ("epoch: %d, updates: %d, validation loss G12: %.5f, validation loss G21: %.5f." % (_epoch, n_iter, val_G12_loss, val_G21_loss))
        # Writing values to SummaryWriter
        writer.add_scalar('val/G12', val_G12_loss, n_iter)
        writer.add_scalar('val/G21', val_G21_loss, n_iter)
        print ('')
        print ("************** Validation Sample Sentences **************")
        for _n_sent in range(len(val_sents_s1_str)):
            print ('Sentence %d' % (_n_sent + 1))
            print ('S1     : ' + val_sents_s1_str[_n_sent])
            print ('S2_hat : ' + val_sents_s2_hat_str[_n_sent])
            print ('S2     : ' + val_sents_s2_str[_n_sent])
            print ('S1_hat : ' + val_sents_s1_hat_str[_n_sent])
            print ('')
        print ("*********************************************************")
        print ('')

        # saving the model to disk
        if val_G12_loss <= best_val_G12_loss or val_G21_loss <= best_val_G21_loss:
            if val_G12_loss <= best_val_G12_loss:
                best_val_G12_loss = val_G12_loss
                best_G12_state_dict = G12.state_dict()
                save_state = {
                    'epoch': _epoch,
                    'G12_state_dict': best_G12_state_dict,
                    'G21_state_dict': best_G21_state_dict,
                    'optimizer_G12': optimizer_G12.state_dict(),
                    'optimizer_G21': optimizer_G21.state_dict(),
                    'opts': opts, 
                    'rec_loss_G12': best_val_G12_loss,
                    'rec_loss_G21': best_val_G21_loss
                }
            if val_G21_loss <= best_val_G21_loss:
                best_val_G21_loss = val_G21_loss
                best_G21_state_dict = G21.state_dict()
                save_state = {
                    'epoch': _epoch,
                    'G12_state_dict': best_G12_state_dict,
                    'G21_state_dict': best_G21_state_dict,
                    'optimizer_G12': optimizer_G12.state_dict(),
                    'optimizer_G21': optimizer_G21.state_dict(),
                    'opts': opts, 
                    'rec_loss_G12': best_val_G12_loss,
                    'rec_loss_G21': best_val_G21_loss
                }
            model_name = opts.model_name + '_best.net'
            torch.save(save_state, os.path.join(opts.save_path, model_name))

        save_state = {
            'epoch': _epoch,
            'G12_state_dict': G12.state_dict(),
            'G21_state_dict': G21.state_dict(),
            'optimizer_G12': optimizer_G12.state_dict(),
            'optimizer_G21': optimizer_G21.state_dict(),
            'opts': opts, 
            'rec_loss_G12': val_G12_loss,
            'rec_loss_G21': val_G21_loss
        }
        model_name = opts.model_name + '_latest.net'
        torch.save(save_state, os.path.join(opts.save_path, model_name))
        
        # Update scheduled sampling factor
        if opts.enable_scheduled_sampling:
            if opts.scheduled_sampling_decay_type == 'linear':
                opts.scheduled_eps = max(1.0 - opts.scheduled_sampling_decay_factor * _epoch, opts.scheduled_sampling_min_eps)
            elif opts.scheduled_sampling_decay_type == 'inv_sigma_decay':
                opts.scheduled_eps = max(opts.scheduled_sampling_decay_factor / (opts.scheduled_sampling_decay_factor + math.exp(_epoch / opts.scheduled_sampling_decay_factor)), opts.scheduled_sampling_min_eps)
            else:
                raise ValueError('Only linear and inv_sigma_decay sampling enabled currently!') 
             
def load_embedding_from_seq2seq(embedding, opts, etype='s1'):
    if etype == 's1':
        s2s_state_dict = torch.load(opts.pretrained_seq2seq_model_path)['G12_state_dict']
    else:
        s2s_state_dict = torch.load(opts.pretrained_seq2seq_model_path)['G21_state_dict']
    for k in s2s_state_dict.keys():
        if not k.startswith('encoder.embedding.embedding.weight'):
            s2s_state_dict.pop(k)
    s2s_state_dict['embedding.weight'] = s2s_state_dict.pop('encoder.embedding.embedding.weight')
    model_dict = embedding.state_dict()
    # filter out unecessary keys
    s2s_state_dict = {k: v for k, v in s2s_state_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(s2s_state_dict)
    # load the new state_dict
    embedding.load_state_dict(model_dict)

def load_generator_from_seq2seq(G, opts, gtype='G12'):
    if gtype == 'G12':
        s2s_state_dict = torch.load(opts.pretrained_seq2seq_model_path)['G12_state_dict']
    else:
        s2s_state_dict = torch.load(opts.pretrained_seq2seq_model_path)['G21_state_dict']
    for k in s2s_state_dict.keys():
        if k.startswith('encoder.embedding') or k.startswith('decoder.embedding'):
            s2s_state_dict.pop(k)
    model_dict = G.state_dict()
    # filter out unecessary keys
    s2s_state_dict = {k: v for k, v in s2s_state_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(s2s_state_dict)
    # load the new state_dict
    G.load_state_dict(model_dict)

def load_discriminator_from_seq2seq(D, opts, dtype='D1'):
    if dtype == 'D1':
        s2s_state_dict = torch.load(opts.pretrained_seq2seq_model_path)['G12_state_dict']
    else:
        s2s_state_dict = torch.load(opts.pretrained_seq2seq_model_path)['G21_state_dict']
    for k in s2s_state_dict.keys():
        if k.startswith('decoder'):
            s2s_state_dict.pop(k)
        if k.startswith('encoder.embedding'):
            s2s_state_dict.pop(k)
    s2s_state_dict['rnn.weight_ih_l0'] = s2s_state_dict.pop('encoder.rnn.weight_ih_l0')
    s2s_state_dict['rnn.weight_hh_l0'] = s2s_state_dict.pop('encoder.rnn.weight_hh_l0')
    s2s_state_dict['rnn.bias_ih_l0'] = s2s_state_dict.pop('encoder.rnn.bias_ih_l0')
    s2s_state_dict['rnn.bias_hh_l0'] = s2s_state_dict.pop('encoder.rnn.bias_hh_l0')
    model_dict = D.state_dict()
    # filter out unecessary keys
    s2s_state_dict = {k: v for k, v in s2s_state_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(s2s_state_dict)
    # load the new state_dict
    D.load_state_dict(model_dict)

def finetune_cyclegan(opts):
    print ("################## Finetuning Cyclegan ##################")
    print ("#########################################################")
    print ('')
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
    opts.alpha = 0.0

    embedding_s1 = EmbeddingLayer(opts.vocab_size_s1, opts.embedding_size, opts.padding_idx_s1)
    embedding_s2 = EmbeddingLayer(opts.vocab_size_s2, opts.embedding_size, opts.padding_idx_s2)
    load_embedding_from_seq2seq(embedding_s1, opts, 's1')
    load_embedding_from_seq2seq(embedding_s2, opts, 's2')
    # Initialising generators with pretrained language models
    G12 = Generator(embedding_s1, embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length)
    G21 = Generator(embedding_s2, embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length)

    load_generator_from_seq2seq(G12, opts, 'G12')
    load_generator_from_seq2seq(G21, opts, 'G21')

    # Initialising discriminators with pretrained language models
    D1 = Discriminator(embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p)
    D2 = Discriminator(embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p)

    load_discriminator_from_seq2seq(D1, opts, 'D1')
    load_discriminator_from_seq2seq(D2, opts, 'D2')

    train_cyclegan_run_iterations(opts, embedding_s1, embedding_s2, G12, G21, D1, D2, loader)

def finetune_cyclegan_gumbel(opts):
    print ("################## Finetuning Cyclegan with Gumbel Sampler ##################")
    print ("#############################################################################")
    print ('')
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
    opts.alpha = 0.0

    embedding_s1 = EmbeddingLayer(opts.vocab_size_s1, opts.embedding_size, opts.padding_idx_s1)
    embedding_s2 = EmbeddingLayer(opts.vocab_size_s2, opts.embedding_size, opts.padding_idx_s2)
    load_embedding_from_seq2seq(embedding_s1, opts, 's1')
    load_embedding_from_seq2seq(embedding_s2, opts, 's2')
    # Initialising generators with pretrained language models
    G12 = Generator(embedding_s1, embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length)
    G21 = Generator(embedding_s2, embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p, opts.max_length)

    load_generator_from_seq2seq(G12, opts, 'G12')
    load_generator_from_seq2seq(G21, opts, 'G21')

    # Initialising discriminators with pretrained language models
    D1 = Discriminator(embedding_s1, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p)
    D2 = Discriminator(embedding_s2, opts.hidden_size, \
        opts.num_rnn_layers, opts.use_lstm, opts.dropout_p)

    load_discriminator_from_seq2seq(D1, opts, 'D1')
    load_discriminator_from_seq2seq(D2, opts, 'D2')

    train_cyclegan_run_iterations(opts, embedding_s1, embedding_s2, G12, G21, D1, D2, loader)

def display_opts(opts):
    args = ['batch_size', 'dataFile', 'jsonFile', 'pdataFile', 'pjsonFile', 'shuffle', \
    'train_mode', 'embedding_size', 'hidden_size', 'num_rnn_layers', 'use_lstm', \
    'use_attention', 'epochs', 'lr', 'weight_decay', 'dropout_p', 'max_norm', \
    'enable_scheduled_sampling', 'scheduled_sampling_decay_type', 'scheduled_sampling_decay_factor', \
    'scheduled_sampling_min_eps', 'log_dir', 'num_sample_sents', 'log_iter', \
    'sent_sample_iter', 'save_path', 'model_name', 'pretrained_lm1_model_path', \
    'pretrained_lm2_model_path', 'pretrained_seq2seq_model_path', 'pretrained_glove_vector_path', \
    'use_glove_embeddings', 'num_searches', 'g_update_step_diff', 'd_update_step_diff', \
    'lr_ratio_D_by_G', 'discount_factor', 'lamda_rl', 'lamda_rec_ii', 'lamda_rec_ij', \
    'lamda_cos_ij', 'skip_weight_decay', 'freeze_embeddings', 'clamp_lower', 'clamp_upper', \
    'd_pretrain_num_epochs', 'disc_recalibrate', 'g_update_step_diff_recalib']

    print ('======================= Parameters =======================')
    print ('==========================================================')
    opts_dict = opts.__dict__
    for key in args:
        print ('%-30s: %s' % (key, str(opts_dict[key])))
    print ('')

def main(opts):
    display_opts(opts)
    if opts.train_mode == 'pretrain_lm':
        opts.pdataFile = opts.dataFile
        opts.pjsonFile = opts.jsonFile
        pretrain_language_model(opts)
    elif opts.train_mode == 'train_seq2seq':
        train_seq2seq(opts)
    elif opts.train_mode == 'train_cyclegan':
        train_cyclegan(opts)
    elif opts.train_mode == 'finetune_cyclegan':
        finetune_cyclegan(opts)
    else:
        print ('Error: Unrecognized train_mode:', opts.train_mode)


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
    parser.add_argument('--train_mode', type=str, default='pretrain_lm', help='Training mode: pretrain_lm | train_cyclegan | train_seq2seq | finetune_cyclegan')
    parser.add_argument('--embedding_size', type=int, default=300, help='Word embedding size')
    parser.add_argument('--hidden_size', type=int, default=350, help='Hidden size for RNN')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='Number of hidden layers in RNN')
    parser.add_argument('--use_lstm', type=str2bool, default=True, help='Use LSTM or GRU?')
    parser.add_argument('--use_attention', type=str2bool, default=True, help='Use attention for the decoder RNN')

    # Optimization parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (for generator in case of cyclegan training)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--max_norm', type=float, default=1, help='Max grad norm')
    parser.add_argument('--enable_scheduled_sampling', type=str2bool, default=False, help='Enable scheduled sampling or not?')
    parser.add_argument('--scheduled_sampling_decay_type', type=str, default='linear', help='[linear | exponential | inv_sigma_decay]')
    parser.add_argument('--scheduled_sampling_decay_factor', type=float, default=5e-3, help='factor for scheduled sampling')
    parser.add_argument('--scheduled_sampling_min_eps', type=float, default=0.5, help='minimum epsilon for scheduled sampling')

    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboardX logs')
    parser.add_argument('--num_sample_sents', type=int, default=5, help='Number of sample sentences to print')
    parser.add_argument('--log_iter', type=int, default=10, help='Logging frequency')
    parser.add_argument('--sent_sample_iter', type=int, default=50, help='Logging frequency for printing the sample sentences ')

    # Save path parameters
    parser.add_argument('--save_path', type=str, default='./models', help='Directory where models are saved')
    parser.add_argument('--model_name', type=str, default='model', help='Name to be given for the saved models')

    # Model load parameters
    parser.add_argument('--pretrained_lm1_model_path', type=str, default='./models/yafc_formal_lm_best.net', help='Pretrained language model for S1.')
    parser.add_argument('--pretrained_lm2_model_path', type=str, default='./models/yafc_informal_lm_best.net', help='Pretrained language model for S2.')
    parser.add_argument('--pretrained_seq2seq_model_path', type=str, default='./models/yafc_formal_informal.net', help='Pretrained seq2seq models G12 and G21')

    # Pretrain LM parameters
    parser.add_argument('--pretrained_glove_vector_path', type=str, default='./data/glove/glove.twitter.27B.200d.txt', help='Pretrained GloVe vectors file path')
    parser.add_argument('--use_glove_embeddings', type=str2bool, default=True, help='Use GloVe vectors to initialise LM')

    # Cyclegan parameters
    parser.add_argument('--num_searches', type=int, default=1, help='Number of Monte Carlo search rounds to run')
    parser.add_argument('--g_update_step_diff', type=int, default=25, help='Frequency with which G is updated')
    parser.add_argument('--d_update_step_diff', type=int, default=1, help='Frequency with which D is updated')
    parser.add_argument('--lr_ratio_D_by_G', type=float, default=1.0, help='Learning rate ratio between generator and discriminator')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor for summing rewards')
    parser.add_argument('--lamda_rl', type=float, default=1e-0, help='Scaling factor for REINFORCE loss between G12 and D1 / G21 and D2')
    parser.add_argument('--lamda_rec_ii', type=float, default=1e-2, help='Scaling factor for reconstruction loss (non RL)')
    parser.add_argument('--lamda_rec_ij', type=float, default=1e-3, help='Scaling factor for reconstruction loss (RL)')
    parser.add_argument('--lamda_cos_ij', type=float, default=1e-1, help='Scaling factor for encoder cosine similarity loss (RL)')
    parser.add_argument('--skip_weight_decay', type=float, default=0.995, help='Exponentially decay for skip connection in the decoder')
    parser.add_argument('--freeze_embeddings', type=str2bool, default=True, help='Freeze embeddings for the cyclegan training')
    parser.add_argument('--clamp_lower', type=float, default=-0.01, help='Lower bound for the D weight clamp (required for WGAN)')
    parser.add_argument('--clamp_upper', type=float, default=0.01, help='Upper bound for the D weight clamp (required for WGAN)')
    parser.add_argument('--d_pretrain_num_epochs', type=int, default=3, help='Num. of epochs to pretrain D for')
    parser.add_argument('--disc_recalibrate', type=int, default=100, help='Num. of discriminator iterations after which generator wait period is increased significantly to allow discriminator to catch up')
    parser.add_argument('--g_update_step_diff_recalib', type=int, default=200, help='Num. of steps after which generator is updated after discriminator during recalibration')

    opts = parser.parse_args()
    
    set_random_seeds(123)
    main(opts)
