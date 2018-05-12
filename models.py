import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from numpy.random import binomial

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 100

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx=2):
        """Create a new EmbeddingLayer model.
        Args:
            vocab_size: Size of the vocabulary
            embedding_size:  Size of the word embedding
            padding_idx: replaces occurences of padding_idx with 0 vector (default: 2)
        """
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

    def forward(self, x):
        """Forward function for EmbeddingLayer
        Args:
            x: Input sentences of shape: batch_size x seq_len
        Returns:
            embedded: Embedded vector of shape: batch_size x seq_len x embedding_size
        """
        embedded = self.embedding(x)
        return embedded

class LanguageModel(nn.Module):
    def __init__(self, input_embedding, hidden_size, num_rnn_layers=1, use_lstm=True, dropout_p=0.1,):
        super(LanguageModel, self).__init__()
        self.vocab_size = input_embedding.vocab_size
        self.embedding_size = input_embedding.embedding_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.use_lstm = use_lstm
        self.dropout_p = dropout_p

        self.embedding = input_embedding
        if use_lstm:
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, num_rnn_layers)
        else:
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, num_rnn_layers)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)

    def forward(self, x, hidden=None):
        """Forward function for LanguageModel
        Args:
            x: Input sentences of shape: batch_size x seq_len
            hidden: Previous hidden layer of shape: num_rnn_layers x batch_size x hidden_size
        Returns:
            output: Log Softmax probs for predicting the words Shape: batch_size x seq_len x vocab_size
        """
        if hidden is None:
            if not self.use_lstm:
                hidden = self.initHidden(x.size(0))
            else:
                hidden = (self.initHidden(x.size(0)), self.initHidden(x.size(0)))

        embedded = self.embedding(x)
        embedded = self.dropout1(embedded)
        embedded = torch.transpose(embedded, 0, 1)
        output, hidden = self.rnn(embedded, hidden)

        output = torch.transpose(output, 0, 1)
        # batch_size x seq_len x hidden_size
        batch_size = output.size(0)
        seq_len = output.size(1)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.dropout2(output)
        output = self.out(output)
        output = output.view(batch_size, seq_len, -1)
        output = F.log_softmax(output, dim=2)
        return output

    def sample(self, x, max_length, hidden=None):
        """Sample sentences from the LanguageModel
        Args:
            x: <sos> tokens of shape: batch_size x seq_len
            max_length: length of sentence to generate
        Returns:
            sents: sentences of shape: batch_size x max_length
        """
        if hidden is None:
            if not self.use_lstm:
                hidden = self.initHidden(x.size(0))
            else:
                hidden = (self.initHidden(x.size(0)), self.initHidden(x.size(0)))

        sents = None
        for step in range(max_length):
            embedded = self.embedding(x)
            embedded = self.dropout1(embedded)
            embedded = torch.transpose(embedded, 0, 1)
            output, hidden = self.rnn(embedded, hidden)

            output = torch.transpose(output, 0, 1)
            # batch_size x 1 x hidden_size
            batch_size = output.size(0)
            output = output.contiguous().view(-1, self.hidden_size)
            output = self.dropout2(output)
            output = self.out(output)
            # batch_size x vocab_size
            output = F.softmax(output, dim=1)
            tokens = output.multinomial(num_samples=1)
            x = tokens
            if sents is None:
                sents = tokens
            else:
                sents = torch.cat([sents, tokens], dim=1)
        return sents

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class EncoderRNN(nn.Module):
    def __init__(self, input_embedding, hidden_size, num_rnn_layers=1, use_lstm=True, dropout_p=0.1):
        """Create a new EmbeddingLayer model.
        Args:
            input_embedding: TODO
            num_rnn_layers: Number of stacked RNN cells (default: 1)
            use_lstm: if True uses LSTM cells otherwise uses GRU cells (default:True)
        """
        super(EncoderRNN, self).__init__()
        self.input_vocab_size = input_embedding.vocab_size
        self.embedding_size = input_embedding.embedding_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.use_lstm = use_lstm
        self.dropout_p = dropout_p

        self.embedding = input_embedding
        self.dropout = nn.Dropout(self.dropout_p)
        if use_lstm:
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_rnn_layers)
        else:
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_rnn_layers)

    def forward(self, x, hidden=None):
        """Forward function for EncoderRNN
        Args:
            x: Input sentences of shape: batch_size x seq_len
            hidden: Previous hidden layer of shape: num_rnn_layers x batch_size x hidden_size
        Returns:
            output: Output features h_t from the last layer of the RNN, for each t. Shape: seq_len x batch_size x hidden_size
            hidden: The hidden state for t = seq_len. Shape: num_rnn_layers x batch_size x hidden_size
        """
        if hidden is None:
            if not self.use_lstm:
                hidden = self.initHidden(x.size(0))
            else:
                hidden = (self.initHidden(x.size(0)), self.initHidden(x.size(0)))

        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        # converting shape from (batch x seq_len x embed_size) to (seq_len x batch x embed_size)
        output = torch.transpose(embedded, 0, 1)
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_embedding, hidden_size, num_rnn_layers=1, use_lstm=True, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.output_vocab_size = output_embedding.vocab_size
        self.embedding_size = output_embedding.embedding_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_rnn_layers = num_rnn_layers
        self.use_lstm = use_lstm

        self.embedding = output_embedding
        self.attn = nn.Linear(self.embedding_size + self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.embedding_size + self.hidden_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if use_lstm:
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_rnn_layers)
        else:
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_rnn_layers)
        self.out = nn.Linear(self.hidden_size, self.output_vocab_size)

    def forward(self, x, hidden, encoder_outputs, alpha):
        # x: b x 1
        embedded = self.embedding(x)
        # b x 1 x embedding_size
        embedded = self.dropout(embedded)
        # b x 1 x embedding_size
        embedded = torch.transpose(embedded, 0, 1)
        # 1 x b x embedding_size

        if self.use_lstm:
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        else:
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # b x max_length
        encoder_outputs = torch.transpose(encoder_outputs, 0, 1)
        # b x max_length x hidden_size
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        # b x 1 x hidden_size
        attn_applied = torch.transpose(attn_applied, 0, 1)
        # 1 x b x hidden_size
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # b x (hidden_size + embedding_size)
        output = self.attn_combine(output).unsqueeze(0)
        # 1 x b x hidden_size

        # TODO: need to make the skip connection from inputs to outputs
        output = ((1.0 - alpha) * output) + (alpha * embedded)

        output = F.relu(output)
        # 1 x b x hidden_size
        output, hidden = self.rnn(output, hidden)
        # output: 1 x b x hidden_size
        # hidden: 1 x b x hidden_size

        output = F.log_softmax(self.out(output[0]), dim=1)
        # b x output_vocab_size
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, output_embedding, hidden_size, num_rnn_layers=1, use_lstm=True, dropout_p=0.1, max_length=MAX_LENGTH):
        super(DecoderRNN, self).__init__()
        self.output_vocab_size = output_embedding.vocab_size
        self.embedding_size = output_embedding.embedding_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_rnn_layers = num_rnn_layers
        self.use_lstm = use_lstm

        self.embedding = output_embedding
        self.dropout = nn.Dropout(self.dropout_p)
        if use_lstm:
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_rnn_layers)
        else:
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_rnn_layers)
        self.out = nn.Linear(self.hidden_size, self.output_vocab_size)

    def forward(self, x, hidden, encoder_outputs, alpha):
        # x: b x 1
        embedded = self.embedding(x)
        # b x 1 x embedding_size
        embedded = self.dropout(embedded)
        # b x 1 x embedding_size
        output = torch.transpose(embedded, 0, 1)
        # 1 x b x embedding_size
        output, hidden = self.rnn(output, hidden)
        # output: 1 x b x hidden_size
        # hidden: 1 x b x hidden_size

        output = F.log_softmax(self.out(output[0]), dim=1)
        # b x output_vocab_size
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Generator(nn.Module):
    def __init__(self, input_embedding, output_embedding, hidden_size, num_rnn_layers=1, use_lstm=True, dropout_p=0.1, max_length=MAX_LENGTH, use_attention=True):
        super(Generator, self).__init__()
        self.input_vocab_size = input_embedding.vocab_size
        self.input_embedding_size = input_embedding.embedding_size
        self.output_vocab_size = output_embedding.vocab_size
        self.output_embedding_size = output_embedding.embedding_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.use_lstm = use_lstm
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_attention = use_attention

        self.encoder = EncoderRNN(input_embedding, self.hidden_size, num_rnn_layers=self.num_rnn_layers, use_lstm=self.use_lstm, \
            dropout_p=self.dropout_p)
        if self.use_attention:
            self.decoder = AttnDecoderRNN(output_embedding, self.hidden_size, num_rnn_layers=self.num_rnn_layers, use_lstm=self.use_lstm, \
                dropout_p=self.dropout_p, max_length=self.max_length)
        else:
            self.decoder = DecoderRNN(output_embedding, self.hidden_size, num_rnn_layers=self.num_rnn_layers, use_lstm=self.use_lstm, \
                dropout_p=self.dropout_p, max_length=self.max_length)

    def encode(self, x, hidden=None):
        if hidden is None:
            if not self.use_lstm:
                hidden = self.initHidden(x.size(0))
            else:
                hidden = (self.initHidden(x.size(0)), self.initHidden(x.size(0)))

        encoder_outputs, hidden = self.encoder(x, hidden)
        return encoder_outputs, hidden

    def decode(self, num_steps, decoder_input, hidden, encoder_outputs, alpha, scheduled_eps=None):
        """Rollout the decoder for a number of steps with teacher forcing
        Args:
            num_steps: integer > 0
            decoder_input: tokens, shape: batch_size x num_steps
            hidden: decoder hidden state, shape: num_rnn_layers x batch_size x hidden_size
            encoder_outputs: encoder intermediate hidden states, shape: max_length x batch_size x hidden_size
        Returns:
            rollouts: the tokens generated by the decoder, shape: batch_size x num_steps
            log_probs_accumulated: batch_size x max_length x vocab_size Variable
            scheduled_eps: epsilon for scheduled sampling, 
                           i.e., p(sampling true input) = epsilon, p(sampling previously predicted word) = 1 - epsilon
        """
        assert (num_steps > 0), "num_steps for decoder rollout is 0"
        rollouts = None
        log_probs_accumulated = None
        scheduled_choice = 1
        for _step in range(num_steps):
            if _step > 0 and scheduled_eps is not None: 
                scheduled_choice = binomial(1, scheduled_eps)
            if scheduled_choice == 1:
                tokens, log_probs, hidden = self.decoder_step(decoder_input[:, _step].contiguous().view(-1, 1), hidden, encoder_outputs, alpha)
            else:
                tokens, log_probs, hidden = self.decoder_step(tokens_past, hidden, encoder_outputs, alpha)

            log_probs = log_probs.view(log_probs.size(0), 1, -1)
            if rollouts is None:
                rollouts = tokens
                log_probs_accumulated = log_probs
            else:
                rollouts = torch.cat((rollouts, tokens), dim=1)
                log_probs_accumulated = torch.cat((log_probs_accumulated, log_probs), dim=1)
            tokens_past = tokens

        return rollouts, log_probs_accumulated

    def decoder_step(self, decoder_input, hidden, encoder_outputs, alpha):
        """Rollout the decoder for a single step
        Args:
            num_steps: integer > 0
            decoder_input: tokens, shape: batch_size x 1
            hidden: decoder hidden state, shape: num_rnn_layers x batch_size x hidden_size
            encoder_outputs: encoder intermediate hidden states, shape: max_length x batch_size x hidden_size
        Returns:
            tokens: the tokens generated by the decoder, shape: batch_size x 1
            log_probs: log probabilites for the tokens, shape: batch_size x vocab_size
            hidden: decoder hidden state, shape: num_rnn_layers x batch_size x hidden_size
        """
        output, hidden = self.decoder(decoder_input, hidden, encoder_outputs, alpha)
        probs = F.softmax(output, dim=1)
        tokens = probs.multinomial(num_samples=1)
        log_probs = F.log_softmax(output, dim=1)
        return tokens, log_probs, hidden

    def decoder_rollout(self, num_steps, decoder_input, hidden, encoder_outputs, alpha):
        """Rollout the decoder for a number of steps
        Args:
            num_steps: integer > 0
            decoder_input: tokens, shape: batch_size x 1
            hidden: decoder hidden state, shape: num_rnn_layers x batch_size x hidden_size
            encoder_outputs: encoder intermediate hidden states, shape: max_length x batch_size x hidden_size
        Returns:
            rollouts: the tokens generated by the decoder, shape: batch_size x num_steps
            log_probs_accumulated: batch_size x max_length x vocab_size Variable
        """
        assert (num_steps > 0), "num_steps for decoder rollout is 0"
        rollouts = None
        log_probs_accumulated = None
        for _step in range(num_steps):
            tokens, log_probs, hidden = self.decoder_step(decoder_input, hidden, encoder_outputs, alpha)
            log_probs = log_probs.view(log_probs.size(0), 1, -1)
            decoder_input = tokens
            if rollouts is None:
                rollouts = tokens
                log_probs_accumulated = log_probs
            else:
                rollouts = torch.cat((rollouts, tokens), dim=1)
                log_probs_accumulated = torch.cat((log_probs_accumulated, log_probs), dim=1)

        return rollouts, log_probs_accumulated

    # TODO add weight initialisation

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Discriminator(nn.Module):
    def __init__(self, output_embedding, hidden_size, num_rnn_layers=1, use_lstm=True, dropout_p=0.1):
        super(Discriminator, self).__init__()
        self.output_vocab_size = output_embedding.vocab_size
        self.embedding_size = output_embedding.embedding_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.use_lstm = use_lstm
        self.dropout_p=dropout_p

        self.embedding = output_embedding
        self.dropout = nn.Dropout(self.dropout_p)
        if use_lstm:
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_rnn_layers)
        else:
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_rnn_layers)
        self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden=None):
        """Forward function for Discriminator
        Args:
            x: Input sentences of shape: batch_size x seq_len
            hidden: Previous hidden layer of shape: num_rnn_layers x batch_size x hidden_size
        Returns:
            output: Single scalar value (needed for WGAN) shape: batch_size x 1
        """
        if hidden is None:
            if not self.use_lstm:
                hidden = self.initHidden(x.size(0))
            else:
                hidden = (self.initHidden(x.size(0)), self.initHidden(x.size(0)))

        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        embedded = torch.transpose(embedded, 0, 1)
        rnn_outputs, hidden = self.rnn(embedded, hidden)

        if self.use_lstm:
            output = hidden[0].view(-1, self.hidden_size)
        else:
            output = hidden.view(-1, self.hidden_size)
        output = self.out(output)
        return output

    # TODO add weight initialisation

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

