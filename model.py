import gzip
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
from torchtext.utils import download_from_url
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

## Make the the multiple attention with word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)

## The word RNN model for generating a sentence vector - Experimental (Not verified yet)
class WordRNN(nn.Module):
    def __init__(self, vocab_size,embedsize, batch_size, hid_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        ## Word Encoder
        self.embed = nn.Embedding(vocab_size, embedsize)
        self.wordRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Word Attention
        self.wordattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 2*hid_size,bias=False)
    def forward(self,inp, hid_state):
        emb_out  = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)

        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation),dim=1)

        sent = attention_mul(out_state,attn)
        return sent, hid_state

## The HAN model - - Experimental (Not verified yet)
class SentenceRNN(nn.Module):
    def __init__(self,vocab_size,embedsize, batch_size, hid_size):
        super(SentenceRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        self.wordRNN = WordRNN(vocab_size,embedsize, batch_size, hid_size)
        ## Sentence Encoder
        self.sentRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Sentence Attention
        self.sentattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 2*hid_size,bias=False)
        self.doc_linear = nn.Linear(2*hid_size, 1)

    def forward(self,inp, hid_state_sent, hid_state_word, max_len):
        s = None
    
        ## Generating sentence vector through WordRNN
        for i in range(len(inp[0])):
            r = None
            for j in range(len(inp)):
                if(r is None):
                    r = [inp[j][i]]
                else:
                    r.append(inp[j][i])
            #r1 = np.asarray([sub_list + [0] * (max_len - len(sub_list)) for sub_list in r])
            #print(r1)
            _s, state_word = self.wordRNN(torch.cuda.LongTensor(r).view(-1,self.batch_size), hid_state_word)
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)

                out_state, hid_state = self.sentRNN(s, hid_state_sent)
        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation),dim=1)

        doc = attention_mul(out_state,attn)
        d = self.doc_linear(doc)
        return d, hid_state
    
    def init_hidden_sent(self):
            return Variable(torch.zeros(2, self.batch_size, self.hid_size)).cuda()
    
    def init_hidden_word(self):
            return Variable(torch.zeros(2, self.batch_size, self.hid_size)).cuda()

def train_data(batch_size, review, targets, sent_attn_model, sent_optimizer, criterion):

    state_word = sent_attn_model.init_hidden_word()
    state_sent = sent_attn_model.init_hidden_sent()
    sent_optimizer.zero_grad()
            
    y_pred, state_sent = sent_attn_model(review, state_sent, state_word)

    loss = criterion(y_pred.cuda(), torch.cuda.LongTensor(targets)) 

    max_index = y_pred.max(dim = 1)[1]
    correct = (max_index == torch.cuda.LongTensor(targets)).sum()
    acc = float(correct)/batch_size

    loss.backward()
    
    sent_optimizer.step()
    
    return loss.data[0],acc

class LSTMWithAttentionAutoEncoder(nn.Module):
    def __init__(self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        batch_size,
        pad_token_src,
        attn_flag = False,
        bidirectional=False,
        batch_first=True,
        nlayers=1,
        nlayers_trg=1,
        dropout=0.,
    ):
        super(LSTMWithAttentionAutoEncoder, self).__init__()
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_vocab_size = src_vocab_size
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.pad_token_src = pad_token_src
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.nlayers = nlayers
        self.nlayers_trg = nlayers_trg
        self.dropout = dropout
        self.batch_first = batch_first
        self.attn_flag = attn_flag
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.src_emb_dim, self.pad_token_src)
        
        self.trg_embedding = nn.Embedding(self.src_vocab_size, self.trg_emb_dim, self.pad_token_src)

        if self.bidirectional:
            src_hidden_dim = src_hidden_dim // 2
        
        self.encoder = nn.LSTM(self.src_emb_dim, src_hidden_dim, 
                num_layers=self.nlayers, dropout=self.dropout,
                bidirectional=self.bidirectional, batch_first=self.batch_first)
        if self.attn_flag == 'True':
            self.wordAttn = nn.Linear(2*src_hidden_dim, 2*src_hidden_dim, bias=False)
            self.attn = nn.Linear(2*src_hidden_dim, 2*src_hidden_dim, bias=False)

        self.decoder = nn.LSTM(self.trg_emb_dim, self.trg_hidden_dim,
                num_layers=self.nlayers_trg, dropout=self.dropout,
                bidirectional=False, batch_first=self.batch_first)

        self.enc2dec = nn.Linear(self.src_hidden_dim, self.trg_hidden_dim)
        
        self.dec2vocab = nn.Linear(self.trg_hidden_dim, self.src_vocab_size).cuda()        
        self.init_weights()

    def attnmul(self, outputs, weights):
        attn = None
        for i in range(outputs.size(0)):
            cur = outputs[i]
            wt = weights[i]
            h_i = cur * wt
            h_i = h_i.unsqueeze(0)
            if attn is None:
                attn = h_i
            else:
                attn = torch.cat((attn, h_i), 0)
        return attn
    
    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.enc2dec.bias.data.fill_(0)
        self.dec2vocab.bias.data.fill_(0)

    def get_state(self, input):
        #print(input.size(0), input.size(1))
        batch_size = input.size(0) \
            if self.batch_first else input.size(1)
        #if self.bidirectional:
        #    self.src_hidden_dim = self.src_hidden_dim // 2

        h0_encoder = Variable(torch.zeros(self.nlayers * self.directions, 
                batch_size,
                self.src_hidden_dim // 2))
        c0_encoder = Variable(torch.zeros(self.nlayers * self.directions,
                batch_size,
                self.src_hidden_dim // 2))
        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input):
        src_embedding = self.src_embedding(input)
        trg_embedding = self.trg_embedding(input)
        h0, c0 = self.get_state(input)
        #self.encoder.flatten_parameters()
        encoded, (h_,c_) = self.encoder(src_embedding, (h0, c0))
        if self.bidirectional:
            h_t = torch.cat((h_[-1], h_[-2]), 1)
            c_t = torch.cat((c_[-1], c_[-2]), 1)
        else:
            h_t = h_[-1]
            c_t = c_[-1]
        
        if self.attn_flag == 'True':
            print("Updating self attn weights\n")
            attnWeights = self.wordAttn(h_t)
            word_attn = F.softmax(self.attn(attnWeights), dim=1)
            h_t = self.attnmul(h_t, word_attn)
        
        dec_init = nn.Tanh()(self.enc2dec(h_t))
        trg_h, (_, _) = self.decoder(trg_embedding,(
            dec_init.view(self.nlayers_trg, dec_init.size(0), dec_init.size(1)),
            c_t.view(self.nlayers_trg, c_t.size(0), c_t.size(1)
                )
            )
            )
        trg_reshape = trg_h.contiguous().view(trg_h.size(0) * trg_h.size(1),  trg_h.size(2))
        decoder_logits = self.dec2vocab(trg_reshape)

        out = decoder_logits.contiguous().view(trg_h.size(0), trg_h.size(1), decoder_logits.size(1))

        return out, h_t, trg_embedding

class LSTMWithAttentionAutoEncoderPretrained(nn.Module):
    def __init__(self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        batch_size,
        pad_token_src,
        weights,
        attn_flag=False,
        update_wt = False,
        bidirectional=False,
        batch_first=True,
        nlayers=1,
        nlayers_trg=1,
        dropout=0.,
    ):
        super(LSTMWithAttentionAutoEncoderPretrained, self).__init__()
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_vocab_size = src_vocab_size
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.pad_token_src = pad_token_src
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.nlayers = nlayers
        self.nlayers_trg = nlayers_trg
        self.dropout = dropout
        self.batch_first = batch_first
        self.embed_weights = weights
        self.attn_flag = attn_flag
        self.update_wt = update_wt

        self.src_embedding = nn.Embedding.from_pretrained(self.embed_weights)
        if update_wt == 'True':
            self.embeddings.weight.requires_grad = True
        #self.src_embedding = nn.Embedding(self.src_vocab_size, self.src_emb_dim, self.pad_token_src)
        #self.src_embedding.weight = nn.Parameter(self.embed_weights)
        #self.src_embedding.weight

        self.trg_embedding = nn.Embedding.from_pretrained(weights)
        #self.trg_embedding = nn.Embedding(self.src_vocab_size, self.trg_emb_dim, self.pad_token_src)

        if self.bidirectional:
            src_hidden_dim = src_hidden_dim // 2
        
        self.encoder = nn.LSTM(self.src_emb_dim, src_hidden_dim, 
                num_layers=self.nlayers, dropout=self.dropout,
                bidirectional=self.bidirectional, batch_first=self.batch_first)
        
        if self.attn_flag == 'True':
            self.wordAttn = nn.Linear(2*src_hidden_dim, 2*src_hidden_dim, bias=False)
            self.attn = nn.Linear(2*src_hidden_dim, 2*src_hidden_dim, bias=False)

        self.decoder = nn.LSTM(self.trg_emb_dim, self.trg_hidden_dim,
                num_layers=self.nlayers_trg, dropout=self.dropout,
                bidirectional=False, batch_first=self.batch_first)

        self.enc2dec = nn.Linear(self.src_hidden_dim, self.trg_hidden_dim)
        
        self.dec2vocab = nn.Linear(self.trg_hidden_dim, self.src_vocab_size)
        
        self.init_weights()
    
    def attnmul(self, outputs, weights):
        attn = None
        for i in range(outputs.size(0)):
            cur = outputs[i]
            wt = weights[i]
            h_i = cur * wt
            h_i = h_i.unsqueeze(0)
            if attn is None:
                attn = h_i
            else:
                attn = torch.cat((attn, h_i), 0)
        return attn

    def init_weights(self):
        initrange = 0.1
        if self.update_wt == 'True':
            self.src_embedding.weight.data.uniform_(-initrange, initrange)
            self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        
        self.enc2dec.bias.data.fill_(0)
        self.dec2vocab.bias.data.fill_(0)

    def get_state(self, input):
        #print(input.size(0), input.size(1))
        batch_size = input.size(0) \
            if self.batch_first else input.size(1)
        #if self.bidirectional:
        #    self.src_hidden_dim = self.src_hidden_dim // 2

        h0_encoder = Variable(torch.zeros(self.nlayers * self.directions, 
                batch_size,
                self.src_hidden_dim // 2))
        c0_encoder = Variable(torch.zeros(self.nlayers * self.directions,
                batch_size,
                self.src_hidden_dim // 2))
        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input):
        src_embedding = self.src_embedding(input)
        trg_embedding = self.trg_embedding(input)
        h0, c0 = self.get_state(input)
        #self.encoder.flatten_parameters()
        encoded, (h_,c_) = self.encoder(src_embedding, (h0, c0))
        if self.bidirectional:
            h_t = torch.cat((h_[-1], h_[-2]), 1)
            c_t = torch.cat((c_[-1], c_[-2]), 1)
        else:
            h_t = h_[-1]
            c_t = c_[-1]
        
        if self.attn_flag == 'True':
            print("Updating attn weights \n")
            attnWeights = self.wordAttn(h_t)
            word_attn = F.softmax(self.attn(attnWeights), dim=1)
            h_t = self.attnmul(h_t, word_attn)
        
        dec_init = nn.Tanh()(self.enc2dec(h_t))
        trg_h, (_, _) = self.decoder(trg_embedding,(
            dec_init.view(self.nlayers_trg, dec_init.size(0), dec_init.size(1)),
            c_t.view(self.nlayers_trg, c_t.size(0), c_t.size(1)
                )
            )
            )
        trg_reshape = trg_h.contiguous().view(trg_h.size(0) * trg_h.size(1),  trg_h.size(2))
        decoder_logits = self.dec2vocab(trg_reshape)

        out = decoder_logits.contiguous().view(trg_h.size(0), trg_h.size(1), decoder_logits.size(1))

        return out, h_t, trg_embedding

class BertModel_pretrained(nn.Module):
    def __init__(self,
            text, 
            pretrain_path, 
            bert_model,
            ):
        super(BertModel_pretrained, self).__init__()
        self.model = None
        self.tokenized_text = []
        self.tokenizer = None

        if bert_model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        elif bert_model == 'scibert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

        if bert_model == 'bert':
            self.model = BertModel.from_pretrained(pretrain_path)
        elif bert_model == 'scibert':
            self.model = BertModel.from_pretrained(pretrain_path)
        else:
            self.model = BertModel.from_pretrained(pretrain_path)    

        for i in text:
            self.tokenized_text.append(self.tokenizer.tokenize(i))
        print("Tokenization from bert done")
        #tokens_tensor = []
        #segments_tensor = []
        

    def tokenize_tensor(self, device):
        tokens_tensor = []
        segments_tensor = []
        for token in self.tokenized_text:
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(token)
            segments_ids = [1] * len(token)

            # Convert inputs to PyTorch tensors
            token_tensor = torch.tensor([indexed_tokens])
            segment_tensor = torch.tensor([segments_ids])
            tokens_tensor.append(token_tensor.to(device))
            segments_tensor.append(segment_tensor.to(device))
        return tokens_tensor, segments_tensor

    def get_sentenceEmbedding(self, tokens_tensor, segments_tensor, length):
        if tokens_tensor.shape[1] > 0:
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensor)
            layer_i = 0
            batch_i = 0
            token_i = 0

            # Will have the shape: [# tokens, # layers, # features]
            token_embeddings = [] 

            # For each token in the sentence...
            for token_i in range(len(encoded_layers[layer_i][batch_i])):
              
              # Holds 12 layers of hidden states for each token 
              hidden_layers = [] 
              
              # For each of the 12 layers...
              for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]
                if vec is None:
                    print("No vector found in layer for token:", layer_i, token_i)
                hidden_layers.append(vec)
                
              token_embeddings.append(hidden_layers)

            # Sanity check the dimensions:
            print ("Number of tokens in sequence:", len(token_embeddings))
            print ("Number of layers per token:", len(token_embeddings[0]))
            #concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
            summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
            sentence_embedding = torch.mean(encoded_layers[11], 1)
            #print ("Our final sentence embedding vector of shape:", sentence_embedding.shape)
            return sentence_embedding

def decode(model, logits, vocab_size):
    """Return probability distribution over words."""
    logits_reshape = logits.view(-1, vocab_size)
    out_probs = F.softmax(logits_reshape)
    word_probs = out_probs.view(
        logits.size()[0], logits.size()[1], logits.size()[2]
    )
    return out_probs, word_probs
 



