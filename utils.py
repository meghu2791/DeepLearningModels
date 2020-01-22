import sys
import os
import operator
import re
import string
import nltk
#Uncomment only for downloading NLTK collections (all): nltk.download()
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import codecs
import torch
from torch import nn
from torch.autograd import Variable
import operator
from torchtext.utils import download_from_url
import io
import os
#import gensim
#from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from numpy import dot
from numpy.linalg import norm


def clean_data(line):
    #stemmer = PorterStemmer()
    srclist = []
    spltokens = ['<START>', '<END>', '<SEC>', '<ENDSEC>']
    punctuation = '!"#$%&’.()*+,-/:;=?@[\]^_\'`{|}\"~<>'
    lemma = WordNetLemmatizer()

    l = re.sub(r'\d+',' ',line) #remove numbers
        
    #remove punctuation
    for c in punctuation:
        l = l.replace(c, ' ')
    #l = l.translate(string.maketrans("", ""), punctuation) #remove punctuation marks
    word_tokens = l.lower().strip().split() #covert to lower case
    
    #remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [ i for i in word_tokens if not i in stop_words ]
    
    #lemmatization - gets words in its root form (having, had -> have for example)
    for w in tokens:
        srclist.append(lemma.lemmatize(w))
    return srclist


def read_docs(srcfile, m):
    src = ''
    print("Reading...." + str(srcfile))

    try:
        with open(srcfile, 'r') as f:
            start = 1000
            end = 0
            data = []
            ind = 0
            #for ind, line in enumerate(f):
            while True:
                line = f.readline()
                #if lang =='en':
                if 'abstract' in line.lower():
                    start = ind
                    print("Start index found!", start)
                elif 'introduction' in line.lower() or ind > start+40:
                    end = ind
                    print("End index found ", end)
                    break
                ind += 1
                data.append(line)
            #print(start,  end, srcfile)
            for ind, line in enumerate(data):
                temp = ''
                if start < ind:
                    list_str = line.split()
                    for l in list_str:
                        if re.match("[a-zA-Z0-9_]", l):
                            if not isinstance(l, str):
                                temp += ' '
                            else:
                                temp = temp + ' ' + l + ' '
                    src += ' '.join(clean_data(temp))
                    src += ' '
    except UnicodeDecodeError:
        with open(srcfile, 'r', encoding="ISO-8859-1") as f:
            start = 1000
            end = 0
            data = []
            for ind, line in enumerate(f):
                #if detect(line) == 'en':
                if 'abstract' in line.lower():
                    start = ind
                elif 'introduction' in line.lower() or ind > start+40:
                    end = ind
                    break
                data.append(line)
            
            for ind, line in enumerate(data):
                temp = ''
                if start < ind:
                    list_str = line.split()
                    for l in list_str:
                        if re.match("[a-zA-Z0-9_]", l):
                            if not isinstance(l, str):
                                temp += ' '
                            else:
                                temp = temp + ' ' + l + ' '
                    src += ' '.join(clean_data(temp))
                    src += ' '

    f.close()
    
    return src.split()
                
def build_vocab(src):
    vocab = dict()

    for line in src:
        for w in line:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

    if '<s>' in vocab:
        del vocab['<s>']
    if '<\s>' in vocab:
        del vocab['<\s>']
    if '<unk>' in vocab:
        del vocab['<unk>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    
    sorted_vocab = sorted(vocab.items(),
            key=operator.itemgetter(1),
            reverse=True)

    sorted_words = [x[0] for x in sorted_vocab[:30000]]

    word2idx = {'<s>' : 0,
            '</s>' : 1,
            '<unk>' : 2,
            '<pad>' : 3 }

    idx2word = { 0 : '<s>',
            1 : '</s>',
            2 : '<unk>',
            3 : '<pad>' }

    for idx, w in enumerate(sorted_words):
        word2idx[w] = idx+4
        idx2word[idx+4] = w
    
    return word2idx, idx2word
    
def read_docs(srcfile):
        src = ''
        print("Reading...." + str(srcfile))
        try:
            with open(srcfile, 'r') as f:
                for ind, line in enumerate(f):
                    if line is not None:
                        src += ' '.join(clean_data(line))
        except UnicodeDecodeError:
            with open(srcfile, 'r', encoding="ISO-8859-1") as f:
                for ind, line in enumerate(f):
                    if line is not None:
                        src += ' '.join(clean_data(line))
                #src.append(temp)
        f.close()
        return src

def read_data(srcfile):
    '''
    src = []
    try:
        with open(srcfile, 'r') as f:
            for ind, line in enumerate(f):
                line = line.split('.')
                for i in line:
                    temp = clean_data(i)
                    src.append(temp)
    except UnicodeDecodeError:
         with open(srcfile, 'r', encoding="ISO-8859-1") as f:
                for ind, line in enumerate(f):
                    line = line.split('.')
                    for i in line:
                        temp = clean_data(i)
                        src.append(temp)
        #remove = len(src) % 10
        #src = src[:len(src)-remove]
    f.close()
    '''
    word2idx, idx2word = build_vocab(srcfile)
    return srcfile, word2idx, idx2word

def find_max_length(src):
    lens = [len(line) for line in src]
    max_len = max(lens)
    return max_len

def get_batch(src, word2idx, idx, batch_size, max_len):
    lens = [len(line) for line in src[idx:idx+batch_size]]
    src_lines = []

    for line in src[idx:idx+batch_size]:
        temp = []
        for w in line:
            if w not in word2idx:
                temp.append(word2idx['<unk>'])
            else:
                temp.append(word2idx[w])
        if len(temp) < max_len:
            for i in range(len(temp), max_len):
                temp.append(word2idx['<pad>'])
        src_lines.append(temp)
    #print(src_lines)
    
    mask = [([1] * (i)) + ([0] * (max_len - i))
        for i in lens
    ]        
    
    src_lines = torch.LongTensor(src_lines)
    mask = torch.FloatTensor(mask)
    out_lines = src_lines

    return src_lines, out_lines, lens, mask


def load_Wikiword2vecModel(cache):

    name_base = 'wiki.en.vec'
    _direct_en_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec'
    '''
    name_base = 'wiki.en.vec'
    _direct_en_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip'
    #_direct_en_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
    #corpus = api.load('text8', return_path=True)  # download the corpus and return it opened as an iterable
    #model = Word2Vec(corpus)
    '''
    destination = os.path.join(cache, str(name_base))
    print(destination)
    if not os.path.isfile(destination):
        download_from_url(_direct_en_url, destination) 
    ext = os.path.splitext(cache)[1][1:]
    if ext == 'zip':
        with zipfile.ZipFile(destination, "r") as zf:
            zf.extractall(cache)
    elif ext == 'gz':
        with tarfile.open(destination, 'r:gz') as tar:
            tar.extractall(path=cache)
    
    model = KeyedVectors.load_word2vec_format(destination)
    return model

def get_embeddingWeights(model, n_words, word2idx, embedding_dim=300):
    embeddings = np.zeros((n_words, embedding_dim))
    for word in word2idx:
        index = word2idx.get(word)
        try:
            vector = model[word]
            embeddings[index] = vector
        except KeyError:
            embeddings[index] = np.random.normal(scale=0.6, size=(embedding_dim, ))
    return torch.from_numpy(embeddings).float()

def read_truthValue(path):
    df = pd.read_csv(os.path.join(path, "bids_file"), sep=' ', header=None)
    df.columns = ['paper_id', 'reviewer', 'preference']
    return df

def build_reviewerAndsubmitter_embeddings(doc_name, embedding, df_filtered):
    reviewer = dict()
    submitter = dict()
    reviewer_paper = dict()
    for i in range(len(doc_name)):
        if "archive_papers" in doc_name[i]:
            author = doc_name[i].split('/')[-2]
            if author in df_filtered['reviewer'].values:
                if author in reviewer:
                    reviewer[author].append(embedding[i])
                    reviewer_paper[author].append(torch.unsqueeze(torch.Tensor(embedding[i]), 0))
                else:
                    reviewer[author] = []
                    reviewer_paper[author] = []
                    reviewer[author].append(embedding[i])
                    reviewer_paper[author].append(torch.unsqueeze(torch.Tensor(embedding[i]), 0))
        else:
            paper = doc_name[i].split('/')[-1]
            paper_id = re.sub('\D', '', paper)
            submitter[paper_id] = torch.unsqueeze(torch.Tensor(embedding[i]), 0)
    return reviewer, submitter, reviewer_paper

def vectorExtrema(reviewer):
    for each in reviewer.keys():
        var = []
        for i in range(len(reviewer[each])):
            var.append(torch.Tensor(reviewer[each][i]).unsqueeze(0))
        reviewer[each] = torch.unsqueeze(torch.max(torch.cat(var), 0).values, 0)
    return reviewer


def vectorMean(reviewer):
    for each in reviewer.keys():
        var = []
        for i in range(len(reviewer[each])):
            var.append(torch.Tensor(reviewer[each][i]).unsqueeze(0))
        reviewer[each] = torch.unsqueeze(torch.mean(torch.cat(var,0), 0), 0)
    return reviewer



def get_batch_eval(paper_emb, rev_emb, trg_value, idx, batch_size):
    paper_lines = Variable(torch.stack(paper_emb[idx:idx+batch_size]).squeeze(), requires_grad=True)
    review_emb = Variable(torch.stack(rev_emb[idx:idx+batch_size]).squeeze(), requires_grad=True)
    trg = torch.stack(trg_value[idx:idx+batch_size]).squeeze()
    return paper_lines, review_emb, trg

def train_classification(epochs, model, train_data_sub, train_data_rev, labels, save_dir, criterion, optimizer, batch_size, m_name=''):
    losses= []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(0, len(labels), batch_size):
            tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, batch_size) 
            optimizer.zero_grad()
            prediction = model(tr_sub, tr_rev)
            loss = criterion(prediction, y.argmax(dim=1))
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/batch_size)
        print("Epoch:", e_num, " Loss:", losses[-1])
        print("GPU memory consumption for epoch" + str(e_num) + " " + str(torch.cuda.memory_allocated()))
        torch.save(model, os.path.join(save_dir, str(m_name+".model")))
    print("Model training completed!!")


def train_d2v(epochs, model, train_data_sub, train_data_rev, labels, save_dir, criterion, optimizer, batch_size):
    losses= []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(0, len(labels), batch_size):
            tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, batch_size) 
            optimizer.zero_grad()
            prediction = model(tr_sub, tr_rev)
            print(prediction, y.argmax(dim=1))
            loss = criterion(prediction, y.argmax(dim=1))
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/batch_size)
        print("Epoch:", e_num, " Loss:", losses[-1])
        print("GPU memory consumption for epoch" + str(e_num) + " " + str(torch.cuda.memory_allocated()))
        torch.save(model, os.path.join(save_dir, "doc2vec_bid.model"))
    print("Model training completed!!")

def train_seq(epochs, model, train_data_sub, train_data_rev, labels, save_dir, criterion, optimizer, batch_size, m_name):
    losses= []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(0, len(labels), batch_size):
            tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, batch_size) 
            optimizer.zero_grad()
            prediction = model(tr_sub, tr_rev)
            loss = criterion(prediction, y.argmax(dim=1))
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/batch_size)
        print("Epoch:", e_num, " Loss:", losses[-1])
        print("GPU memory consumption for epoch" + str(e_num) + " " + str(torch.cuda.memory_allocated()))
        torch.save(model, os.path.join(save_dir, str("seq_bid"+m_name+".model")))
    print("Model training completed!!")

def train_bert(epochs, model, train_data_sub, train_data_rev, labels, save_dir, criterion, optimizer, m, batch_size):
    losses= []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(0, len(labels), batch_size):
            tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, batch_size) 
            optimizer.zero_grad()
            prediction = model(tr_sub, tr_rev)
            loss = criterion(prediction, y.argmax(dim=1))
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/batch_size)
        print("Epoch:", e_num, " Loss:", losses[-1])
        print("GPU memory consumption for epoch" + str(e_num) + " " + str(torch.cuda.memory_allocated()))
        torch.save(model, os.path.join(save_dir, "bid.model"))
    print("Model training completed!!")

def eval_bert(path, train_data_sub, train_data_rev, labels, m, criterion, submitter_ids, reviewer_ids):
    model = torch.load(os.path.join(path, "bid.model"))
    with torch.no_grad():
        model.eval()
        class_label = 0
        trg_label = 0
        correct = 0
        wrong = 0
        loss_test = 0
        with open(os.path.join(path,"test_results.txt"), "w") as out:
            for i in range(len(labels)):
                prediction = model(train_data_sub[i], train_data_rev[i])
                print(prediction, labels[i].unsqueeze(0).argmax(dim=1))
                class_label = prediction.argmax(dim=1).squeeze()
                trg_label = labels[i].argmax(dim=-1)
                loss = criterion(prediction, labels[i].unsqueeze(0).argmax(dim=1))   # must be (1. nn output, 2. target)
                loss_test += loss.item()
                if class_label == trg_label:
                    correct += 1
                else:
                    print(class_label, trg_label)
                    wrong += 1
                out.write(str(submitter_ids[i]) + " " + str(reviewer_ids[i]) + " " + str(class_label.data.cpu()) + " " + str(trg_label.data.cpu()))
            print("Accuracy:", correct/len(labels), " Test Loss:", loss_test/len(labels))
            out.write("Accuracy:"+ str(correct/len(labels)) + " Test Loss:" + str(loss_test/len(labels)))
        out.close()

def eval_d2v(path, train_data_sub, train_data_rev, labels, criterion, submitter_ids, reviewer_ids):
    model = torch.load(os.path.join(path, "doc2vec_bid.model"))
    with torch.no_grad():
        model.eval()
        class_label = 0
        trg_label = 0
        correct = 0
        wrong = 0
        loss_test = 0
        with open(os.path.join(path,"test_results.txt"), "w") as out:
            for i in range(len(labels)):
                #tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, 1)
                #print(tr_sub, y)
                prediction = model(train_data_sub[i], train_data_rev[i])
                print(prediction, labels[i])
                class_label = prediction.argmax(dim=1)
                trg_label = labels[i].argmax(dim=-1)
                loss = criterion(prediction, labels[i].unsqueeze(0).argmax(dim=1))   # must be (1. nn output, 2. target)
                loss_test += loss.item()
                if class_label == trg_label:
                    correct += 1
                else:
                    print(class_label, trg_label)
                    wrong += 1
                out.write(str(submitter_ids[i]) + " " + str(reviewer_ids[i]) + " " + str(class_label.data.cpu()) + " " + str(trg_label.data.cpu()))
            print("Accuracy:", correct/len(labels), " Test Loss:", loss_test/len(labels))
            out.write("Accuracy:"+ str(correct/len(labels)) + " Test Loss:" + str(loss_test/len(labels)))
        out.close()


