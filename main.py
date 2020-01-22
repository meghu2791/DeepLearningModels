import sys
#sys.path.append("/network/home/bhatmemo/CERMINE")
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
#from xml_parser import ParseXML - Use this only if you need to parse pdfs
from code.utils import read_data, get_batch, find_max_length, load_Wikiword2vecModel, get_embeddingWeights, build_reviewerAndsubmitter_embeddings, vectorExtrema, vectorMean, read_truthValue
from code.utils import read_docs
from code.model import LSTMWithAttentionAutoEncoder, SentenceRNN, LSTMWithAttentionAutoEncoderPretrained, BertModel_pretrained, decode
from code.eval import Match_Classify, Match_LR, prepare_data, prepare_data_LR, train_LR, eval_LR, train_classification, eval_classification, seq_cosineSimilarity, seq_cosineSimilarity_paper
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import operator

#Read command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)
parser.add_argument('--load_dir', type=str)
parser.add_argument('--pretrain_path', type=str)
parser.add_argument('--train_or_test', type=str)
parser.add_argument('--attn', type=str)
parser.add_argument('--ground_truth', type=str)
parser.add_argument('--embedding_dim', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--layers', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--eval_metric', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--vector', type=str)

args = parser.parse_args()
save_dir = args.save_dir
load_dir = args.load_dir
cache_path = args.pretrain_path
tr_tt = args.train_or_test
attn = args.attn
bids_file = args.ground_truth
model_name = args.model
eval_method = args.eval_metric
batch_size = args.batch_size
src_dim = args.embedding_dim
rnn_dim = args.hidden_size
rnn_depth = args.layers
lr = args.learning_rate
epochs = args.epochs
vector = args.vector

optimizer = "adam"
pad_token_src = 3
#if (model_name is not 'bert') and (model_name is not 'scibert'):
#    token_len = 256
#else:
#    token_len = 512

docVector = []
doc_names = []
#wt = torch.from_numpy(np.zeros((n_words, src_dim)))

#Read the source files
data = []    
for root, dirn, files in os.walk(load_dir):
    for f in files:
        if f.endswith(".txt"):
            print(f)
            #Replace self tokenization method by BPE
            src = read_docs(os.path.join(root,f))
            doc_names.append(os.path.join(root, f))
            if len(src) > token_len:
                data.append(src[:token_len])
            else:
                data.append(src)
            print("\n")

if (model_name is not 'bert') and (model_name is not 'scibert'):
    src, word2idx, idx2word = read_data(data)
    n_words = len(word2idx)
    vocab_size = n_words
    vocab_size = 30004
#preTrain_model = load_Wikiword2vecModel(cache_path)
#wt = get_embeddingWeights(preTrain_model, n_words, word2idx, src_dim) 

logging.info('Model Parameters : ')
logging.info('Source Word Embedding Dim  : %s' % (src_dim))
logging.info('Source RNN Hidden Dim  : %s' % (rnn_dim))
logging.info('Source RNN Depth : %d ' % (rnn_depth))
logging.info('Batch Size : %d ' % (batch_size))
logging.info('Learning Rate : %f ' % (lr))
logging.info('Found %d words in src ' % (vocab_size))

weight_mask = torch.ones(vocab_size).cuda()
weight_mask[word2idx['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
torch.backends.cudnn.benchmark=True

#Set GPU mode if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Ground truth filtering

if model_name == 'LSTMWithAttentionAutoEncoderPretrained':
    preTrain_model = load_Wikiword2vecModel(cache_path)
    wt = get_embeddingWeights(preTrain_model, vocab_size, word2idx, src_dim) 
    model = LSTMWithAttentionAutoEncoderPretrained(
        src_emb_dim=src_dim,
        trg_emb_dim =src_dim,
        src_vocab_size=vocab_size,
        src_hidden_dim=rnn_dim,
        trg_hidden_dim=rnn_dim,
        bidirectional=True,
        batch_size=batch_size,
        pad_token_src=word2idx['<pad>'],
        nlayers=rnn_depth,
        batch_first=True,
        nlayers_trg=1,
        dropout=0.,
        weights=wt,
    )
elif model_name == 'LSTMWithAttentionAutoEncoder':
    model = LSTMWithAttentionAutoEncoder(
        src_emb_dim=src_dim,
        trg_emb_dim =src_dim,
        src_vocab_size=vocab_size,
        src_hidden_dim=rnn_dim,
        trg_hidden_dim=rnn_dim,
        attn_flag=attn,
        bidirectional=True,
        batch_size=batch_size,
        pad_token_src=word2idx['<pad>'],
        nlayers=rnn_depth,
        batch_first=True,
        nlayers_trg=1,
        dropout=0.,

    )
elif model_name =='bert' or model_name == 'scibert':
    model = BertModel_pretrained(data, cache_path, model_name)
else:
    print("No such option exists! Please refer ReadME")
    exit()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=lr)

#Parallelization - Noticed drop in test loss upon DataParallel Wrapper. Investigation needed!
if torch.cuda.device_count() > 0 and (model_name not in 'scibert' and model_name not in 'bert'):
    print("Use", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
model.to(device)

hidden_embed = []
docVector = []
doc_embedding = []
max_len = 10
max_len = find_max_length(src)

if tr_tt == 'train':
    for i in range(epochs):
        losses = []
        for j in range(0, len(src), batch_size):

            input_lines_src, output_lines_src, lens_src, mask_src = get_batch(
                src, word2idx, j,
                batch_size, max_len) 
            decoder_logit, hidden_states, trg_e = model(input_lines_src)
            optimizer.zero_grad()
            #encoder_op.append(encoded)
            print(decoder_logit.shape)
            loss = loss_criterion(
                decoder_logit.contiguous().view(-1, vocab_size),
                output_lines_src.view(-1).cuda()
            )
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
           
            #TODO - Add early stopping; hyper parameter tuning
            print("GPU memory consumption for epoch" + str(i) + " " + str(torch.cuda.memory_allocated()))
            print("Epoch:" + str(i) + "Minibatch:" + str(j) + "Loss:" + str(loss.item()))
        
        torch.save(
            model.state_dict(),
            open(os.path.join(
                save_dir,
                "autoEncoder_preTrained" + str(i) + '.model'), 'wb'
            )
        )
elif tr_tt == 'test':
    if model_name == 'bert' or model_name == 'scibert':
        tokens_tensor, segments_tensor = model.tokenize_tensor(device)
        sentence_embedding = []
        doc_name = []
        for i, j in enumerate(data):
            t = model.get_sentenceEmbedding(tokens_tensor[i], segments_tensor[i], len(data[i]))
            if t is not None:
                doc_name.append(doc_names[i])
                sentence_embedding.append(t.cpu())
        print("Reviewer count:", len(set(df_filtered['reviewer'])))
        reviewer_trg, submitter_trg, reviewer_paper_trg  = build_reviewerAndsubmitter_embeddings(doc_name, sentence_embedding, df_filtered)
        for i in submitter_trg:
            submitter_trg[i] = submitter_trg[i][0]
    else:    
        model.load_state_dict(torch.load(os.path.join(save_dir, 'autoEncoder_preTrained300.model'))) 
        print(model)
        with torch.no_grad():
            model.eval()
            test_data = []
            test_data = src
            
            for i in range(0, len(src)):
                input_lines_src, output_lines_src, lens_src, mask_src = get_batch(src, word2idx, i,
                       1, max_len)
                print("Total:", len(src), " Current:", i , "Tracker On")
                if i > 911:
                    print(input_lines_src)
                decoder_logit, hid_state, trg = model(input_lines_src)
                prob_2d, word_probs  = decode(model, decoder_logit, vocab_size)
                #word_probs = word_probs.data.cpu().numpy().argmax(axis=-1)
                output_lines_src = output_lines_src.data.cpu().numpy()
                print("GPU memory consumption" + str(i) + " " + str(torch.cuda.memory_allocated()))

                hidden_embed.append(hid_state.data.cpu().numpy())
                print("Hidden embeddings list size:", sys.getsizeof(hidden_embed))
                #encoder_op.append(decoder_logit[0].data.cpu().numpy())
                doc_embedding.append(trg[0].data.cpu().numpy())
                print("Doc embedding list size:", sys.getsizeof(doc_embedding))
            #for i in encoder_op:
                #    array = i/np.linalg.norm(encoder_op)
                #    docVector.append(array) 
                #print(doc_embedding)
                #print(docVector)
        
        docVector = []
        for i in range(len(doc_embedding)):
            docVector.append(doc_embedding[i][0])

        hid = []
        for i in range(len(hidden_embed)):
            hid.append(hidden_embed[i][0])
            
        ##Document based cosine similarity##
        doc_name = doc_names

        print("Reviewer count:", len(set(df_filtered['reviewer'])))
        reviewer_hid, submitter_hid, reviewer_paper_hid = build_reviewerAndsubmitter_embeddings(doc_name, hid, df_filtered)
        reviewer_trg, submitter_trg, reviewer_paper_trg  = build_reviewerAndsubmitter_embeddings(doc_name, docVector, df_filtered)
        del hid, docVector

    print("Inference complete! Next phase: Evaluation")
    #Evaluation - Clustering paper based
    if eval_method == 'all' or eval_method == 'cluster':
        if "AutoEncoder" in model_name:
            affinity_hid_paper = seq_cosineSimilarity_paper(reviewer_paper_hid, submitter_hid)
            with open(os.path.join(save_dir, "hidden_emb.txt"), "w") as hid_out:
                for each in affinity_hid_paper.keys():
                    for i in affinity_hid_paper[each]:
                        hid_out.write(str(each)+" "+str(i)+" "+ str(np.mean(affinity_hid_paper[each][i]))+"\n")
            hid_out.close()
            
        if model_name == 'bert' or model_name == 'scibert':
            for i in submitter_trg:
                submitter_trg[i] = submitter_trg[i][0]

        affinity_trg_paper = seq_cosineSimilarity_paper(reviewer_paper_trg, submitter_trg)
        with open(os.path.join(save_dir, "trg_emb.txt"), "w") as trg_out:
            for each in affinity_trg_paper.keys():
                for i in affinity_trg_paper[each]:
                    trg_out.write(str(each)+" "+str(i)+" "+ str(np.mean(affinity_hid_paper[each][i]))+"\n")
        trg_out.close()
        
    #Vector extrema for building individual reviewer vectors
    if vector == 'extrema':
        reviewer_trg = vectorExtrema(reviewer_trg)
        if "AutoEncoder" in model_name:
            reviewer_hid = vectorExtrema(reviewer_hid)
        elif model_name == 'bert' or model_name =='scibert':
            for i in reviewer_trg:
                reviewer_trg[i] = reviewer_trg[i][0]

    if vector =='mean':
        reviewer_trg = vectorMean(reviewer_trg)
        if "AutoEncoder" in model_name:
            reviewer_hid = vectorMean(reviewer_hid)
        elif model_name == 'bert' or model_name =='scibert':
            for i in reviewer_trg:
                reviewer_trg[i] = reviewer_trg[i][0]


    ##Evaluation reviewer based
    if eval_method == 'all' or eval_method == 'cluster':
        #Affinity scores
        if "AutoEncoder" in model_name:
            affinity_hid = seq_cosineSimilarity(reviewer_hid, submitter_hid)
            #Sort individual scores
            for each in affinity_hid.keys():
                sorted_x = sorted(affinity_hid[each].items(), key=operator.itemgetter(1), reverse=True)
                affinity_hid[each] = sorted_x

            #Write to file
            with open(os.path.join(save_dir, "hidden.txt"), "w") as hid_out:
                for each in affinity_hid.keys():
                    for i in affinity_hid[each]:
                        if i[1].numpy() > 0.8:
                            hid_out.write(str(each)+" "+str(i[0])+" "+ str(i[1].numpy())+"\n")
            hid_out.close()
            
        affinity_trg = seq_cosineSimilarity(reviewer_trg, submitter_trg)
        for each in affinity_trg.keys():
            sorted_x = sorted(affinity_trg[each].items(), key=operator.itemgetter(1), reverse=True)
            affinity_trg[each] = sorted_x

        with open(os.path.join(save_dir, "trg.txt"), "w") as trg_out:
            for each in affinity_trg.keys():
                for i in affinity_trg[each]:
                    if i[1].numpy() > 0.8:
                        trg_out.write(str(each)+" "+str(i[0])+" "+ str(i[1].numpy())+"\n")
        trg_out.close()
    if (eval_method == 'regression') or (eval_method == 'all'):
        #Supervised learning against bids_file
        epochs = 1000
        batch_size = 32
       
        #for hidden_states
        if "AutoEncoder" in model_name:
            data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data_LR(submitter_hid, reviewer_hid, df_filtered, True)
            train_ratio = int(0.8*len(data_sub))
            test_ratio = len(data_sub) - train_ratio

            train_sub = data_sub[:train_ratio]
            test_sub = data_sub[train_ratio:]

            train_rev = data_rev[:train_ratio]
            test_rev = data_rev[train_ratio:]

            y_train = data_y[:train_ratio]
            y_test = data_y[train_ratio:]
                
            lr_model = Match_LR(submitter_emb_dim=rnn_dim,
                            reviewer_emb_dim=rnn_dim,
                            batch_size=batch_size,
                            n_classes=1,)
            lr_model.to(device)
            criterion = nn.MSELoss().cuda()
            optimizer = optim.SGD(lr_model.parameters(), lr=0.01, momentum=0.9)

            print("Length of labels:",len(data_y))
            if len(data_y) > 0:
                train_LR(epochs, lr_model, train_sub, train_rev, y_train, save_dir, criterion, optimizer, batch_size, "hid_LR")

                eval_LR(save_dir, test_sub, test_rev, y_test, criterion, "hid_LR", submitter_ids, reviewer_ids)
            
            del lr_model 
            print("Eval metric for hidden states done!")

        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data_LR(submitter_trg, reviewer_trg, df_filtered)
        train_ratio = int(0.8*len(data_sub))
        test_ratio = len(data_sub) - train_ratio

        train_sub = data_sub[:train_ratio]
        test_sub = data_sub[train_ratio:]

        train_rev = data_rev[:train_ratio]
        test_rev = data_rev[train_ratio:]

        y_train = data_y[:train_ratio]
        y_test = data_y[train_ratio:]
         
        lr_model = Match_LR(submitter_emb_dim=src_dim,
                        reviewer_emb_dim=src_dim,
                        batch_size=batch_size,
                        n_classes=1,)
        lr_model.to(device)
        criterion = nn.MSELoss().cuda()
        optimizer = optim.SGD(lr_model.parameters(), lr=0.005, momentum=0.9)

        print("Length of labels:",len(data_y))
        if len(data_y) > 0:
            train_LR(epochs, lr_model, train_sub, train_rev, y_train, save_dir, criterion, optimizer, batch_size, "trg_LR")

            eval_LR(save_dir, test_sub, test_rev, y_test, criterion, "trg_LR", submitter_ids, reviewer_ids)
        del lr_model

    if (eval_method == 'classification') or (eval_method == 'all'):
        #for hidden_states - valid only for seq2seq
        if "AutoEncoder" in model_name:
            data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(submitter_hid, reviewer_hid, df_filtered)
            train_ratio = int(0.8*len(data_sub))
            test_ratio = len(data_sub) - train_ratio

            train_sub = data_sub[:train_ratio]
            test_sub = data_sub[train_ratio:]

            train_rev = data_rev[:train_ratio]
            test_rev = data_rev[train_ratio:]

            y_train = data_y[:train_ratio]
            y_test = data_y[train_ratio:]
             
           
            cls_model = Match_Classify(submitter_emb_dim=rnn_dim,
                            reviewer_emb_dim=rnn_dim,
                            batch_size=batch_size,
                            n_classes=4,)
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = optim.SGD(cls_model.parameters(), lr=0.01, momentum=0.9)
            ''' 
            if torch.cuda.device_count() > 0:
                print("Use", torch.cuda.device_count(), "GPUs")
                seq_model = nn.DataParallel(seq_model)
            '''
            cls_model.to(device)
            
            if len(data_y) > 0:
                train_classification(epochs, cls_model, train_sub, train_rev, y_train, save_dir, criterion, optimizer, batch_size, "hid")

            eval_classification(save_dir, test_sub, test_rev, y_test, criterion, "hid", submitter_ids, reviewer_ids)
            print("Eval metric for hidden states done!") 
            del cls_model

        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(submitter_trg, reviewer_trg, df_filtered)
        cls_model = Match_Classify(submitter_emb_dim=src_dim,
                        reviewer_emb_dim=src_dim,
                        batch_size=batch_size,
                        n_classes=4,)

        train_ratio = int(0.8*len(data_sub))
        test_ratio = len(data_sub) - train_ratio

        train_sub = data_sub[:train_ratio]
        test_sub = data_sub[train_ratio:]

        train_rev = data_rev[:train_ratio]
        test_rev = data_rev[train_ratio:]

        y_train = data_y[:train_ratio]
        y_test = data_y[train_ratio:]
        
        cls_model.to(device)
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(cls_model.parameters(), lr=0.005, momentum=0.9)
        
        print("Length of labels:",len(data_y))
        if len(data_y) > 0:
            train_classification(epochs, cls_model, train_sub, train_rev, y_train, save_dir, criterion, optimizer, batch_size, "trg")

            eval_classification(save_dir, test_sub, test_rev, y_test, criterion, "trg", submitter_ids, reviewer_ids)
        

