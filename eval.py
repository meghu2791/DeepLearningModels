import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
from code.utils import get_batch_eval

class Match_Classify(nn.Module):
    def __init__(self,
                 submitter_emb_dim,
                 reviewer_emb_dim,
                 batch_size,
                 n_classes,):
        super(Match_Classify, self).__init__()
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim = reviewer_emb_dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.weights_add = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True).cuda()
        self.weights_diff = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True).cuda()
        self.weights_multi = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True).cuda()
        
        self.fc2 = nn.Linear(self.reviewer_emb_dim, self.reviewer_emb_dim)
        self.output = nn.Linear(128, n_classes)
        self.combined = nn.Linear(self.submitter_emb_dim, 128)
    
        self.init_weights()
        
    def init_weights(self):
        initrange = 4.0
        #self.weights.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.fill_(0)
        self.weights_add.data.uniform_(-initrange, initrange)
        self.weights_diff.data.uniform_(-initrange, initrange)
        self.weights_multi.data.uniform_(-initrange, initrange)
        
    def forward(self, submitter_emb, reviewer_emb):
        #submitter_f = self.fc_submitter(submitter_emb)
        #reviewer_f = self.fc_reviewer(reviewer_emb)
        add = submitter_emb + self.fc2(reviewer_emb)
        diff = submitter_emb - self.fc2(reviewer_emb)
        multi = submitter_emb * (self.fc2(reviewer_emb))
        
        combo = self.combined(nn.Tanh()(self.weights_add * add) + nn.Tanh()(self.weights_diff * diff) + nn.Tanh()(self.weights_multi * multi))
        op = F.softmax(self.output(combo))
        return op 

class Match_LR(nn.Module):
    def __init__(self,
                 submitter_emb_dim,
                 reviewer_emb_dim,
                 batch_size,
                 n_classes,):
        super(Match_LR, self).__init__()
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim = reviewer_emb_dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.weights_add = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True).cuda()
        self.weights_diff = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True).cuda()
        self.weights_multi = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True).cuda()
        
        self.fc2 = nn.Linear(self.reviewer_emb_dim, self.reviewer_emb_dim)
        self.output = nn.Linear(128, 1)
        self.combined = nn.Linear(self.submitter_emb_dim, 128)
    
        self.init_weights()
        
    def init_weights(self):
        initrange = 4.0
        #self.weights.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.fill_(0)
        self.weights_add.data.uniform_(-initrange, initrange)
        self.weights_diff.data.uniform_(-initrange, initrange)
        self.weights_multi.data.uniform_(-initrange, initrange)
        
    def forward(self, submitter_emb, reviewer_emb):
        #submitter_f = self.fc_submitter(submitter_emb)
        #reviewer_f = self.fc_reviewer(reviewer_emb)
        add = submitter_emb + self.fc2(reviewer_emb)
        diff = submitter_emb - self.fc2(reviewer_emb)
        multi = submitter_emb * (self.fc2(reviewer_emb))
         
        combo = self.combined(nn.Tanh()(self.weights_add * add) + nn.Tanh()(self.weights_diff * diff) + nn.Tanh()(self.weights_multi * multi))
        
        op = 3*torch.sigmoid(self.output(combo))
        return op 

def prepare_data(submitter, reviewer, df, gpu_flag=False):
    train_data_sub = []
    train_data_rev = []
    submit = submitter.keys()
    submitter_ids = []
    reviewer_ids = []
    rev = reviewer.keys()
    labels = []
    for i in range(len(df)):
        if str(df.iloc[i]['paper_id']) in submit and df.iloc[i]['reviewer'] in reviewer:
            train_data_sub.append(torch.tensor(submitter[str(df.iloc[i]['paper_id'])],requires_grad=True).cuda())
            train_data_rev.append(torch.tensor(reviewer[df.iloc[i]['reviewer']], requires_grad=True).cuda())
            idx = int(df.iloc[i]['preference'])
            temp = torch.LongTensor([0, 0, 0, 0]).cuda()
            for i in range(4):
                if i == idx:
                    temp[i] = 1
            labels.append(temp)
            submitter_ids.append(df.iloc[i]['paper_id'])
            reviewer_ids.append(df.iloc[i]['reviewer'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids


def prepare_data_LR(submitter, reviewer, df, gpu_flag=False):
    train_data_sub = []
    train_data_rev = []
    submit = submitter.keys()
    rev = reviewer.keys()
    submitter_ids = []
    reviewer_ids = []
    labels = []
    for i in range(len(df)):
        if str(df.iloc[i]['paper_id']) in submit and df.iloc[i]['reviewer'] in reviewer:
            train_data_sub.append(torch.tensor(submitter[str(df.iloc[i]['paper_id'])],requires_grad=True).cuda())
            train_data_rev.append(torch.tensor(reviewer[df.iloc[i]['reviewer']], requires_grad=True).cuda())
            idx = int(df.iloc[i]['preference'])
            temp = torch.FloatTensor([idx]).cuda()
            #labels.append(torch.LongTensor([idx]).cuda())
            labels.append(temp)
            submitter_ids.append(df.iloc[i]['paper_id'])
            reviewer_ids.append(df.iloc[i]['reviewer'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids


def train_LR(epochs, model, train_data_sub, train_data_rev, labels, save_dir, criterion, optimizer, batch_size, m_name):
    losses= []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(0, len(labels), batch_size):
            tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, batch_size) 
            optimizer.zero_grad()
            prediction = model(tr_sub, tr_rev)
            print(prediction.view(1,len(tr_sub)), y.view(1,len(tr_sub)))
            loss = criterion(prediction.view(1,len(tr_sub)), y.view(1,len(tr_sub)))
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/len(y))
        print("Epoch:", e_num, " Loss:", losses[-1])
        print("GPU memory consumption for epoch" + str(e_num) + " " + str(torch.cuda.memory_allocated()))
        torch.save(model, os.path.join(save_dir, str(m_name+"_lr.model")))
    print("Model training completed!!")

def train_d2v_LR(epochs, model, train_data_sub, train_data_rev, labels, save_dir, criterion, optimizer, batch_size):
    losses= []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(0, len(labels), batch_size):
            tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, batch_size) 
            optimizer.zero_grad()
            prediction = model(tr_sub, tr_rev)
            print(prediction.view(1,len(tr_sub)), y.view(1, len(tr_sub)))
            loss = criterion(prediction.view(1,len(tr_sub)), y.view(1, len(tr_sub)))
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/len(y))
        print("Epoch:", e_num, " Loss:", losses[-1])
        print("GPU memory consumption for epoch" + str(e_num) + " " + str(torch.cuda.memory_allocated()))
        torch.save(model, os.path.join(save_dir, "d2v_bid_lr.model"))
    print("Model training completed!!")
 
def train_bert_LR(epochs, model, train_data_sub, train_data_rev, labels, save_dir, criterion, optimizer, m, batch_size):
    losses= []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(0, len(labels), batch_size):
            tr_sub, tr_rev, y = get_batch_eval(train_data_sub, train_data_rev, labels, i, batch_size) 
            optimizer.zero_grad()
            prediction = model(tr_sub, tr_rev)
            #print(prediction, y)
            print(prediction.view(1,len(tr_sub)), y.view(1, len(tr_sub)))
            loss = criterion(prediction.view(1,len(tr_sub)), y.view(1, len(tr_sub)))
            loss_ep += loss.item()
            print("Batch_number:", i, "Batch loss:", loss.item())
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/len(y))
        print("Epoch:", e_num, " Loss:", losses[-1])
        print("GPU memory consumption for epoch" + str(e_num) + " " + str(torch.cuda.memory_allocated()))
        torch.save(model, os.path.join(save_dir, "bid_lr.model"))
    print("Model training completed!!")
    ''' 
    losses = []
    for e_num in range(epochs):
        loss_ep = 0
        for i in range(int(0.8*len(labels))):
            optimizer.zero_grad()
            prediction = model(train_data_sub[i], train_data_rev[i])
            loss = criterion(prediction, labels[i].argmax(dim=1))   # must be (1. nn output, 2. target)
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()
        losses.append(loss_ep/batch_size)
        print("Epoch:", e_num, " Loss:", losses[-1])
    '''
def eval_LR(path, train_data_sub, train_data_rev, labels, criterion, m_name, submitter_ids, reviewer_ids):
    model = torch.load(os.path.join(path, str(m_name+"_lr.model")))
    with torch.no_grad():
        model.eval()
        class_label = 0
        trg_label = 0
        correct = 0
        wrong = 0
        loss_test = 0
        loss_inttest = 0
        with open(os.path.join(path,str("test_results"+m_name+".txt")), "w") as out:
            for i in range(len(labels)):
                prediction = model(train_data_sub[i], train_data_rev[i])
                print(prediction, labels[i].unsqueeze(0))
                loss = criterion(prediction, labels[i].unsqueeze(0))   # must be (1. nn output, 2. target)
                loss_test += loss.item()
                loss_inttest += int(loss.item())
                out.write(str(submitter_ids[i]) + " " + str(reviewer_ids[i]) + " " + str(prediction.data.cpu()) + " " + str(labels[i].unsqueeze(0).data.cpu()))
            print(" Test Loss:", loss_test/len(labels))
            out.write(" Test Loss:" + str(loss_test/len(labels)) +  " " + str(loss_inttest/len(labels)))
        out.close()

def eval_bert_LR(path, train_data_sub, train_data_rev, labels, m, criterion, submitter_ids, reviewer_ids):
    model = torch.load(os.path.join(path, "bid_lr.model"))
    with torch.no_grad():
        model.eval()
        class_label = 0
        trg_label = 0
        correct = 0
        wrong = 0
        loss_test = 0
        loss_inttest = 0
        with open(os.path.join(path,"test_results.txt"), "w") as out:
            for i in range(len(labels)):
                prediction = model(train_data_sub[i], train_data_rev[i])
                print(prediction, labels[i].unsqueeze(0))
                loss = criterion(prediction, labels[i].unsqueeze(0))   # must be (1. nn output, 2. target)
                loss_test += loss.item()
                loss_inttest += int(loss.item())
                out.write(str(submitter_ids[i]) + " " + str(reviewer_ids[i]) + " " + str(prediction.data.cpu()) + " " + str(labels[i].unsqueeze(0).data.cpu()))
            print(" Test Loss:", loss_test/len(labels), loss_inttest/len(labels))
            out.write(" Test Loss:" + str(loss_test/len(labels)) + " " + str(loss_inttest/len(labels)))
        out.close()

def eval_d2v_LR(path, train_data_sub, train_data_rev, labels, criterion):
    model = torch.load(os.path.join(path, "d2v_bid_lr.model"))
    with torch.no_grad():
        model.eval()
        class_label = 0
        trg_label = 0
        correct = 0
        wrong = 0
        loss_test = 0
        loss_inttest = 0
        with open(os.path.join(path,"test_results.txt"), "w") as out:
            for i in range(len(labels)):
                prediction = model(train_data_sub[i], train_data_rev[i])
                print(prediction, labels[i].unsqueeze(0))
                loss = criterion(prediction, labels[i].unsqueeze(0))   # must be (1. nn output, 2. target)
                loss_test += loss.item()
                loss_inttest += int(loss.item()) 
                out.write(str(submitter_ids[i]) + " " + str(reviewer_ids[i]) + " " + str(prediction.data.cpu()) + " " + str(labels[i].unsqueeze(0).data.cpu()))
            print(" Test Loss:", loss_test/len(labels))
            out.write(" Test Loss:" + str(loss_test/len(labels)) +  " " + str(loss_inttest/len(labels)))
        out.close()

#Experimental
def eval_bertClassification(model, train_data_sub, train_data_rev, labels):
    with torch.no_grad():
        model.eval()
        class_label = 0
        trg_label = 0
        correct = 0
        wrong = 0
        loss_test = 0
        input_ids = []
        print(train_data_sub[0].size())
        for i in range(len(labels)):
            input_ids.append(train_data_sub[i][0] - train_data_rev[i][0])
            print(torch.sigmoid(input_ids[-1]))
        prediction = model(torch.stack(input_ids,0), labels=labels)
        print(" Test Loss:", prediction[0])
        with open(os.path.join(path,"test_results.txt"), "w") as out:
            out.write(" Test Loss:" + str(prediction[0]))
        out.close()
##

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


def eval_classification(path, train_data_sub, train_data_rev, labels, criterion, m_name, submitter_ids, reviewer_ids):
    model = torch.load(os.path.join(path, str(m_name+".model")))
    with torch.no_grad():
        model.eval()
        class_label = 0
        trg_label = 0
        correct = 0
        wrong = 0
        loss_test = 0
        with open(os.path.join(path,str("test_results"+m_name+".txt")), "w") as out:
            for i in range(len(labels)):
                #print(tr_sub, y)
                prediction = model(train_data_sub[i], train_data_rev[i])
                print(prediction, labels[i].unsqueeze(0).argmax(dim=1))
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

def seq_cosineSimilarity_paper(reviewer, submitter):
    affinity = dict()
    for each in submitter.keys():
        temp = dict()
        for review in reviewer.keys():
            if len(reviewer[review]) == 1:
                cos = nn.CosineSimilarity(dim=1)
                temp[review] = [cos(submitter[each], reviewer[review][0]).numpy()[0]]
            else:
                cos = nn.CosineSimilarity(dim=1)
                if review not in temp:
                    temp[review] = [cos(submitter[each], reviewer[review][0]).numpy()[0]]
                for j in range(1,len(reviewer[review])):
                    temp[review].append(cos(submitter[each], reviewer[review][j]).numpy()[0])
            sorted_scores = sorted(temp[review], key=float, reverse=True)
            temp_review = [] 
            if len(sorted_scores) < 5:
                temp[review] = sorted_scores
            else:
                temp[review] = sorted_scores[:5]
        if each in affinity.keys():
            affinity[each].update(temp)
        else:
            affinity[each] = temp
    return affinity

def seq_cosineSimilarity(reviewer, submitter):
    affinity = dict()
    for each in submitter.keys():
        temp = dict()
        for review in reviewer.keys():
            if review not in temp:
                cos = nn.CosineSimilarity(dim=1)
                temp[review] = cos(submitter[each], reviewer[review])[0]
            else:
                cos = nn.CosineSimilarity(dim=1)
                temp[review].update(cos(submitter[each], reviewer[review])[0])
        
        if each in affinity.keys():
            affinity[each].update(temp)
        else:
            affinity[each] = temp
    return affinity

#from sklearn.neighbors import NearestNeighbors
#nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(papers)
#dist, indices = nbrs.kneighbors(papers)
