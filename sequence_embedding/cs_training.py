# test

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:27:08 2016

@author: YUH015
"""

#!/usr/bin/python
import theano
import datetime
import random
from random import shuffle
import pandas
#from nltk.tokenize import word_tokenize
import csv
import itertools
import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from gru_theano import GRUTheano
from gru_theano_batch2 import GRU_theano_batch2
#from gru_theano_batch import GRU_theano_batch
from gru_theano_l1 import gru_theano_l1
import config

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "128"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "4"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "8"))
NEPOCH = int(os.environ.get("NEPOCH", "2"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5000"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE", "model.out")

print "learning rate = ", LEARNING_RATE
print "vocabulary size = ", VOCABULARY_SIZE
print "embedding dimension = ", EMBEDDING_DIM
print "hidden dimension = ", HIDDEN_DIM
print "number of training epoch = ", NEPOCH
print "batch size = ", config.batch_size

batch_size = config.batch_size

ts = time.time()

seq_all = []
freq = []
dict_page = {}
#os.chdir('C:\Users\YUH015\Desktop\code\cs_pathing')

with open('./small_clean.tsv', 'rb') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
#        print type(row)
        rows = row[0].split("\t")
#        print rows
        if len(rows) <= 4:
            continue
#        print "good line"
        seq_all.append(rows[3:])
        freq.append(int(rows[1]))
        for page in rows[3:]:
#            print page
            if page not in dict_page:
                dict_page[page] = int(rows[1])
            else:
                dict_page[page] = dict_page[page] + int(rows[1])

dict_page_index = {}
count = 0
for e in sorted(dict_page, key=dict_page.get, reverse=True):
    dict_page_index[e] = count
    count = count + 1

print "number of sequences: ", len(seq_all)
print "number of pages: ", len(dict_page)
#print len(dict_page_index)

#we assume data sequence has the right vocabulary
#moreover, it needs to have more than one URL/word to enable running
data_sequence = np.asarray([[min(dict_page_index[w],VOCABULARY_SIZE-1) for w in sent] for sent in seq_all if len(sent) >= 2])
# get length of each sequence
data_len = [len(e) for e in data_sequence]
# get max length

# we now need to reorganize data_sequence into batches
# since we have batch size, we need to ensure that each batch are equally long
# and that if a batch is incomplete, pad it randomly
# we know the random patching is imperfect, but think it will not have any impact
# on training
maxl = np.max(data_len)
random.seed(12345)
gx = []
gy = []
gx_start = []
gx_batch_len = []
pos_start = 0
for i in np.arange(2,maxl+1):
    sl = [k for k,j in enumerate(data_len) if j == i]
    if len(sl) == 0:
        continue
#    print sl
    shuffle(sl)
    #padding
    # if we are missing a part of batch size, we need to add them back in
    # we don't want to cut

    cut = (len(sl)) % batch_size
    if cut > 0:
        sl2 = [random.choice(sl) for _ in range(batch_size-cut)]
        sl = sl + sl2
    #now sl has all the right indices of length i
    #we just need to generate gx and gy
    for s in sl:
        gx = gx + data_sequence[s][:-1]
        gy = gy + data_sequence[s][1:]
#        if i < 3 and len(gx) != len(gy):
#            print "error: ", len(data_sequence[s][:-1]), len(data_sequence[s][1:])
#            print data_sequence[s]
#            print gx
#            print gy

    # we have so far got len(sl) sequences with length i
    # need to add them to indices
    gx_start = gx_start + range(pos_start, pos_start+(i-1)*len(sl), (i-1)*batch_size)
    gx_batch_len = gx_batch_len + [i-1]*int(len(sl)/batch_size)
    pos_start = pos_start + (i-1)*len(sl)
#    for n in np.arange(0, int(len(sl)/batch_size)):
#        gx_start.append(pos_start)
#        gx_batch_len.append(i)

#print sum([len(s) for s in data_sequence])
print len(gx)
print len(gy)
#print len(gx_start)
#print len(gx_batch_len)
#sys.exit()

#print "how does it look like:"
#for bstart,blen in zip(gx_start, gx_batch_len):
    #for i in [0,4,8]:
#    if bstart >= 158000:
#        print "start iterations for starting position: ", bstart, blen
#    model2.batch_step(bstart, blen, LEARNING_RATE)

gx = np.asarray(gx)
gy = np.asarray(gy)
print "data prep completed"
#sys.exit()

t1 = time.time()
model2 = GRU_theano_batch2(gx, gy, VOCABULARY_SIZE, hidden_dim=16, bptt_truncate=-1)
#model2 = GRU_theano_batch(VOCABULARY_SIZE, hidden_dim=16, bptt_truncate=-1)
print "batch model construction time: ", "%.3f" % (time.time()-t1), " seconds"

#sys.exit()

# training start
print "*****\ntraining start\n*****"
t1 = time.time()
for epoch in np.arange(0,NEPOCH):
    t1 = time.time()
    # for each epoch, generate batch sequences by shuffling and padding
    print "Running epoch ", epoch
#    if epoch == 0:
#        print "max, min, and average length of inputs:  ", np.max([len(e) for e in x_train]), np.min([len(e) for e in x_train]), np.mean([len(e) for e in x_train])
#        print "# of batches in training data: %d" % int(len(x_train) / batch_size)

    for bstart,blen in zip(gx_start, gx_batch_len):
    #for i in [0,4,8]:
#        print "start iterations for starting position: ", bstart
        model2.batch_step(bstart, blen, LEARNING_RATE)

    # note: loss estimation should ONLY happen AFTER all training is done
#    loss = 0.0
#    for bstart,blen in zip(gx_start, gx_batch_len):
#        loss += model2.optimization_error(bstart, blen)
#    loss = loss*batch_size/len(x_train)
    loss = 0.0
    accuracy = 0.0
    total_len = 0.0
    for e in data_sequence:
        if len(e) >= 2:
            [_, pred_class, sce] = model2.example_prediction(e[:-1],e[1:])
            loss += sce
            accuracy += np.sum(np.allclose(e[1:], pred_class))
            total_len += len(pred_class)
    loss = loss/len(data_sequence)
    print "average sample loss for epoch ", epoch, " is %.3f" % loss
    print "accuracy = %.4f" % (1.0*accuracy/total_len), " out of ", total_len, " samples"
    print "epoch training time = ", "%.3f seconds" % (time.time()-t1)

print "*****\ntraining end\n*****"

print "total program running time time = %.3f seconds" % (time.time()-ts)
sys.exit()

def sgd_callback(model, num_examples_seen):
    dt = datetime.now().isoformat()
    loss = model.calculate_loss(x_train[:10000], y_train[:10000])
    print("--------------------------------------------------")
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("Loss: %f" % loss)
    # don't do anything yet
#    generate_sentences(model, 10, index_to_word, word_to_index)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    ts = ts.replace(" ","")
    outfile = "./model/cs_gru_%s.dat" % ts
#    save_model_parameters_theano(model, outfile)
    print("\n")
    sys.stdout.flush()

for epoch in range(NEPOCH):
#for epoch in range(5):
    t1 = time.time()
    train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
                   callback_every=PRINT_EVERY, callback=sgd_callback)
    t2 = time.time()
    print "SGD Step time: %.3f seconds" % (t2 - t1)

#x_test = np.asarray([sent[:-2] for sent in d2])


x_test = [sent[:-2] for sent in data_sequence]


np.random.seed(12345)
count = 0
correct = 0
print "processing ", len(data_sequence), " elements"

for s in data_sequence:
    if type(s) != type([1,2,3]) or len(s) <= 5:
        continue
    length = len(s)
    cutoff = np.random.randint(1,length-2)
    x_test = s[0:cutoff]
    pred = model.predict(x_test).argmax()
    actual = s[cutoff+1]
    count = count + 1
    if pred == actual:
        correct = correct + 1

print "% accuracy = ", correct*1.0/count

#outfile = "./output/output%s.txt" % (time.strftime("%m.%d.%y %H:%M", time.localtime()))
#outfile = outfile.replace(" ","")
#target = open(outfile, 'w')

#target.write("model accuracy %f\n" % (correct*1.0/count) )
print "total time %.3f seconds" % (time.time()-ts)
#target.close()


print "done"
print "done"
print "done"
print " "
print " "
print " "
