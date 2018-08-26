# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:27:08 2016

@author: YUH015
"""

#!/usr/bin/python
import math
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
#from utils import *
from datetime import datetime
#import pydot
from gru_theano import GRUTheano
from gru_theano_batch2 import GRU_theano_batch2
#from gru_theano_batch import GRU_theano_batch
from gru_theano_l1 import gru_theano_l1
import config
#import cProfile, pstats, StringIO

from itertools import chain

np.set_printoptions(threshold=np.inf)
csv.field_size_limit(20000000)

#theano.config.compute_test_value = 'warn'

def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value())
    print "Saved model parameters to %s." % outfile


LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "1024"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "32"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "1"))
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

page_length_by_count = [0]*4096
with open('./top400k.clean.tsv', 'rb') as csvfile:
#with open('/dev/shm/results.clean.tsv', 'rb') as csvfile:
#with open('/dev/shm/results.clean.tsv', 'rb') as csvfile:
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
        page_length_by_count[len(rows)-3] = page_length_by_count[len(rows)-3] + 1
        for page in rows[3:]:
#            print page
            if page not in dict_page:
                dict_page[page] = int(rows[1])
            else:
                dict_page[page] = dict_page[page] + int(rows[1])

print "print raw sequence leng distribution:"

#batch_size = 128
max_seq_length = 2048

dict_page_index = {}
count = 0
for e in sorted(dict_page, key=dict_page.get, reverse=True):
    dict_page_index[e] = count
    count = count + 1

print "number of sequences: ", len(seq_all)
print "number of unique pages: ", len(dict_page)
#print len(dict_page_index)
#we assume data sequence has the right vocabulary
#moreover, it needs to have more than one URL/word to enable running
data_sequence = np.asarray([[min(dict_page_index[w],VOCABULARY_SIZE-1) for w in sent] for sent in seq_all if len(sent) >= 2])
# get length of each sequence
#data_len = [len(e) for e in data_sequence]

#base is the set of queue lengths that have enough data
#base = set()
#for l in np.arange(len(page_length_by_count)):
#    if page_length_by_count[l]
#    if page_length_by_count[l] >= batch_size*0.9:
#        base.add(l)
#print "base:"
#print base

acc = 0
threshold = 0.9*batch_size
start = -1
end = -1
list_bundle = []
for i,e in reversed(list(enumerate(page_length_by_count))):
    if e <= 0:
        continue
#    print i,e
    if start < 0:
        start = i
    acc += e
    if acc >= threshold and i <= config.sequence_length_cap+1:
        end = i
#        print "bundle:", start,end,acc
        list_bundle.append((start,end))
        start = -1
        end = -1
        acc = 0

seq_map = dict()
for s,e in list_bundle:
    for i in np.arange(e,s+1):
        seq_map[i] = e
#print seq_map

#at this point we know size of truncation, but we don't know how to cut
#for a sequence of length x and a cut size of c, we will cut it into
# 0/c-1, c/2c-1,2c/3c-1, ...

def cut_long_sequence(seq,cut):
    cuts = np.arange(0,seq,cut)
    cuts = [(i,j) for i,j in zip(cuts,cuts+cut) if j <= seq]
    if seq % cut != 0:
        # need to add a piece
        cuts.append((seq-cut,seq))
    return cuts

#print cut_long_sequence(9,4)

#sys.exit()

# perform truncation
data_sequence_raw = []

word_freq = dict()
for s in data_sequence:
    # perform truncation
    slen = len(s)
    if slen <= 1:
        continue

    for front,back in zip(s[1:],s[:-1]):
        if (front,back) in word_freq:
            word_freq[(front,back)] = word_freq[(front,back)] + 1
        else:
            word_freq[(front,back)] = 1

    target_len = seq_map[slen]

    if target_len == slen:
        data_sequence_raw.append(s)
    else:
        # we need to cut
        for s1,e1 in cut_long_sequence(slen,target_len):
            data_sequence_raw.append(s[s1:e1])
# end truncation
print "frequency table has: ", len(word_freq), " items"
word_freq3 = dict()
for k,v in word_freq.iteritems():
    if v >= 20000:
        print k,v

    if k[0] in word_freq3:
        word_freq3[k[0]] = word_freq3[k[0]] + v
    else:
        word_freq3[k[0]] = v

base_freq = 1.0/VOCABULARY_SIZE
for k in word_freq:
    word_freq[k] = (1.0*word_freq[k])/word_freq3[k[0]]
    if word_freq[k] >= 0.25:
        print k[0],k[1],word_freq[k]
print "mle table has ", len(word_freq3), " entries"

#split data into training, validation, and testing
pct_validation = 0.05
pct_testing = 0.15
np.random.seed(12345678)
choice = np.random.uniform(size=len(data_sequence_raw))

data_sequence_training = [e for e,r in zip(data_sequence_raw,choice) if r > pct_validation+pct_testing]
data_sequence_validation = [e for e,r in zip(data_sequence_raw,choice) if r > pct_testing and r <= pct_validation+pct_testing]
data_sequence_testing = [e for e,r in zip(data_sequence_raw,choice) if r <= pct_testing]

print "after truncation the we have %d sequences" % len(data_sequence_raw)
print len(data_sequence_training), " training"
print len(data_sequence_validation), " validation"
print len(data_sequence_testing), " training"

with open("training.txt","w") as outfile:
    for e in data_sequence_training:
        for e2 in e:
            outfile.write(str(e2)+" ")
        outfile.write("\n")
outfile.close()

with open("validation.txt","w") as outfile:
    for e in data_sequence_validation:
        for e2 in e:
            outfile.write(str(e2)+" ")
        outfile.write("\n")
outfile.close()

with open("testing.txt","w") as outfile:
    for e in data_sequence_testing:
        for e2 in e:
            outfile.write(str(e2)+" ")
        outfile.write("\n")
outfile.close()

sys.exit()

