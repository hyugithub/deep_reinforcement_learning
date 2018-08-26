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
#from datetime import datetime
#import pydot
from gru_theano import GRUTheano
from gru_theano_batch2 import GRU_theano_batch2
#from gru_theano_batch import GRU_theano_batch
from gru_theano_l1 import gru_theano_l1
import config
#import cProfile, pstats, StringIO

from itertools import chain
from time import gmtime, strftime

np.set_printoptions(threshold=np.inf, precision=3)
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

def print_param(param):
    print "%.4f" % np.mean(param), "%.4f" % np.mean(np.abs(param)), "%.4f" % np.max(param), "%.4f" % np.min(param)

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.1"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "1024"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "128"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "256"))
NEPOCH = int(os.environ.get("NEPOCH", "300"))
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

data_sequence_training = []
data_sequence_validation = []
data_sequence_testing = []

with open("top400k.training.txt","r") as infile:
#with open("fake.training.txt","r") as infile:
    lines = csv.reader(infile)
    for l in lines:
        seq = l[0].split()
        data_sequence_training.append(map(int, seq))

with open("top400k.validation.txt","r") as infile:
#with open("fake.training.txt","r") as infile:
    lines = csv.reader(infile)
    for l in lines:
        seq = l[0].split()
        data_sequence_validation.append(map(int, seq))

with open("top400k.testing.txt","r") as infile:
    lines = csv.reader(infile)
    for l in lines:
        seq = l[0].split()
        data_sequence_testing.append(map(int, seq))

data_len = [len(e) for e in data_sequence_training]

print "total number of urls: ", np.sum(data_len)
# get max length
maxl = np.max(data_len)

sequence_block = dict()
for i in np.arange(len(data_sequence_training)):
    slen = len(data_sequence_training[i])
    if slen in sequence_block:
        sequence_block[slen].append(i)
    else:
        sequence_block[slen] = [i]


random.seed(12345)
gx = []
gy = []
gx_start = []
gx_batch_len = []
pos_start = 0
#print "start the most inefficient part of data prep"

for i in np.arange(2,maxl+1):
#for i in np.arange(2,50):
    sl = []
    if i in sequence_block:
        sl = sequence_block[i]
    if len(sl) == 0:
        continue

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

    gx = gx + [item for s in sl for item in data_sequence_training[s][:-1] ]
    gy = gy + [item for s in sl for item in data_sequence_training[s][1:] ]

    # we have so far got len(sl) sequences with length i
    # need to add them to indices
    gx_start = gx_start + range(pos_start, pos_start+(i-1)*len(sl), (i-1)*batch_size)
    gx_batch_len = gx_batch_len + [i-1]*int(len(sl)/batch_size)
    pos_start = pos_start + (i-1)*len(sl)

print "number of batch blocks: ", len(gx)
#print len(gy)
#print len(gx_start)
#print len(gx_batch_len)
#sys.exit()

gx = np.asarray(gx)
gy = np.asarray(gy)
print "data prep completed"
t2 = time.time()
print "file read and step 1 data prep takes %.3f seconds" % (t2-ts)

np.random.seed(54321)
t1 = time.time()
#model2 = GRU_theano_batch2(gx, gy, VOCABULARY_SIZE, hidden_dim=16, bptt_truncate=-1)
model2 = gru_theano_l1(gx, gy, VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
#model2 = GRU_theano_batch(VOCABULARY_SIZE, hidden_dim=16, bptt_truncate=-1)
print "batch model construction time: ", "%.3f" % (time.time()-t1), " seconds"
#model2.batch_step(gx_start[0], gx_batch_len[0], LEARNING_RATE)
#sys.exit()
print "print batch step start"
#theano.printing.debugprint(model2.batch_step)
#print "try pprint"
#theano.printing.pprint(model2.cost)
#print "try pprint end"
#theano.printing.pydotprint(model2.batch_step, outfile="./batch_step.png", var_with_name_simple=True)

print "print batch step end"

#print temp[0], temp[1]
#print "test consistency between optimization error and cross entropy"


# training start
print "*****\ntraining start\n*****"
t1 = time.time()

best_loss = 1e20
for epoch in np.arange(0,NEPOCH):
    t1 = time.time()
    # for each epoch, generate batch sequences by shuffling and padding
    print "Running epoch ", epoch
    count = 0
    for bstart,blen in zip(gx_start, gx_batch_len):
        #print "count = ", count, "bstart,blen=", bstart,blen
        model2.batch_step(bstart, blen, LEARNING_RATE)
        
        if 0:            
            print "count = ", count, "bstart,blen=", bstart,blen
            print "print derivatives:"
            dE, dU, dW, db, dV, dc = model2.bptt(bstart, blen)
            param = np.asarray(dE)
            print_param(param)
            param = np.asarray(dU)
            print_param(param)
            param = np.asarray(dW)
            print_param(param)
            param = np.asarray(dV)
            print_param(param)
            param = np.asarray(db)
            print_param(param)
            param = np.asarray(dc)
            print_param(param)

            print "print parameters:"
            param = model2.E.get_value()
            print_param(param)
            param = model2.U.get_value()
            print_param(param)
            param = model2.W.get_value()
            print_param(param)
            param = model2.V.get_value()
            print_param(param)
            param = model2.b.get_value()
            print_param(param)
            param = model2.c.get_value()
            print_param(param)

            temp_param = 0.0
            temp_param += np.sum(np.power(model2.E.get_value(),2))
            temp_param += np.sum(np.power(model2.U.get_value(),2))
            temp_param += np.sum(np.power(model2.V.get_value(),2))
            temp_param += np.sum(np.power(model2.W.get_value(),2))
            temp_param += np.sum(np.power(model2.b.get_value(),2))
            temp_param += np.sum(np.power(model2.c.get_value(),2))
            if np.isnan(temp_param):
                print "nan found at count", count, bstart, blen
                print "stop"
                sys.exit()

#        if count >= 3:
#            sys.exit()
        count += 1

    # monitor validation dataset performance
    if epoch % 50 == 0:
        loss = 0.0
        accuracy = 0.0
        total_len = 0.0
        count = 0
        set_correct = []
        for e in data_sequence_validation:
            count = count + 1
            if len(e) >= 2 and 1:
                [_, pred_class, sce] = model2.example_prediction(e[:-1],e[1:])
                loss += sce
                #accuracy += np.sum(np.allclose(e[1:], pred_class))
                accuracy += sum(1 for (l,r) in zip(e[1:],pred_class) if np.abs(l-r) <= 1e-6)
                total_len += len(pred_class)
#               for e2 in e[1:]:
#                   set_correct.append(e2)

        loss = (1.0*loss)/len(data_sequence_validation)
        print "average sample loss for epoch ", epoch, " is %.3f" % loss
        print type(accuracy),type(total_len)
        print "accuracy = %.3f" % (1.0*accuracy/total_len)
        print " out of ", total_len, " samples"
        print "epoch training time = ", "%.3f seconds" % (time.time()-t1)
#        print "correct symbol predictions are: ", set_correct

        if loss < best_loss:
            save_model_parameters_theano(model2, "./output_param.npz")
            best_loss = loss

        fname = ("./log/epoch."+str(epoch)+".txt").replace(" ","")
        with open(fname,"w") as f2:
            f2.write("writing epoch ")
            f2.write(str(epoch))
            f2.write(" at time ")
            f2.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            f2.write(" local time ")
            f2.write(str(datetime.datetime.now()))
            f2.write("\n")
            f2.write("average sample loss for epoch ")
            f2.write(str(epoch))
            f2.write(" is %.3f" % loss)
#        f2.write(" accuracy = %.4f" % (1.0*accuracy/total_len))
            f2.close()


print "*****\ntraining end\n*****"

if 0:
    count = 0
    for e in data_sequence_validation:
        count = count+1
        [tout, pred, sce] = model2.example_prediction(e[:-1],e[1:])
        if np.isnan(sce):
            print "error for debugging ", count
            sys.exit()


print "testing scalar forward softmax:"
tout = model2.sample_scalar_forward(data_sequence_validation[0][:-1])
print tout.shape
print_param(tout)
print np.sum(tout, axis=(1,2))
print "testing end"
print "\n\ntest gradient"
#print "before perturbation"
e = data_sequence_validation[0]
[tout, pred, sce] = model2.example_prediction(e[:-1],e[1:])
print "entire sequence: ", e
print "prediction: ", pred
print "sample prediction statistics:"
print tout.shape
#print tout[0,0,0], tout[0,0,1]
#print -np.log(tout[0,0,0])
print np.sum(tout, axis=(1,2))
print_param(tout)
#print tout
print "sample cross entropy = ", sce, "equivalent cross entropy error in batch = ", sce * batch_size
#temp_param = 0.0
#temp_param += np.sum(np.power(model2.E.get_value(),2))
#temp_param += np.sum(np.power(model2.U.get_value(),2))
#temp_param += np.sum(np.power(model2.V.get_value(),2))
#temp_param += np.sum(np.power(model2.W.get_value(),2))
#temp_param += np.sum(np.power(model2.b.get_value(),2))
#temp_param += np.sum(np.power(model2.c.get_value(),2))

print "batch prediction statistics for one sample:"
print "gx_start = ", gx_start[0], " batch len = ", gx_batch_len[0]
[c0, bout3] = model2.optimization_error(gx_start[0], gx_batch_len[0])
print "print bout3 shape: ", bout3.shape
print "print bout3 stat: "
print_param(bout3)
#print bout3[0,0]
#print -np.log(bout3[0,0])
print "optimization error = ", c0
dE, dU, dW, db, dV, dc = model2.bptt(gx_start[0], gx_batch_len[0])
dc = np.asarray(dc)
print "analytic derivative = ", dc[1]
print "after perturbation:"
#temp = model2.c.eval()
temp = model2.c.get_value()
print type(temp)
temp[1] = temp[1] + 0.0001
model2.c.set_value(temp)
c1, _ = model2.optimization_error(gx_start[0], gx_batch_len[0])
print "finite difference derivative = ", (c1-c0)/0.0001
print "end test gradients"

print "\ncurrent parameters:"
print_param(model2.E.get_value())
print_param(model2.U.get_value())
print_param(model2.V.get_value())
print_param(model2.W.get_value())
print_param(model2.b.get_value())
print_param(model2.c.get_value())

print "current derivatives"
dE, dU, dW, db, dV, dc = model2.bptt(gx_start[0], gx_batch_len[0])
param = np.asarray(dE)
print_param(param)
param = np.asarray(dU)
print_param(param)
param = np.asarray(dW)
print_param(param)
param = np.asarray(dV)
print_param(param)
param = np.asarray(db)
print_param(param)
param = np.asarray(dc)
print_param(param)
print "total program running time time = %.3f seconds" % (time.time()-ts)
sys.exit()
