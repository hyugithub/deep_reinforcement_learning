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

#theano.config.compute_test_value = 'warn'

VOCABULARY_SIZE = 1024

num_rows = 4096*8
np.random.seed(12345)

with open("fake2.training.txt","w") as outfile:
    for i in np.arange(num_rows):
        start = np.random.choice(VOCABULARY_SIZE)
        end = start + 1
	if end >= VOCABULARY_SIZE - 0.5:
	    end = 0
        outfile.write(str(start)+" ")
        outfile.write(str(end))
        outfile.write("\n")
    outfile.close()

