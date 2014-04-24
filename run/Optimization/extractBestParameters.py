#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

to extract best parameters from scipy optimize

run parameterTest.py -i data_model_date -m 'model'

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from Selection import *
from matplotlib import *
from pylab import *

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
parser.add_option("-o", "--output", action="store", help="The name of the output files", default=False)

(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def rank(value, lambdaa, epsilon):
    m,n = value.shape
    #assert m>=n
    assert len(lambdaa) == n
    assert np.sum(lambdaa) == 1
    assert epsilon < 1.0
    ideal = np.max(value, 0)
    nadir = np.min(value, 0)
    tmp = lambdaa*((ideal-value)/(ideal-nadir))
    return np.max(tmp, 1)+epsilon*np.sum(tmp,1) 

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
cats = CATS(0)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions)})

p_order = models['fusion'].bounds.keys()
# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
f = open(options.input, 'rb')
data = pickle.load(f)
f.close()
# -----------------------------------
r = dict()
p_test = dict()
for s in data.keys():
    r[s] = rank(data[s][:,1:3], [0.5, 0.5], 0.1)
    ind = np.argmin(r[s])
    p_test[s] = dict({'fusion':{}})
    for i in xrange(len(p_order)):
        p_test[s]['fusion'][p_order[i]] = data[s][ind][i]
# -----------------------------------
# Order data
# -----------------------------------
fig_pareto = figure(figsize = (12, 9))
ax1 = fig_pareto.add_subplot(1,1,1)
for i, s in zip(range(len(data.keys())), data.iterkeys()):    
    ax1.plot(data[s][:,1], data[s][:,2], "-o")
    
    ax1.set_title(s)
    

show()        

with open("parameters.pickle", 'wb') as f:
    pickle.dump(p_test, f)