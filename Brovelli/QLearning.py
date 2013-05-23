#!/usr/bin/python
# encoding: utf-8
"""
QLearning.py

Implementent simple Q-learning with brovelli experiment
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from copy import deepcopy
from optparse import OptionParser
import numpy as np
from fonctions import *
from pylab import plot, figure, show, subplot, legend, xlim
from ColorAssociationTasks import CATS

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------    
def iterationStep(iteration, values, display = True):
    
    #display current stimulus
    state = cats.getStimulus(iteration)

    #choose best action 
    action = getBestActionSoftMax(state, values, beta)
    
    # set response + get outcome
    reward = cats.getOutcome(state, action, iteration)
    answer.append((reward==1)*1)

    # QLearning
    delta = reward + gamma*np.max(values[0][values[state]]) - values[0][values[(state,action)]]
    values[0][values[(state, action)]] = values[0][values[(state, action)]] + alpha*delta

    if display == True:
        print (state, action, reward)
        print cats.correct
        displayQValues(cats.states, cats.actions, values)
        print '\n'

        

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
gamma = 0.1 #discount facto
alpha = 1
beta = 1

nb_trials = 42

cats = CATS()
values = createQValuesDict(cats.states, cats.actions)
Qdata = []
# -----------------------------------
# Learning session
# -----------------------------------
for i in xrange(72):
    answer = []
    cats.reinitialize(nb_trials)
    for j in xrange(nb_trials):
        iterationStep(j, values, False)
    Qdata.append(list(answer))
# -----------------------------------








