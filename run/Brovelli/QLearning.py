#!/usr/bin/python
# encoding: utf-8
"""
QLearning.py

Implementent simple Q-learning with brovelli experiment
python Qlearning.py

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
sys.path.append("../../src")
import os
from copy import deepcopy
from optparse import OptionParser
import numpy as np
from fonctions import *
from pylab import plot, figure, show, subplot, legend, xlim, errorbar, grid
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
    stimulus[-1].append(state)

    #choose best action 
    action = getBestActionSoftMax(state, values, beta)
    action_list[-1].append(action)
    reaction[-1].append(computeEntropy(values[0][values[state]], beta))
    # set response + get outcome
    reward = cats.getOutcome(state, action)
    answer.append((reward==1)*1)

    # QLearning
    delta = reward + gamma*np.max(values[0][values[state]]) - values[0][values[(state,action)]]
    values[0][values[(state, action)]] = values[0][values[(state, action)]] + alpha*delta

    if display == True:
        print (state, action, reward)
        print cats.correct
        displayQValues(cats.states, cats.actions, values)
        print '\n'
        #sys.stdin.read(1)

        

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
gamma = 0.1 #discount facto
alpha = 0.1
beta = 1

nb_trials = 42

cats = CATS()
Qdata = []
stimulus = []
action_list = []
reaction = []
# -----------------------------------
# Learning session
# -----------------------------------
for i in xrange(72):
    answer = []
    action_list.append([])
    reaction.append([])
    stimulus.append([])
    cats.reinitialize()
    values = createQValuesDict(cats.states, cats.actions)
    for j in xrange(nb_trials):
        iterationStep(j, values, True)
    Qdata.append(list(answer))
# -----------------------------------

responses = np.array(Qdata)
action = convertAction(np.array(action_list))
stimulus = convertStimulus(np.array(stimulus))

data = extractStimulusPresentation(stimulus, action, responses)

for m, s in zip(data['mean'],data['sem']):
    errorbar(range(1, len(m)+1), m, s)

grid()
show()







