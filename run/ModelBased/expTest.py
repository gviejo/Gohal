#!/usr/bin/python
# encoding: utf-8
"""
expGraph.py

script to test the modelbased

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
from pylab import *
from matplotlib import *
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from GraphConstruction import ModelBased



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
def iterationStep(iteration, mods, display = True):
    state = cats.getStimulus(iteration)
    stimulus[-1].append(state)

    #choose best action from model based
    action = mods.chooseAction(mods.g[state])
    action_list[-1].append(action)

    # set response + get outcome
    reward = cats.getOutcome(state, action)
    answer.append((reward==1)*1)

    # update Model Based
    mods.updateTrees(state, reward)
    if display == True:
        print (state, action, reward)
        print_dict(mods.g[state])
        print cats.correct
        print '\n'
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
gamma = 0.1 #discount factor
alpha = 1
beta = 3
noise_width = 0.01

nb_trials = 42
nb_blocs = 100

cats = CATS()

mods = ModelBased(cats.states, cats.actions, noise_width)

responses = []
stimulus = []
action_list = []

# -----------------------------------
# Learning session
# -----------------------------------
for i in xrange(nb_blocs):
    answer = []
    action_list.append([])
    stimulus.append([])
    cats.reinitialize()
    mods.reinitialize(cats.states, cats.actions)
    for j in xrange(nb_trials):
        iterationStep(j, mods, False)
    responses.append(answer)

responses = np.array(responses)
action = convertAction(np.array(action_list))
stimulus = convertStimulus(np.array(stimulus))

data = extractStimulusPresentation(stimulus, action, responses)
# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
ticks_size = 15
legend_size = 15
title_size = 22
label_size = 19

rc('legend',**{'fontsize':legend_size})
tick_params(labelsize = ticks_size)
for m, s in zip(data['mean'],data['sem']):
    errorbar(range(1, len(m)+1), m, s)


show()
# -----------------------------------







