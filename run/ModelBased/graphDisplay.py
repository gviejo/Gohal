#!/usr/bin/python
# encoding: utf-8
"""
expGraph.py

script to test the modelbased
display the evoluation of the tree with matplotlib
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
from Plot import PlotTree
from GraphConstruction import ModelBased
from Models import TreeConstruction
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

    #choose best action from model based
    action = mods.chooseAction(state)

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
    return state, action
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
gamma = 0.1 #discount factor
alpha = 1
beta = 3
noise_width = 0.1

nb_trials = 20
nb_blocs = 1
cats = CATS()

#mods = ModelBased(cats.states, cats.actions, noise_width)
mods = TreeConstruction('tree', cats.states, cats.actions, noise_width)

plottree = PlotTree(mods.g, mods.dict_action, time_sleep = 0.01)
# -----------------------------------
# Learning session
# -----------------------------------
for i in xrange(nb_blocs):
    cats.reinitialize()
    #mods.reinitialize(cats.states, cats.actions)
    mods.initialize()
    answer = []
    for j in xrange(nb_trials):
        print j
        if j == 2:
            plottree.saveFigure('../../../Dropbox/ISIR/JournalClub/images/fig9.pdf')
        state, action = iterationStep(j, mods, False)
        plottree.updateTree(mods.g, (state, action))
        
# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------

# -----------------------------------







