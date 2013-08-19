#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory Entropy Evolution

script to analyse the eoluation of entropy under differents conditions


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Models import BayesianWorkingMemory
from matplotlib import *
from Sweep import Optimization
from HumanLearning import HLearning
from matplotlib import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
noise = 0.0106
length_memory = 10

#nb_trials = 42
nb_trials = 42
nb_blocs = 100
cats = CATS()

bmw = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise)

opt = Optimization(human, cats, nb_trials, nb_blocs)

data = dict()
pcr = dict()
rs = []
for i in [1,2,3]:
    data[i] = []
    pcr[i] = []

# -----------------------------------

    
# -----------------------------------
# SESSION MODELS
# -----------------------------------
opt.testModel(bmw)
# -----------------------------------

# -----------------------------------
#order data
# -----------------------------------
bmw.state = convertStimulus(np.array(bmw.state))
bmw.action = convertAction(np.array(bmw.action))
bmw.responses = np.array(bmw.responses)
bmw.reaction = np.array(bmw.reaction)

for i in xrange(nb_blocs):
    for j in xrange(nb_trials):
    

# -----------------------------------
# Plot
# -----------------------------------

