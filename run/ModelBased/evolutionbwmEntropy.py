#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory Entropy Evolution

3d plot of the evolution of entropy for differents levels of inference

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
def createStimulusList(i,j):
    n_states = len(cats.states)    
    s = np.tile(np.array(cats.states), ((nb_trials/n_states)+1, 1))
    map(np.random.shuffle, s)
    return s.flatten()

def modelTest(stimuli_list):
    for i in xrange(nb_blocs):
        cats.reinitialize()
        bww.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)

# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
beta = 1.7
length_memory = 100
noise_width = 0.01

correlation = "Z"

nb_trials = 42
nb_blocs = 100

cats = CATS(nb_trials)


bww = BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0)
bww.setEntropyEvolution(nb_blocs, nb_trials)
# -----------------------------------

# -----------------------------------
# Training session
# -----------------------------------
modelTest(createStimulusList(0,0))
bww.state = convertStimulus(np.array(bww.state))
bww.action = convertAction(np.array(bww.action))
bww.responses = np.array(bww.responses)
bww.reaction = np.array(bww.reaction)
#----------------------------------

#----------------------------------
# DATA Extraction
#----------------------------------
pcr = extractStimulusPresentation(bww.responses, bww.state, bww.action, bww.responses)

n_inferences = 5
gain = np.zeros(n_inferences)

step, indice = getRepresentativeSteps(bww.reaction, bww.state, bww.action, bww.responses)
    
distance = np.zeros((nb_blocs, len(cats.states)))
for i in xrange(1,4):
    tmp = np.reshape(np.where(bww.state[:,0:6] == i)[1], (nb_blocs, 2))
    distance[:,i-1] = tmp[:,1]-tmp[:,0]



#----------------------------------
# Plot
#----------------------------------

figure()
plot(np.transpose(pcr['mean']))
show()
