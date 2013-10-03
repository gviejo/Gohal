#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory :
based on bayesian inference to calcul p(a/s)

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from Models import BayesianWorkingMemory
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
# -----------------------------------
# FONCTIONS
# -----------------------------------
def testModel():
    bww.initializeList()
    for i in xrange(nb_blocs):
        sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                    
        cats.reinitialize()
        bww.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)
    bww.state = convertStimulus(np.array(bww.state))
    bww.action = convertAction(np.array(bww.action))
    bww.responses = np.array(bww.responses)
    bww.reaction = np.array(bww.reaction)
    
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
beta = 1.0
noise = 0.0
length_memory = 10
threshold = 1.0

nb_trials = 42
nb_blocs = 48
cats = CATS()

bww = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise, threshold)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
bww.initializeList()
bww.initialize()

testModel()
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(bww.responses, bww.state, bww.action, bww.responses)
pcr_human = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

ratio = ((np.max(human.reaction['meg'])-np.min(human.reaction['meg']))/(np.max(bww.reaction)-np.min(bww.reaction)))
bww.reaction = bww.reaction*(0.1*ratio)
#bww.reaction = bww.reaction + (np.min(bww.reaction)-np.min(human.reaction['meg']))
bww.reaction = bww.reaction + 0.45

step, indice = getRepresentativeSteps(bww.reaction, bww.state, bww.action, bww.responses)
rt = computeMeanRepresentativeSteps(step)

step, indice = getRepresentativeSteps(human.reaction['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
rt_human = computeMeanRepresentativeSteps(step) 
    
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

# Probability of correct responses
figure()
for i in xrange(3):
    subplot(4,1,i+1)
    plot(range(len(pcr['mean'][i])), pcr['mean'][i], linewidth = 2, linestyle = '-', color = 'black')    
    errorbar(range(len(pcr['mean'][i])), pcr['mean'][i], pcr['sem'][i], linewidth = 2, linestyle = '-', color = 'black')
    plot(range(len(pcr_human['mean'][i])), pcr_human['mean'][i], linewidth = 2, linestyle = ':', color = 'black')    
    errorbar(range(len(pcr_human['mean'][i])), pcr_human['mean'][i], pcr_human['sem'][i], linewidth = 2, linestyle = ':', color = 'black')
    
    grid()
    ylim(0,1)
subplot(4,1,4)
plot(range(len(rt[0])), rt[0], linewidth = 2, linestyle = '-', color = 'black')
errorbar(range(len(rt[0])), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
plot(range(len(rt_human[0])), rt_human[0], linewidth = 2, linestyle = ':', color = 'black')
errorbar(range(len(rt_human[0])), rt_human[0], rt_human[1], linewidth = 2, linestyle = ':', color = 'black')
show()
