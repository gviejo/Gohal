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


# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
beta = 1.0
noise = 0.0
length_memory = 10

nb_trials = 42
nb_blocs = 48
cats = CATS()

bww = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise, beta)


# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
bww.initializeList()
bww.initialize()
action = bww.chooseAction('s1')
#bww.updateValue(0.0)
bww.updateValue(1.0)
sys.exit()

#testModel()
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(bww.responses, bww.state, bww.action, bww.responses)
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

# Probability of correct responses
figure()
subplot(211)
for mean, sem in zip(pcr['mean'], pcr['sem']):
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('pcr')

show()
