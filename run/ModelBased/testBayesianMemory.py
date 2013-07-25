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
def iterationStep(iteration, m, display = True):
    state = cats.getStimulus(iteration)
    action = m.chooseAction(state)
    reward = cats.getOutcome(state, action)
    m.updateValue(reward)

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
beta = 5.0
noise = 0.0
length_memory = 100

nb_trials = 42
nb_blocs = 100
cats = CATS()

bmw = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise)

responses = []
stimulus = []
action_list = []
reaction = []

#bmw.initialize()
#cats.reinitialize()
#state = 's1'

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
for i in xrange(nb_blocs):
    #sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                        
    cats.reinitialize()
    bmw.initialize()
    for j in xrange(nb_trials):
        iterationStep(j, bmw, True)
        print "\n"
        #sys.stdin.read(1)
# -----------------------------------
bmw.responses = np.array(bmw.responses)


# -----------------------------------
#order data
# -----------------------------------
"""
data = dict()
bmw.state = convertStimulus(np.array(bmw.state))
bmw.action = convertAction(np.array(bmw.action))
bmw.responses = np.array(bmw.responses)
pcr = extractStimulusPresentation(bmw.responses, bmw.state, bmw.action, bmw.responses)
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
"""
plot(np.mean(bmw.responses, axis = 0))
show()

