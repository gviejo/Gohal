#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory Entropy Evolution

plot the evolution of entropy for differents conditions

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
noise = 0.01
length_memory = 15

nb_trials = 42
nb_blocs = 46
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
    sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                       
    cats.reinitialize()
    bmw.initialize()
    for j in xrange(nb_trials):
        iterationStep(j, bmw, True)        
        #sys.stdin.read(1)

# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------

data = dict()
bmw.state = convertStimulus(np.array(bmw.state))
bmw.action = convertAction(np.array(bmw.action))
bmw.responses = np.array(bmw.responses)
pcr = extractStimulusPresentation(bmw.responses, bmw.state, bmw.action, bmw.responses)
bmw.reaction = np.array(bmw.reaction)
rt = getRepresentativeSteps(bmw.reaction, bmw.state, bmw.action, bmw.responses)
m_rt, sem_rt = computeMeanRepresentativeSteps(rt[0])

rt2 = extractStimulusPresentation(bmw.reaction, bmw.state, bmw.action, bmw.responses)

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

# Probability of correct responses
figure()
subplot(311)
for mean, sem in zip(pcr['mean'], pcr['sem']):
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('pcr')

subplot(312)
plot(range(1, len(m_rt)+1), m_rt, 'o-',linewidth = 2)
fill_between(range(1,len(m_rt)+1), m_rt-sem_rt, m_rt+sem_rt, alpha = 0.4)
axvspan(1, 5, facecolor='g', alpha=0.2)
axvspan(5, len(m_rt)+1, facecolor='cyan', alpha=0.2)
xticks(range(len(m_rt)))
xlim(1,len(m_rt))
grid()
title('entropy')

subplot(313)
for mean, sem in zip(rt2['mean'], rt2['sem']):
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
title('entropy')

show()
