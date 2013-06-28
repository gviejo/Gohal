#!/usr/bin/python
# encoding: utf-8
"""

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from HumanLearning import HLearning
from Models import QLearning
from Models import KalmanQLearning
from Models import TreeConstruction
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
    if display == True:
        print (state, action, reward)
        print cats.correct
        displayQValues(cats.states, cats.actions, m.values)

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.05     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 3.0

nb_trials = 42
nb_blocs = 100
cats = CATS()

klearning = KalmanQLearning('k', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa)

responses = []
stimulus = []
action_list = []
reaction = []

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
for i in xrange(nb_blocs):
    sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                        
    cats.reinitialize()
    klearning.initialize()
    responses.append([])
    stimulus.append([])
    action_list.append([])
    reaction.append([])
    for j in xrange(nb_trials):
        iterationStep(j, klearning, True)
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
data = dict()
klearning.state = convertStimulus(np.array(klearning.state))
klearning.action = convertAction(np.array(klearning.action))
klearning.responses = np.array(klearning.responses)
pcr = extractStimulusPresentation(klearning.responses, klearning.state, klearning.action, klearning.responses)
# -----------------------------------
step, indice = getRepresentativeSteps(np.array(klearning.reaction), klearning.state, klearning.action, klearning.responses)
m, s = computeMeanRepresentativeSteps(step) 
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

# Reaction time with representative steps
subplot(212)
errorbar(range(1,len(m)+1), m, s, linewidth = 2)
grid()
title('rt')
show()
