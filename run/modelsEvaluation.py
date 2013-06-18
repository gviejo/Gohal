#!/usr/bin/python
# encoding: utf-8
"""
modelsEvaluation.py

evaluation of all models on ColorAssociationTask.py
       -> Human Behaviour : <MEG>      <fMRI>
       -> ModelBased :      <No-noise> <Noise>
       -> KalmanQlearning : <Slow>     <Fast>
       -> QLearning :       <Slow>     <Fast>

For each model => Time Reaction + Accuracy
see 'Differential roles of caudate nucleus and putamen during instrumental learning.
     Brovelli & al, 2011'

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
#from pylab import *
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
def convertStimulus(state):
    return (state == 's1')*1+(state == 's2')*2 + (state == 's3')*3
def convertAction(action):
    return (action=='thumb')*1+(action=='fore')*2+(action=='midd')*3+(action=='ring')*4+(action=='little')*5

def iterationStep(iteration, qlearning, klearning, tlearning, display = True):
    state = cats.getStimulus(iteration)

    for m in [qlearning, klearning, tlearning]:
        action = m.chooseAction(state)
        reward = cats.getOutcome(state, action, m.name)
        if m.name == 't':
            m.updateTrees(state, reward)
        else:
            m.updateValue(reward)

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
gamma = 0.1 #discount factor
alpha = 1
beta = 1
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.9     # discount factor
sigma = 0.02    # updating rate of the average reward
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_trials = 42
nb_blocs = 100
cats = CATS()
qlearning = QLearning('q', cats.states, cats.actions, gamma, alpha, beta)
klearning = KalmanQLearning('k', cats.states, cats.actions, gamma, beta*3.0, eta, var_obs, sigma, init_cov, kappa)
tlearning = TreeConstruction('t', cats.states, cats.actions, alpha, beta, gamma)
models=[m.name for m in [qlearning, klearning, tlearning]]    
cats = CATS_MODELS(nb_trials, models)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
for i in xrange(nb_blocs):
    sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                        
    cats.reinitialize()
    qlearning.initialize()
    klearning.initialize()
    tlearning.initialize()
    for j in xrange(nb_trials):
        iterationStep(j, qlearning, klearning, tlearning, False)
# -----------------------------------


# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../fMRI',39)}))
# -----------------------------------

#order data
data = dict()
data['pcr'] = dict()
for i in human.directory.keys():
    data['pcr'][i] = extractStimulusPresentation(human.stimulus[i], human.action[i], human.responses[i])
for m in [qlearning, klearning, tlearning]:
    m.state = convertStimulus(np.array(m.state))
    m.action = convertAction(np.array(m.action))
    m.responses = np.array(m.responses)
    data['pcr'][m.name] = extractStimulusPresentation(m.state, m.action, m.responses)

data['rt'] = dict()
for i in human.directory.keys():
    data['rt'][i] = dict()
    step, indice = getRepresentativeSteps(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
    data['rt'][i]['mean'], data['rt'][i]['sem'] = computeMeanRepresentativeSteps(step) 

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
figure()
subplot(421)
for mean, sem in zip(data['pcr']['fmri']['mean'], data['pcr']['fmri']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('fMRI')
subplot(422)
for mean, sem in zip(data['pcr']['meg']['mean'], data['pcr']['meg']['sem']):
    #ax2.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('MEG')

subplot(423)
for mean, sem in zip(data['pcr']['t']['mean'], data['pcr']['t']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Tree-Learning')


subplot(425)
for mean, sem in zip(data['pcr']['q']['mean'], data['pcr']['q']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Q-learning')

subplot(427)
for mean, sem in zip(data['pcr']['k']['mean'], data['pcr']['k']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Kalman-Qlearning')


figure()
subplot(421)
errorbar(range(1,16), data['rt']['fmri']['mean'], data['rt']['fmri']['sem'], linewidth = 2)
grid()
title('fMRI')
subplot(422)
errorbar(range(1,16), data['rt']['meg']['mean'], data['rt']['meg']['sem'], linewidth = 2)
grid()
title('MEG')

show()











