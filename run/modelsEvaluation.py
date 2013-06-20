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
def iterationStep(iteration, models, display = True):
    state = cats.getStimulus(iteration)

    for m in models.itervalues():
        action = m.chooseAction(state)
        reward = cats.getOutcome(state, action, m.name)
        if m.__class__.__name__ == 'TreeConstruction':
            m.updateTrees(state, reward)
        else:
            m.updateValue(reward)


# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
#gamma = 0.9 #discount factor
#alpha = 0.9
beta = 1
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.9     # discount factor
sigma = 0.02    # updating rate of the average reward
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

noise_width = 0.008 #model based noise If 0, no noise

nb_trials = 42
nb_blocs = 100
cats = CATS()
models = dict({'qslow':QLearning('qslow', cats.states, cats.actions, 0.1, 0.1, beta),
               'qfast':QLearning('qfast', cats.states, cats.actions, 0.9, 0.9, beta),
               'kslow':KalmanQLearning('kslow', cats.states, cats.actions, 0.1, beta, eta, var_obs, sigma, init_cov, kappa),
               'kfast':KalmanQLearning('kfast', cats.states, cats.actions, 0.1, beta, eta, var_obs, sigma, init_cov, kappa),
               'tree':TreeConstruction('tree', cats.states, cats.actions),
               'treenoise':TreeConstruction('treenoise', cats.states, cats.actions, noise_width)})

cats = CATS_MODELS(nb_trials, models.keys())

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
for i in xrange(nb_blocs):
    sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                        
    cats.reinitialize()
    [m.initialize() for m in models.itervalues()]
    for j in xrange(nb_trials):
        iterationStep(j, models, False)
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
for m in models.itervalues():
    m.state = convertStimulus(np.array(m.state))
    m.action = convertAction(np.array(m.action))
    m.responses = np.array(m.responses)
    data['pcr'][m.name] = extractStimulusPresentation(m.state, m.action, m.responses)

data['rt'] = dict()
for i in human.directory.keys():
    data['rt'][i] = dict()
    step, indice = getRepresentativeSteps(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
    data['rt'][i]['mean'], data['rt'][i]['sem'] = computeMeanRepresentativeSteps(step) 
for i in ['tree', 'treenoise']:
    data['rt'][i] = dict()
    step, indice = getRepresentativeSteps(np.array(models[i].reaction), models[i].state, models[i].action, models[i].responses)
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
for mean, sem in zip(data['pcr']['tree']['mean'], data['pcr']['tree']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Tree-Learning')
subplot(424)
for mean, sem in zip(data['pcr']['treenoise']['mean'], data['pcr']['treenoise']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Noisy Tree-Learning')



subplot(425)
for mean, sem in zip(data['pcr']['qslow']['mean'], data['pcr']['qslow']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Slow Q-learning')

subplot(426)
for mean, sem in zip(data['pcr']['qfast']['mean'], data['pcr']['qfast']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Fast Q-learning')


subplot(427)
for mean, sem in zip(data['pcr']['kslow']['mean'], data['pcr']['kslow']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Slow Kalman-Qlearning')

subplot(428)
for mean, sem in zip(data['pcr']['kfast']['mean'], data['pcr']['kfast']['sem']):
    #ax1.plot(range(len(mean)), mean, linewidth = 2)
    errorbar(range(len(mean)), mean, sem, linewidth = 2)
legend()
grid()
ylim(0,1)
title('Fast Kalman-Qlearning')



figure()
subplot(421)
errorbar(range(1,16), data['rt']['fmri']['mean'], data['rt']['fmri']['sem'], linewidth = 2)
grid()
title('fMRI')
subplot(422)
errorbar(range(1,16), data['rt']['meg']['mean'], data['rt']['meg']['sem'], linewidth = 2)
grid()
title('MEG')

subplot(423)
errorbar(range(1,16), data['rt']['tree']['mean'], data['rt']['tree']['sem'], linewidth = 2)
grid()
title('Tree Learning')
subplot(424)
errorbar(range(1,16), data['rt']['treenoise']['mean'], data['rt']['treenoise']['sem'], linewidth = 2)
grid()
title('Noisy Tree Learning')



show()











