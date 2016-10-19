#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
modelsEvaluation.py

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
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.9     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

noise_width = 0.008 #model based noise If 0, no noise

nb_trials = 42
nb_blocs = 100
cats = CATS()

models = dict({'qslow':QLearning('qslow', cats.states, cats.actions, 0.3, 0.3, 3.0), # gamma, alpha, beta
               'kslow':KalmanQLearning('kslow', cats.states, cats.actions, 0.99, 2.0, eta, var_obs, init_cov, kappa),#gamma, beta
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

# -----------------------------------
#order data
# -----------------------------------
data = dict()
data['pcr'] = dict()
for i in human.directory.keys():
    print i
    data['pcr'][i] = extractStimulusPresentation(human.responses[i], human.stimulus[i], human.action[i], human.responses[i])
for m in models.itervalues():
    print m.name
    m.state = convertStimulus(np.array(m.state))
    m.action = convertAction(np.array(m.action))
    m.responses = np.array(m.responses)
    data['pcr'][m.name] = extractStimulusPresentation(m.responses, m.state, m.action, m.responses)
# -----------------------------------
data['rt'] = dict()
for i in human.directory.keys():
    print i
    data['rt'][i] = dict()
    step, indice = getRepresentativeSteps(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
    data['rt'][i]['mean'], data['rt'][i]['sem'] = computeMeanRepresentativeSteps(step) 
for i in models.iterkeys():
    print i
    data['rt'][i] = dict()
    step, indice = getRepresentativeSteps(np.array(models[i].reaction), models[i].state, models[i].action, models[i].responses)
    data['rt'][i]['mean'], data['rt'][i]['sem'] = computeMeanRepresentativeSteps(step) 
# -----------------------------------
data['rt2'] = dict()
for i in human.directory.keys():
    print i
    data['rt2'][i] = extractStimulusPresentation(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
for m in models.itervalues():
    print m.name
    m.reaction = np.array(m.responses)
    data['rt2'][m.name] = extractStimulusPresentation(m.reaction, m.state, m.action, m.responses)
    
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
plot_order = ['fmri','meg','tree','treenoise','qslow','kslow']
plot_name = ['fMRI', 'MEG', 'Tree-Learning', 'Noisy Tree-Learning', 'Q-Learning', 'Kalman Q-Learning']


ticks_size = 15
legend_size = 15
title_size = 20
label_size = 15

fig = figure()
rc('legend',**{'fontsize':legend_size})

# Probability of correct responses
count = 1
for i,j in zip(plot_order, plot_name):
    ax = fig.add_subplot(3,2,count)
    for mean, sem in zip(data['pcr'][i]['mean'], data['pcr'][i]['sem']):
        #ax1.plot(range(len(mean)), mean, linewidth = 2)
        ax.errorbar(range(len(mean)), mean, sem, linewidth = 2)
    legend()
    grid()
    ylim(0,1)
    title(j, fontsize = title_size)
    tick_params(labelsize = label_size)
    ylabel('Performance %', fontsize = label_size)
    if count == 5 or count == 6:
        xlabel('Trial', fontsize = label_size)
    ax.set_yticklabels( () )
    if count in [1,2,3,4]:
        ax.set_xticklabels( () )
    count +=1

rcParams['figure.figsize'] = 26, 4
    
fig = figure()
rc('legend',**{'fontsize':legend_size})
count = 1
# Probability of correct responses
for i,j in zip(['fmri','meg'],['fMRI', 'MEG']):
    ax = fig.add_subplot(1,2,count)
    for mean, sem in zip(data['pcr'][i]['mean'], data['pcr'][i]['sem']):
        #ax1.plot(range(len(mean)), mean, linewidth = 2)
        ax.errorbar(range(len(mean)), mean, sem, linewidth = 2)
    legend()
    grid()
    ylim(0,1)
    title(j, fontsize = title_size)
    tick_params(labelsize = label_size)
    ylabel('Performance %', fontsize = label_size)
    xlabel('Trial', fontsize = label_size)
    ax.set_yticklabels( [0,20,40,60,80,100] )
    ax.set_xticklabels( () )
    count +=1


fig = figure()
rc('legend',**{'fontsize':legend_size})
count = 1
for i,j in zip(['tree','treenoise'],['Tree-Learning', 'Noisy Tree-Learning']):
    ax = fig.add_subplot(1,2,count)
    for mean, sem in zip(data['pcr'][i]['mean'], data['pcr'][i]['sem']):
        #ax1.plot(range(len(mean)), mean, linewidth = 2)
        ax.errorbar(range(len(mean)), mean, sem, linewidth = 2)
    legend()
    grid()
    ylim(0,1)
    title(j, fontsize = title_size)
    tick_params(labelsize = label_size)
    ylabel('Performance %', fontsize = label_size)
    xlabel('Trial', fontsize = label_size)
    ax.set_yticklabels( () )
    ax.set_xticklabels( () )
    count +=1

fig = figure()
rc('legend',**{'fontsize':legend_size})
count = 1
for i,j in zip(['qslow','kslow'],['Q-Learning', 'Kalman Q-Learning']):
    ax = fig.add_subplot(1,2,count)
    for mean, sem in zip(data['pcr'][i]['mean'], data['pcr'][i]['sem']):
        #ax1.plot(range(len(mean)), mean, linewidth = 2)
        ax.errorbar(range(len(mean)), mean, sem, linewidth = 2)
    legend()
    grid()
    ylim(0,1)
    title(j, fontsize = title_size)
    tick_params(labelsize = label_size)
    ylabel('Performance %', fontsize = label_size)
    xlabel('Trial', fontsize = label_size)
    ax.set_yticklabels( () )
    ax.set_xticklabels( () )
    count +=1

show()






