#!/usr/bin/python
# encoding: utf-8
"""
parametersWexploration.py

computation of a w variable
part of each model
w*V1 + (1-w)*V2

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
from HumanLearning import HLearning
from Models import *
from Sweep import Sweep_performances
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
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.9     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

noise_width = 0.008

nb_trials = 42
nb_blocs = 100

cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, 0.99, 2.0, eta, var_obs, init_cov, kappa),
               'tree':TreeConstruction('tree', cats.states, cats.actions, noise_width)})

cats = CATS_MODELS(nb_trials, models.keys())
# -----------------------------------


# -----------------------------------
# Learning
# -----------------------------------
for i in xrange(nb_blocs):
    sys.stdout.write("\r Blocs : %i " % i); sys.stdout.flush()                        
    cats.reinitialize()
    [m.initialize() for m in models.itervalues()]
    for j in xrange(nb_trials):
        iterationStep(j, models, False)

# -----------------------------------



# -----------------------------------
# Comparison with Human Learnign
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
sweep = Sweep_performances(human, cats, nb_trials, nb_blocs)

data = dict()
data['human'] = extractStimulusPresentation2(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
for m in models.itervalues():
    m.state = convertStimulus(np.array(m.state))
    m.action = convertAction(np.array(m.action))
    m.responses = np.array(m.responses)
    data[m.name] = extractStimulusPresentation2(m.responses, m.state, m.action, m.responses)
    
for m in models.iterkeys():
    data[m]['jsd'] = dict()
    for i in [1,2,3]:
        data[m]['jsd'][i] = []
        for j in xrange(data[m][i].shape[1]):            
            data[m]['jsd'][i].append(sweep.computeSingleCorrelation(data['human'][i][:,j], data[m][i][:,j]))
        data[m]['jsd'][i] = np.array(data[m]['jsd'][i])
# -----------------------------------

# -----------------------------------
# Computation of weight
# -----------------------------------
w = dict()
for i in [1,2,3]:
    w[i] = data['tree']['jsd'][i]/(data['tree']['jsd'][i]+data['kalman']['jsd'][i])

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
ticks_size = 15
legend_size = 15
title_size = 20
label_size = 15

fig = figure()
rc('legend',**{'fontsize':legend_size})
count = 1
for i in [1,2,3]:
    subplot(3,1,count)
    for m in ['tree', 'kalman', 'human']:
        plot(np.mean(data[m][i], 0), 'o-', linewidth = 1.5, label = m)
    grid()
    legend()
    count+=1

fig = figure()
rc('legend',**{'fontsize':legend_size})
count = 1
for i in [1,2,3]:
    subplot(3,1,count)
    plot(w[i], 'o-', color= 'red')
    axhline(0.5,0,10, '--', color= 'black', linewidth = 4, alpha = 0.4)
    fill_between(range(len(w[i])), np.zeros((len(w[i]))), w[i], alpha = 0.06, color= 'blue')
    fill_between(range(len(w[i])), w[i], np.ones((len(w[i]))), alpha = 0.06, color = 'green')
    annotate('Tree Learning', (0, 0.1), color = 'blue')
    annotate('Kalman Q-Learning', (0, 0.8), color = 'green')
    ylim(0,1)
    grid()
    legend()
    count+=1
    
    
fig = figure()
rc('legend',**{'fontsize':legend_size})
count = 1
for i in [1,2,3]:
    subplot(3,1,count)
    plot(data['tree']['jsd'][i], 'o-', label = 'tree')
    plot(data['kalman']['jsd'][i], 'o-', label = 'kalman')
    grid()
    legend()
    count+=1

show()
