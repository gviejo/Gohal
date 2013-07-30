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
beta = 2.0
length_memory = 15
noise_width = 0.01
correlation = "Z"

nb_trials = 42
nb_blocs = 100

cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0)})

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
# Comparison with Human Learnig
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

f = dict()  
for i in [1,2,3]:
    tmp = []
    for j in xrange(len(data.keys())):
        f[data.keys()[j]] = j
        tmp.append(np.mean(data[data.keys()[j]][i], axis = 0))
    f[i] = np.array(tmp)
# -----------------------------------

# -----------------------------------
# Computation of weight
# -----------------------------------
w = []
for i in [1,2,3]:
    w.append([])
    for j in xrange(f[i].shape[1]):
        w[-1].append((f[i][2,j]-f[i][1,j])/(f[i][0,j]-f[i][1,j]))
w = np.array(w)
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
    fill_between(range(len(w[i])), np.zeros((len(w[i]))), w[i], alpha = 0.2, color= 'blue')
    fill_between(range(len(w[i])), w[i], np.ones((len(w[i]))), alpha = 0.2, color = 'green')
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

