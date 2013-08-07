#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

scripts to load and test parameters

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization

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
        reward = cats.getOutcome(state, action)
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
noise_width = 0.01
correlation = "diff"
length_memory = 15

nb_trials = human.responses['meg'].shape[1]
#nb_blocs = human.responses['meg'].shape[0]
nb_blocs = 46

cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, 15, 0.01, 1.0)})

# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
#p = dict()
#for m in models.iterkeys():    
 #   f = open(m, 'rb')
  #  p[m] = pickle.load(f)



# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------

opt = Optimization(human, cats, nb_trials, nb_blocs)
p = dict()

for m in models.iterkeys():
    p[m] = opt.stochasticOptimization(models[m], correlation, 10000)
    

data = dict()

for m in models.iterkeys():
    models[m].setAllParameters(p[m])
    opt.testModel(models[m])
    models[m].state = convertStimulus(np.array(models[m].state))
    models[m].action = convertAction(np.array(models[m].action))
    models[m].responses = np.array(models[m].responses)
    data[m] = extractStimulusPresentation2(models[m].responses, models[m].state, models[m].action, models[m].responses)




# -----------------------------------
# Plot
# -----------------------------------
ticks_size = 15
legend_size = 15
title_size = 20
label_size = 19

# Probability of correct responses
figure(correlation)
rc('legend',**{'fontsize':legend_size})
tick_params(labelsize = ticks_size)

for i,m in zip([1,2], ['bmw','kalman']):
    subplot(2,2,i)
    for j in [1,2,3]:
        plot(np.mean(data[m][j], 0), linewidth = 2)
        plot(np.mean(opt.data_human[j], 0), '--', alpha= 0.8)
    ylim(0,1)
    title(m)
    grid()

show()
