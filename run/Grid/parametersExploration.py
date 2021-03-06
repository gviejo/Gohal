#!/usr/bin/python
# encoding: utf-8
"""
parameters_exploration.py

exploration of parameters for different models.
throught class Sweep
For each trial of each parameters set, 
compute a p-value by comparing with human data

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
noise_width = 0.008
correlation = "Z"


nb_trials = human.responses['meg'].shape[1]
#nb_blocs = human.responses['meg'].shape[0]
nb_blocs = 100

cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'tree':TreeConstruction('tree', cats.states, cats.actions, noise_width)})

# -----------------------------------
ticks_size = 15
legend_size = 15
title_size = 20
label_size = 15
fig = figure()
rc('legend',**{'fontsize':legend_size})
nb_row = 1
nb_col = 3
bad = dict({1:0,
            2:nb_col,
            3:2*nb_col,
            0:1})
# -----------------------------------
# PARAMETERS Exploration
# -----------------------------------
sweep = Sweep_performances(human, cats, nb_trials, nb_blocs)

parameters = dict({'tree':dict({'noise':[0.0, 0.001, 0.1]}),
                   'kalman':dict({'gamma':[0.001, 0.1, 0.9],
                                  'beta':[1.0, 3.0, 5.0]})})

for m in parameters.iterkeys():
    for p in parameters[m].iterkeys():        
        value = sweep.exploreParameters(models[m], p, parameters[m][p], correlation)
        print m+" "+p
        for i in [1, 2, 3]:
            for v in parameters[m][p]:
                subplot(nb_row*3, nb_col, bad[0]+bad[i])
                plot(value[v][i], 'o-', linewidth = 1.5, label = str(v))                    
            ylim(0,1)
            grid()
            legend()
        subplot(nb_row*3, nb_col, bad[0])
        title(m+" "+p)
        bad[0]+=1
# -----------------------------------

show()

# -----------------------------------
# Plot
# -----------------------------------
