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

noise_width = 0.0

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]

cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, 0.99, 2.0, eta, var_obs, init_cov, kappa),
               'tree':TreeConstruction('tree', cats.states, cats.actions, noise_width),
               'q':QLearning('q', cats.states, cats.actions, 0.3, 0.3, 3.0)})
# -----------------------------------


# -----------------------------------
# PARAMETERS Exploration
# -----------------------------------
sweep = Sweep_performances(human.responses['meg'], cats, nb_trials, nb_blocs)

parameters = dict({'tree':dict({'noise':[0.0, 0.001, 0.11]}),
                   'kalman':dict({'gamma':[0.01, 0.1, 0.5, 0.75, 0.99],
                                  'beta':[1.0, 2.0, 3.0]}),
                   'q':dict({'alpha':[0.01, 0.1, 0.5, 0.9],
                             'gamma':[0.01, 0.1, 0.5, 0.9]})})

data = dict()
nb_p = 0
for m in parameters.iterkeys():
    data[m] = dict()
    for p in parameters[m].iterkeys():        
        data[m][p] = sweep.exploreParameters(models[m], p, parameters[m][p])
        nb_p+=1
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

ticks_size = 15
legend_size = 15
title_size = 20
label_size = 15

#tree learning
fig = figure()
rc('legend',**{'fontsize':legend_size})
tmp = 1
for m in data.iterkeys():
    for p in data[m].iterkeys():
        subplot(nb_p,1,tmp)
        for v in data[m][p].iterkeys():
            plot(data[m][p][v], 'o-', linewidth = 1.4, label = str(v))
        legend()
        grid()
        title(m+" / "+p, fontsize = title_size)
        tmp+=1
show()
