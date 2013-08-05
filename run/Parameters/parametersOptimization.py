#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

optimization of parameters for the differents models
through a genetic algorithm

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
correlation = "Z"
length_memory = 15

nb_trials = human.responses['meg'].shape[1]
#nb_blocs = human.responses['meg'].shape[0]
nb_blocs = 46

cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0)})

# -----------------------------------

# -----------------------------------
# PARAMETERS Optimization
# -----------------------------------
opt = Optimization(human, cats, nb_trials, nb_blocs)


for m in models.iterkeys():
    p = opt.simulatedAnnealing(models[m], correlation)
    with open(m, 'wb') as fp:
        pickle.dump(p, fp)
    
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
p = dict()
for m in models.iterkeys():    
    f = open(m, 'rb')
    p[m] = pickle.load(f)
