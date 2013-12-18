#!/usr/bin/python
# encoding: utf-8
"""
subjectOptimization.py

scripts to optimize subject parameters 

See : Trial-by-trial data analysis using computational models, Daw, 2009

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
from optparse import OptionParser

import datetime
sys.path.append("../../src")
from fonctions import *
from HumanLearning import HLearning
from Models import *
from Selection import FSelection
from time import time
from ColorAssociationTasks import CATS
from Sweep import Likelihood

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt


# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
   sys.stdout.write("Sorry: you must specify at least 1 argument")
   sys.stdout.write("More help avalaible with -h or --help option")
   sys.exit(0)
parser = OptionParser()
parser.add_option("-m", "--model", action="store", help="The name of the model to optimize", default=False)
parser.add_option("-s", "--subject", action="store", help="The subject to optimize", default=False)
parser.add_option("-o", "--output", action="store", help="Output directory", default=False)
parser.add_option("-f", "--fonction", action="store", help="Scipy function", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------



# -----------------------------------
# FONCTIONS
# -----------------------------------
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001            # variance of evolution noise v
var_obs = 0.05          # variance of observation noise n
gamma = 0.95            # discount factor
init_cov = 10           # initialisation of covariance matrice
kappa = 0.1             # unscentered transform parameters
beta = 5.5              # temperature for kalman soft-max
noise = 0.000           # variance of white noise for working memory
length_memory = 7       # size of working memory
threshold = 1           # inference threshold
sigma = 0.00002         # updating rate of the average reward
gain = 0.5
alpha = 0.5 
#########################
#optimization parameters
n_run = 3
n_grid = 30
maxiter = 10000
maxfun = 10000
xtol = 0.01
ftol = 0.01
disp = True
#########################
cats = CATS(0)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bwm_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold),
               'qlearning':QLearning('q', cats.states, cats.actions, gamma, alpha, beta),
               'fusion':FSelection("test", cats.states, cats.actions, alpha, beta, gamma, length_memory, noise, threshold, gain)
              })

opt = Likelihood(human, models[options.model], options.fonction, n_run, n_grid, maxiter, maxfun, xtol, ftol, disp)
#########################
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
t1 = time()
opt.current_subject = options.subject
#p = np.array([1.93, 9.0, 0.96])
#llh = opt.computeLikelihood(p)
#print llh
opt.optimize(options.subject)
opt.save(options.output+options.subject+"_"+options.model+"_"+options.fonction+"_"+str(datetime.datetime.now()).replace(" ", "_"))

t2 = time()



print "\n"
print t2-t1
# -----------------------------------


