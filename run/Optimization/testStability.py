#!/usr/bin/python
# encoding: utf-8
"""
testStability.py

to see if log-likelihood is robust to noise

See : Trial-by-trial data analysis using computational models, Daw, 2009

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *
from HumanLearning import HLearning
from Models import *
from Selection import KSelection
from ColorAssociationTasks import CATS
from Sweep import Likelihood
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
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------


# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
X = human.subject['meg']
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001            # variance of evolution noise v
var_obs = 0.05          # variance of observation noise n
gamma = 0.4            # discount factor
init_cov = 10           # initialisation of covariance matrice
kappa = 0.1             # unscentered transform parameters
beta = 5.2              # temperature for kalman soft-max
noise = 0.000           # variance of white noise for working memory
length_memory = 7      # size of working memory
threshold = 1           # inference threshold
sigma = 0.00002         # updating rate of the average reward
alpha = 1.0
#########################
#optimization parameters
fname = 'minimize'
n_run = 5
n_grid = 10
maxiter = 100
maxfun = 100
xtol = 0.001
ftol = 0.001
disp = False
#########################
cats = CATS(0)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bmw_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold),
               'qlearning':QLearning('ql', cats.states, cats.actions, gamma, alpha, beta)
			   })
model = models[options.model]
opt = Likelihood(human, model, fname, n_run, n_grid, maxiter, maxfun, xtol, ftol, disp)


opt.set(model, 'S1')

ll = list()
psampled = list()

moving_p = 2

cst = np.array([opt.p[i][1] for i in opt.p_order])

for i in xrange(200):		
	cst[moving_p] = np.random.uniform(opt.ranges[moving_p][0], opt.ranges[moving_p][1])
	ll.append(opt.computeLikelihood(cst))
	print i, cst
	psampled.append(cst[moving_p])

ll = np.array(ll)
psampled = np.array(psampled)

# ---------------------------------
# Plot
# ---------------------------------
ion()
fig = plt.figure()
plot(psampled, ll, 'o')

plt.show()



