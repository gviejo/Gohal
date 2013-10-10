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
gamma = 0.95            # discount factor
init_cov = 10           # initialisation of covariance matrice
kappa = 0.1             # unscentered transform parameters
beta = 5.5              # temperature for kalman soft-max
noise = 0.000           # variance of white noise for working memory
length_memory = 10     # size of working memory
threshold = 0.2         # inference threshold
sigma = 0.00002         # updating rate of the average reward

cats = CATS(0)

opt = Likelihood(human, 1)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise, threshold),
			   'ksel':KSelection(KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
                       BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise, threshold),
                       sigma)})

bww = models['bmw']
opt.set(bww, 'S9')
opt.searchStimOrder()
ll = list()
psampled = list()

for i in xrange(1000):
	print i
	t = np.random.uniform(0.01, 2.0)
	ll.append(opt.computeLikelihood(np.array([t, length_memory, noise])))
	psampled.append(t)

ll = np.array(ll)
psampled = np.array(psampled)

# ---------------------------------
# Plot
# ---------------------------------
fig = plt.figure()
plot(psampled, ll, 'o')

plt.show()



