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
import numpy as np
import datetime
sys.path.append("../../src")
from fonctions import *
from HumanLearning import HLearning
from Models import *
from Selection import KSelection
from ColorAssociationTasks import CATS
from Sweep import Likelihood

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

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
kalman = models['kalman']

p_opt, p_start = opt.optimize(bww)
#p_opt, p_start = opt.optimize(kalman)

opt2 = []
start2 = []
for i in p_opt.iterkeys():
	for j in xrange(len(p_opt[i])):
		opt2.append(p_opt[i][j])
		start2.append(p_start[i][j])


opt2 = np.array(opt2)
start2 = np.array(start2)

data = dict({'start':start2,
			 'opt':opt2,
			 'p_order':opt.p_order,
			 'subject':opt.subject,
			 'parameters':opt.p})

output = open("data_bwm_"+str(datetime.datetime.now()).replace(" ", "_"), 'wb')
pickle.dump(data, output)
output.close()
# ---------------------------------
# Plot
# ---------------------------------
# fig = plt.figure()

# tmp = bww.getAllParameters()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(opt[:,0], opt[:,1], opt[:,2], marker = '^')
# #ax.scatter(start[:,0], start[:,1], start[:,2], marker = 'o', alpha = 1)
# ax.set_xlim(tmp[tmp.keys()[0]][0], tmp[tmp.keys()[0]][2])
# ax.set_xlabel(tmp.keys()[0])
# ax.set_ylim(tmp[tmp.keys()[1]][0], tmp[tmp.keys()[1]][2])
# ax.set_ylabel(tmp.keys()[1])
# ax.set_zlim(tmp[tmp.keys()[2]][0], tmp[tmp.keys()[2]][2])
# ax.set_zlabel(tmp.keys()[2])
# plt.show()



