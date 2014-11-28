#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
# from Sweep import Likelihood
from Sferes import EA
from scipy.optimize import fmin_tnc, brute, fmin, leastsq, fmin_powell
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt


# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
# if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
# parser = OptionParser()
# parser.add_option("-m", "--model", action="store", help="The name of the model to optimize", default=False)
# parser.add_option("-s", "--subject", action="store", help="The subject to optimize", default=False)
# parser.add_option("-o", "--output", action="store", help="Output directory", default=False)
# parser.add_option("-f", "--fonction", action="store", help="Scipy function", default=False)
# (options, args) = parser.parse_args() 
# -----------------------------------



# -----------------------------------
# FONCTIONS
# -----------------------------------
def func(x):
	parameters = dict(zip(order, x))
	try :
		model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'],parameters, sferes = True)	
		opt = EA(data, s, model)
		llh, lrs = opt.getFitness()                                                                                      
		return np.sum((opt.mean[1]-opt.mean[0])**2)
	except :
		return 100000.0


# -----------------------------------
with open("../Sferes/fmri/S19.pickle", "rb") as f:
   data = pickle.load(f)

s = 'S19'
order = ['alpha','beta', 'noise','length','gain','threshold', 'sigma']
model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'],sferes = True)
bounds = [tuple(model.bounds[k]) for k in order]



x0 = [0.8, 10.0, 0.1, 1, 1.0, 0.1, 0.1]


# x, nfeval, rc = fmin_tnc(func, x0, approx_grad = True, bounds = bounds)

# resbrute = brute(func, tuple(bounds), Ns = 4, full_output = False, finish = fmin)

# x = leastsq(func, x0)

# x = fmin(func, x0)
# x = fmin_powell(func, x0, disp = True, retall=True)

parameters = dict(zip(order,x))
print parameters























sys.exit()
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
gamma = 0.81            # discount factor
init_cov = 10           # initialisation of covariance matrice
kappa = 0.1             # unscentered transform parameters
beta = 10.12              # temperature for kalman soft-max
noise = 0.1           # variance of white noise for working memory
length_memory = 9       # size of working memory
threshold = 56.0           # inference threshold
sigma = 0.00002         # updating rate of the average reward
gain = 84.0
alpha = 0.39
#########################
#optimization parameters
n_run = 1
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
#p = np.array([10, 0.0, 0.5, 2.0, 1.0, 2.0, 0.3])
#llh = opt.computeLikelihood(p)
rt = opt.computeFullLikelihood()
#print llh
#opt.optimize(options.subject)
#opt.save(options.output+options.subject+"_"+options.model+"_"+options.fonction+"_"+str(datetime.datetime.now()).replace(" ", "_"))

t2 = time()



print "\n"
print t2-t1
# -----------------------------------


