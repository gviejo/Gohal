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

sys.path.append("../../src")
from fonctions import *
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Selection import KSelection
from ColorAssociationTasks import CATS

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
def testSubject(ptr_m, sar):	
	print m
	ptr_m.initialize()

	sum_ll = 0.0
	for trial in sar:
		#sys.stdin.readline()
		state = cvt[trial[0]]
		print "State :", state
		action = ptr_m.chooseAction(state)		
		print "Model Action :", convertAction(action)
		true_action = trial[1]-1		
		print "True Action :", true_action
		if type(ptr_m.values) == dict:
			p_a = ptr_m.values[0][ptr_m.values[state]][true_action]
		else:			
			p_a = ptr_m.values[true_action]
		print "p_a", p_a
		print "log (p_a) ", np.log(p_a)
		sum_ll = sum_ll + np.log(p_a)
		print "sum ", sum_ll
        reward = trial[2]
        ptr_m.updateValue(reward)
	return tmp



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
length_memory = 100     # size of working memory
threshold = 0.2         # inference threshold
sigma = 0.00002         # updating rate of the average reward

cats = CATS(0)

cvt = {i:'s'+str(i) for i in [1,2,3]}

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise, threshold),
			   'ksel':KSelection(KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
                       BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise, threshold),
                       sigma)})

data = dict({m:dict() for m in models.iterkeys()})

for m in models.iterkeys():
	print m
	models[m].initializeList()
	for subject in X.iterkeys():
	 	llh = 0.0
		for bloc in X[subject].iterkeys():			
			llh = llh * testSubject(models[m], X[subject][bloc]['sar'])
		data[m][subject] = llh




