#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test a dictionnary of parameters for each subject

run subjectTest.py -i kalman.txt -m kalman -s "S1"

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Likelihood
import matplotlib.pyplot as plt
from time import time

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
# if not sys.argv[1:]:
#     sys.stdout.write("Sorry: you must specify at least 1 argument")
#     sys.stdout.write("More help avalaible with -h or --help option")
#     sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the parameters file to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
parser.add_option("-s", "--subject", action="store", help="Which subject to plot \n Ex : -s S1", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def testParameters():
    p = eval(open(options.input, 'r').read())[options.model]
    model.initializeList()
    for s in p.keys():
        for i in p[s].iterkeys():
            model.setParameter(i, p[s][i])

        for i in xrange(nb_blocs):
            sys.stdout.write("\r Sujet : %s | Blocs : %i" % (s,i)); sys.stdout.flush()                    
            cats.reinitialize()
            model.initialize()
            for j in xrange(nb_trials):
                state = cats.getStimulus(j)
                action = model.chooseAction(state)
                reward = cats.getOutcome(state, action)
                model.updateValue(reward)

    model.state = convertStimulus(np.array(model.state))
    model.action = convertAction(np.array(model.action))
    model.responses = np.array(model.responses)
    model.reaction = np.array(model.reaction)
    return p

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.630     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.6666   
alpha = 0.5
noise = 0.01
length_memory = 8
threshold = 1.2

nb_trials = human.responses['meg'].shape[1]
nb_blocs = 10
cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bwm_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold),
               'qlearning':QLearning('q', cats.states, cats.actions, gamma, alpha, beta)
               })



# ------------------------------------
# Parameter testing
# ------------------------------------
data = []
for s in human.subject['meg'].iterkeys():
    data.append([s])    
    for m in ['bwm_v1', 'bwm_v2', 'kalman', 'qlearning']:
        p = eval(open(m+".txt", 'r').read())
        print m, s
        opt = Likelihood(human, models[m], None, 0, 0, 0, 0, 0, 0, True)
        opt.current_subject = s
        param = [p[s][i] for i in opt.p_order]
        llh = opt.computeLikelihood(param)
        data[-1].append(np.round(llh,2))
        


