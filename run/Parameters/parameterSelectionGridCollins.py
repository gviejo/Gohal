#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

Grid-search for Selection model of Collins
Kalman : beta, gamma
Bayesian : length, noise
Collins : w_0


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Selection import CSelection
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization
import datetime


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
def testModel():
    selection.initializeList()
    for i in xrange(nb_blocs):
        cats.reinitialize()
        selection.initialize()
        for j in xrange(nb_trials):
            #opt.iterationStep(j, ptr_model, False)
            state = cats.getStimulus(j)
            action = selection.chooseAction(state)
            reward = cats.getOutcome(state, action)
            selection.updateValue(reward)
    selection.state = convertStimulus(np.array(selection.state))
    selection.action = convertAction(np.array(selection.action))
    selection.responses = np.array(selection.responses)

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001        # variance of evolution noise v
var_obs = 0.05      # variance of observation noise n
gamma = 0.9         # discount factor
init_cov = 10       # initialisation of covariance matrice
kappa = 0.1         # unscentered transform parameters
beta = 1.7          # soft-max
sigma = 0.02        # reward rate update
tau = 0.08          # time step
noise_width = 0.01  # noise of model-based
w_0 = 0.5           # initial weigh for collins model

correlation = "Z"
length_memory = 10

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]


cats = CATS()

selection = CSelection(KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
                       BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0),
                       w_0)

inter = 6
# -----------------------------------

# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------
opt = Optimization(human, cats, nb_trials, nb_blocs)

data = np.zeros((inter,inter,inter,inter,inter))
values = dict()
fall = dict()
p = selection.getAllParameters()
for k in p.iterkeys():
    values[k] = np.linspace(p[k][0], p[k][2], inter)

count = 0
for i in xrange(len(values['beta'])):
    selection.free.beta = values['beta'][i]
    for j in xrange(len(values['gamma'])):
        selection.free.gamma = values['gamma'][j]
        for k in xrange(len(values['lenght'])):
            selection.free.length_memory = values['lenght'][k]
            for l in xrange(len(values['noise'])):
                selection.free.noise = values['noise'][l]
                for m in xrange(len(values['w0'])):
                    selection.sigma = values['w0'][m]
                    count+=1; print str(count)+" | "+str(inter**5)
                    testModel()                    
                    fall = extractStimulusPresentation2(selection.responses, selection.state, selection.action, selection.responses)
                    data[i,j,k,l,m] = opt.computeCorrelation(fall, correlation)
                        
data_to_save = dict()  
data_to_save['data'] = data
data_to_save['values'] = values
data_to_save['order'] = ['beta', 'gamma', 'lenght', 'noise', 'w0']
output = open("../../../Dropbox/ISIR/Plot/datagrid_Collins_"+str(datetime.datetime.now()).replace(" ", "_"), 'wb')
pickle.dump(data_to_save, output)
output.close()

