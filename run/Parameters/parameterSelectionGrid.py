#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

Grid-search for Selection model of Keramati
Kalman : beta, gamma
Bayesian : length, noise
Keramati : sigma, tau


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
from Selection import KSelection
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization
from mpl_toolkits.mplot3d import Axes3D
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
def testModel(ptr_model):
    ptr_model.initializeList()
    for i in xrange(nb_blocs):
        #sys.stdout.write("\r Testing model | Blocs : %i" % i); sys.stdout.flush()                        
        cats.reinitialize()
        ptr_model.initialize()
        for j in xrange(nb_trials):
            opt.iterationStep(j, ptr_model, False)

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


correlation = "Z"
length_memory = 15

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]


cats = CATS()

selection = KSelection(KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
                       BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0),
                       sigma, tau)

inter = 10
# -----------------------------------

# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------
opt = Optimization(human, cats, nb_trials, nb_blocs)

data = np.zeros((inter,inter,inter,inter,inter,inter))
values = dict()
fall = dict()
p = selection.getAllParameters()
for k in p.iterkeys():
    values[k] = np.linspace(p[k][0], p[k][2], inter)

count = 0
for i in xrange(len(values['beta'])):
    selection.kalman.beta = values['beta'][i]
    for j in xrange(len(values['gamma'])):
        selection.kalman.gamma = values['gamma'][j]
        for k in xrange(len(values['lenght'])):
            selection.kalman.length_memory = values['lenght'][k]
            for l in xrange(len(values['noise'])):
                selection.kalman.noise = values['noise'][l]
                for m in xrange(len(values['sigma'])):
                    selection.sigma = values['sigma'][m]
                    for n in xrange(len(values['tau'])):
                        selection.tau = values['tau'][n]
                        count+=1; print str(count)+" | "+str(inter**6)
                        testModel(selection)
                        selection.state = convertStimulus(np.array(selection.state))
                        selection.action = convertAction(np.array(selection.action))
                        selection.responses = np.array(selection.responses)

                        fall = extractStimulusPresentation2(selection.responses, selection.state, selection.action, selection.responses)
                        data[i,j,k,l,m,n] = opt.computeCorrelation(fall, correlation)

        
data['values'] = values
data['order'] = ['beta', 'gamma', 'lenght', 'noise', 'sigma', 'tau']
output = open("../../../Dropbox/ISIR/Plot/datagrid_Keramati_"+str(datetime.datetime.now()).replace(" ", "_"), 'wb')
pickle.dump(data, output)
output.close()

