#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

scripts to load and plot parameters

run parameterTest.py -i data_kalman_date

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
   sys.stdout.write("Sorry: you must specify at least 1 argument")
   sys.stdout.write("More help avalaible with -h or --help option")
   sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def testModel():    
    for i in xrange(nb_blocs):
        sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                    
        cats.reinitialize()
        model.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = model.chooseAction(state)
            reward = cats.getOutcome(state, action)
            model.updateValue(reward)
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
gamma = 0.630     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.6666   
noise = 0.01
length_memory = 8
threshold = 1.2

nb_trials = human.responses['meg'].shape[1]
#nb_blocs = human.responses['meg'].shape[0]
nb_blocs = 40
cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm':BayesianWorkingMemory('bwm', cats.states, cats.actions, length_memory, noise, threshold)})

model = models[options.model]

# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
f = open(options.input, 'rb')
p = pickle.load(f)
# -----------------------------------

# -----------------------------------
#order data
# -----------------------------------
data = dict()
n_search = p['search']
subject = p['subject']
n_parameters = len(p['p_order'])
X = p['opt']
X = np.reshape(X, (len(subject), n_search, n_parameters))

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
fig = figure(figsize = (9,4))
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          

if n_parameters == 2:
  for i in xrange(len(X)):
    c = np.random.rand(3,)
    plot(X[i,:,0], X[i,:,1], 'o', color = c, markersize = i+3)
  grid()
elif n_parameters == 3:
  ax = fig.add_subplot(111, projection='3d')
  for i in xrange(len(X)):
    c = np.random.rand(3,)
    ax.scatter(X[i,:,0], X[i,:,1], X[i,:,2], marker ='o', alpha = 1, color = c)
  


show()

