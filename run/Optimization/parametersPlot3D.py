#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

scripts to load and plot parameters

run parameterTest.py -i data_model_date -m 'model'

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
parser.add_option("-s", "--subject", action="store", help="Which subject to plot \n Ex : -s S1", default=False)
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
               'bwm_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bwm_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold)
               })
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
parameters = p['p_order']
n_search = p['search']
subject = p['subject']
n_parameters = len(parameters)
fname = p['fname']
X = p['opt']

if fname == 'minimize':
    fun = p['max']        
elif fname == 'fmin':
    X = np.reshape(X, (len(subject), n_search, n_parameters))
elif fname == 'brute':
    X = []
    fun = []
    for s in xrange(len(subject)):
        grid_ = np.transpose(np.reshape(p['grid'][s], (3, p['grid'][s].shape[1]*p['grid'][s].shape[2]*p['grid'][s].shape[3])))
        llh = p['grid_fun'][s].flatten()
        threshold = np.min(llh)+0.1*(np.max(llh)-np.min(llh))/2                        
        X.append(grid_[llh<threshold])
        fun.append(llh[llh<threshold])
        #X.append(grid_)
        #fun.append(llh)
    X = np.array(X)
    fun = np.array(fun)
else:
    print "scipy function not specified\n"
    sys.exit()
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          


figure(figsize = (14, 9))
ion()
for i in xrange(n_parameters):
    subplot(n_parameters, 1, i+1)
    if options.subject:
        ind = subject.index(options.subject)
        plot(X[ind][:,i], fun[ind], 'o')        
    else :
        for j in xrange(len(subject)):
            c = np.random.rand(3,)
            plot(X[j,:,i], fun[j], 'o', color = fun[j], markersize = 5)
    xlabel(parameters[i])      
    xlim(p['parameters'][parameters[i]][0], p['parameters'][parameters[i]][2])
    ylabel("Likelihood")
    grid()
subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)

fig = figure(figsize = (14, 9))
ax = fig.add_subplot(111, projection = '3d')
if options.subject:
    ind = subject.index(options.subject)
    ax.scatter(X[ind][:,0], X[ind][:,1], X[ind][:,2], c = -fun[ind])
    ax.set_xlabel(parameters[0])
    ax.set_ylabel(parameters[1])
    ax.set_zlabel(parameters[2])
    if fname == 'brute':
        ax.scatter(p['opt'][ind][0], p['opt'][ind][1], p['opt'][ind][2], s = 1000, color = 'green')
        ax.set_xlim(np.min(p['grid'][ind][0]),np.max(p['grid'][ind][0]))
        ax.set_ylim(np.min(p['grid'][ind][1]),np.max(p['grid'][ind][1]))
        ax.set_zlim(np.min(p['grid'][ind][2]),np.max(p['grid'][ind][2]))        
else:
    ax.scatter(X[ind][:,0], X[ind][:,1], X[ind][:,2], c = fun[ind])
    ax.set_xlabel(parameters[0])
    ax.set_ylabel(parameters[1])
    ax.set_zlabel(parameters[2])

print p['opt'][ind][1]
print p['opt'][ind][2]
print p['opt'][ind][0]


show()        




