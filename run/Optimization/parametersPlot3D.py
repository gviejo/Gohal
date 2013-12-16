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
parser.add_option("-t", "--threshold", action="store", help="Threshold of likelihood \n Ex : -t 0.1", default=False)
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
alpha = 0.5

nb_trials = human.responses['meg'].shape[1]
#nb_blocs = human.responses['meg'].shape[0]
nb_blocs = 40
cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bwm_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold),
               'qlearning':QLearning('q', cats.states, cats.actions, gamma, alpha, beta)
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

#p['parameters']['threshold'][2] = 1.1

if fname == 'minimize':
    fun = p['max']        
elif fname == 'fmin':
    ind = subject.index(options.subject)
    flag = p['warnflag'][ind]
    X = p['opt'][ind][flag == 0]
    fun = -p['max'][ind][flag == 0]
    for i in xrange(len(p['p_order'])):
        bounds = np.array([p['parameters'][p['p_order'][i]][0],p['parameters'][p['p_order'][i]][2]])
        X = X[np.where((X[:,i]>bounds[0])&(X[:,i]<bounds[1]))]
        fun = fun[np.where((X[:,i]>bounds[0])&(X[:,i]<bounds[1]))]
    if len(X) < 5:
        X = p['opt'][ind]
        fun = -p['max'][ind]        

elif fname == 'brute':
    ind = subject.index(options.subject)
    #X = np.transpose(np.reshape(p['grid'][ind], (3, p['grid'][ind].shape[1]*p['grid'][ind].shape[2]*p['grid'][ind].shape[3])))
    #X = np.transpose(np.vstack((p['grid'][ind][0].flatten(), p['grid'][ind][1].flatten(), p['grid'][ind][2].flatten())))
    #fun = p['grid_fun'][ind].flatten()
    # CHeck if nan in fun
    #fun = fun[np.logical_not(np.isnan(fun))]
    #X = X[np.logical_not(np.isnan(fun))]    
    #threshold = np.min(fun)+float(options.threshold)*(np.max(fun)-np.min(fun))/2
    X = []
    fun = []
    for i in xrange(p['grid'][ind].shape[1]):
        for j in xrange(p['grid'][ind].shape[2]):
            for k in xrange(p['grid'][ind].shape[3]):                
                X.append(p['grid'][ind][:,i,j,k])
                fun.append(p['grid_fun'][ind][i,j,k])
    X = np.array(X)
    fun = np.array(fun)
    threshold = np.min(fun)+float(options.threshold)*(np.max(fun)-np.min(fun))

    fun = fun[fun<threshold]    
    X = X[fun<threshold]  
    popt = p['opt'][ind]  
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

ion()


fig1 = figure(figsize = (14, 9))
for i in xrange(n_parameters):
    subplot(n_parameters, 1, i+1)        
    plot(X[:,i], fun, 'o')            
    if fname == 'brute':
        axvline(popt[i], linewidth = 4, color = 'green')
    xlabel(parameters[i])      
    xlim(p['parameters'][parameters[i]][0], p['parameters'][parameters[i]][2])
    ylabel("Likelihood")
    grid()
subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)



fig2 = figure(figsize = (14, 9))
ax = fig2.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c = -fun)#, s=(-fun+np.max(fun))/np.min(fun)*10)
ax.set_xlabel(parameters[0])
ax.set_ylabel(parameters[1])
ax.set_zlabel(parameters[2])
ax.set_xlim(p['parameters'][parameters[0]][0], p['parameters'][parameters[0]][2])
ax.set_ylim(p['parameters'][parameters[1]][0], p['parameters'][parameters[1]][2])
ax.set_zlim(p['parameters'][parameters[2]][0], p['parameters'][parameters[2]][2])
if fname == 'brute':
    ax.scatter(popt[0], popt[1], popt[2], s = 1000, color = 'green')


if fname == 'fmin':
    print parameters[1], np.mean(X, 0)[1]    
    print parameters[0], np.mean(X, 0)[0]
    print parameters[2], np.mean(X, 0)[2]

    fig3 = figure(figsize = (14,9))
    for i in xrange(n_parameters):
        subplot(n_parameters,1,i+1)
        hist(X[:,i],30)
        xlim(p['parameters'][parameters[i]][0], p['parameters'][parameters[i]][2])
        xlabel(parameters[i])
        grid()

if fname == 'brute':
    print popt[1]
    print popt[2]
    print popt[0]
    if "bwm" in options.model and p['p_order'].index('lenght') == 1:
        fig3 = figure(figsize = (14, 9))
        value = np.unique(p['grid'][ind][1])
        length = np.unique(value.astype(int))
        n = int(np.ceil(np.sqrt(len(length))))
        extents = [np.min(p['grid'][ind][0]), np.max(p['grid'][ind][0]), np.min(p['grid'][ind][2]), np.max(p['grid'][ind][2])]
        for i in xrange(len(length)):
            subplot(n,n,i+1)
            t = np.where(value.astype(int)==length[i])[0][0]
            imshow(-p['grid_fun'][ind][:,t,:], origin = 'lowest', extent=extents, aspect = 'auto')
            title(str(length[i]))
        subplots_adjust(hspace=0.4,wspace = 0.4)        



show()        




