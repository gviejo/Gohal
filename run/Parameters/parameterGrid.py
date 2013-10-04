#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

Grid-search for Kalman and Bayesian Model
Kalman : beta, gamma
Bayesian : length, noise

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
from ColorAssociationTasks import CATS_MODELS
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
beta = 1.7          # temperature
noise = 0.001
length_memory = 10
threshold = 1.5

inter = 10
correlation = "JSD"

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]

cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
              'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, 15, 0.01, 1.0)})


# -----------------------------------

data = {}
xs = {}
ys = {}
label = {}
# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------
opt = Optimization(human, cats, nb_trials, nb_blocs)
tm = 0
for m in models.iterkeys():
    p = models[m].getAllParameters()    
    #xs[m] = np.arange(p.values()[0][0], p.values()[0][2])
    xs[m] = np.linspace(p.values()[0][0], p.values()[0][2], inter)
    ys[m] = np.linspace(p.values()[1][0], p.values()[1][2], inter)
    label[m] = p.keys()
    

for t in ['Z', 'JSD']:
    data[t] = dict()
    for m in models.iterkeys():
        data[t][m] = np.zeros((len(xs[m]), inter))
        p = models[m].getAllParameters()
        tmp = p
        for i,x in zip(xs[m], xrange(len(xs[m]))):
            tmp[tmp.keys()[0]][1] = i
            for j,y in zip(ys[m], xrange(inter)):
                tm+=1
                print tm," / ", 4*inter*inter
                tmp[tmp.keys()[1]][1] = j
                data[t][m][x,y] = opt.evaluate(models[m], tmp, correlation)
        
data['xs'] = xs
data['ys'] = ys
data['label'] = label
output = open("../../../Dropbox/ISIR/Plot/datagrid_"+str(datetime.datetime.now()).replace(" ", "_"), 'wb')
pickle.dump(data, output)
output.close()


# -----------------------------------
# Plot
# -----------------------------------
ticks_size = 15
legend_size = 15
title_size = 20
label_size = 19

"""
fig1 = figure()
for m, i in zip(models.iterkeys(), [1,2]):
    ax = fig1.add_subplot(1,2,i)
    ax.imshow(data[m])
    xlabel(label[m][0])
    ylabel(label[m][1])    
    title(m)


fig = figure(figsize=plt.figaspect(0.5))

rc('legend',**{'fontsize':legend_size})
tick_params(labelsize = ticks_size)

for m, i in zip(models.iterkeys(), [1,2]):
    ax = fig.add_subplot(1, 2, i, projection='3d')
    X, Y = np.meshgrid(xs[m], ys[m])
    surf = ax.plot_surface(X, Y, data[m], rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    xlabel(label[m][0])
    ylabel(label[m][1])    
    title(m)
    fig.colorbar(surf, shrink=0.5, aspect=10)

"""
