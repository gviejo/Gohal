#!/usr/bin/python
# encoding: utf-8
"""
scripts to plot figure pour le rapport IAD
figure 2 : performances des sujets / perf pour bWM / perf pour Kalman

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
#from pylab import *
sys.path.append("../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization
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
def omegaFunc(cible, freq1, freq2):
    if cible == freq1 == freq2 == 0:
        return 0.0
    elif freq1 == freq2:
        w = float(cible/freq1)
        if w < 0.0:
            return 0.0
        elif w > 1.0:
            return 1.0
        else:
            return w        
    else:
        w = float((cible-freq2)/(freq1-freq2))
        if w < 0.0:
            return 0.0
        elif w > 1.0:
            return 1.0
        else:
            return w


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.34     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 4.0
noise_width = 0.29
length_memory = 12.33

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]
#nb_blocs = 46

cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0)})

opt = Optimization(human, cats, nb_trials, nb_blocs)

data = dict()
data['pcr'] = dict()
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
for m in models.iterkeys():
    opt.testModel(models[m])
    models[m].state = convertStimulus(np.array(models[m].state))
    models[m].action = convertAction(np.array(models[m].action))
    models[m].responses = np.array(models[m].responses)
    data['pcr'][m] = extractStimulusPresentation(models[m].responses, models[m].state, models[m].action, models[m].responses) 
# -----------------------------------



# -----------------------------------
#order data
# -----------------------------------
data['pcr']['meg'] = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
# -----------------------------------

#------------------------------------
# compute Omega
#------------------------------------
data['w'] = np.zeros((3,10))
data['pcr']['w'] = np.zeros((3,10))
for i in xrange(3):
    for j in xrange(10):
        data['w'][i,j] = omegaFunc(data['pcr']['meg']['mean'][i,j],data['pcr']['bmw']['mean'][i,j],data['pcr']['kalman']['mean'][i,j])
        data['pcr']['w'][i,j] = data['w'][i,j]*data['pcr']['bmw']['mean'][i,j]+(1-data['w'][i,j])*data['pcr']['kalman']['mean'][i,j]
# -----------------------------------
# Plot
# -----------------------------------

fig1 = figure(figsize=(5, 9))
params = {'backend':'pdf',
          'axes.labelsize':9,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}
dashes = ['-', '--', ':']
rcParams.update(params)                  
for i,l in zip([1,3,5],xrange(3)):
    subplot(3,2,i)
    for j,k in zip(['bmw','kalman','meg'],dashes):
        plot(data['pcr'][j]['mean'][l], label = j, linewidth = 1.5)
    plot(data['pcr']['w'][l], label = 'w', color = 'grey', linewidth = 3)
    grid()
    legend()
    ylim(0,1)

for i,l in zip([2,4,6],xrange(3)):
    subplot(3,2,i)
    for j,k in zip(['bmw','kalman','meg'],dashes):
        plot(data['w'][l])
    grid()
    legend()
    ylim(0,1)


subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86, hspace = 0.35)

#fig1.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig2.pdf', bbox_inches='tight')
show()
