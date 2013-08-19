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
sys.path.append("../../src")
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
def iterationStep(iteration, models, display = True):
    state = cats.getStimulus(iteration)
    for m in models.itervalues():
        action = m.chooseAction(state)
        reward = cats.getOutcome(state, action, m.name)
        m.updateValue(reward)


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
gamma = 0.63     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.7      
noise_width = 0.0106
length_memory = 10

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
subplot(311)
m = data['pcr']['meg']['mean']
s = data['pcr']['meg']['sem']
for i in xrange(3):
    errorbar(range(1, len(m[i])+1), m[i], s[i], linestyle = dashes[i], color = 'black')
    plot(range(1, len(m[i])+1), m[i], linestyle = dashes[i], color = 'black',linewidth = 2, label = 'Stim '+str(i+1))
grid()
legend(loc = 'lower right')
xticks(range(2,11,2))
xlabel("Trial")
xlim(0.8, 10.2)
ylim(-0.05, 1.05)
yticks(np.arange(0, 1.2, 0.2))
ylabel('Probability Correct Responses')
title('A. MEG')

subplot(312)
m = data['pcr']['bmw']['mean']
s = data['pcr']['bmw']['sem']
for i in xrange(3):
    errorbar(range(1, len(m[i])+1), m[i], s[i], linestyle = dashes[i], color = 'black')
    plot(range(1, len(m[i])+1), m[i], linestyle = dashes[i], color = 'black',linewidth = 2, label = 'Stim '+str(i+1))
grid()
legend(loc = 'lower right')
xticks(range(2,11,2))
xlabel("Trial")
xlim(0.8, 10.2)
ylim(-0.05, 1.05)
yticks(np.arange(0, 1.2, 0.2))
ylabel('Probability Correct Responses')
title('B. Bayesian Working Memory')

subplot(313)
m = data['pcr']['kalman']['mean']
s = data['pcr']['kalman']['sem']
for i in xrange(3):
    errorbar(range(1, len(m[i])+1), m[i], s[i], linestyle = dashes[i], color = 'black')
    plot(range(1, len(m[i])+1), m[i], linestyle = dashes[i], color = 'black',linewidth = 2, label = 'Stim '+str(i+1))
grid()
legend(loc = 'lower right')
xticks(range(2,11,2))
xlabel("Trial")
xlim(0.8, 10.2)
ylim(-0.05, 1.05)
yticks(np.arange(0, 1.2, 0.2))
ylabel('Probability Correct Responses')
title('C. Kalman Q-Learning')

subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86, hspace = 0.35)

fig1.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig2.pdf', bbox_inches='tight')
#show()
