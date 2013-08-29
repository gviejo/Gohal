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
from Selection import KSelection
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
human = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.12     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.0    
noise_width = 0.17
length_memory = 3.0
sigma = 0.02
tau = 0.08

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]
#nb_blocs = 46

cats = CATS(nb_trials)

selection = KSelection(KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
                       BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0),
                       sigma, tau)
                       

opt = Optimization(human, cats, nb_trials, nb_blocs)

data = dict()
data['pcr'] = dict()
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------

testModel(selection)
selection.state = convertStimulus(np.array(selection.state))
selection.action = convertAction(np.array(selection.action))
selection.responses = np.array(selection.responses)
data['pcr']['keramati'] = extractStimulusPresentation(selection.responses, selection.state, selection.action, selection.responses) 
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
subplot(211)
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

subplot(212)
m = data['pcr']['keramati']['mean']
s = data['pcr']['keramati']['sem']
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
title('B. Keramati selection')


subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86, hspace = 0.35)

#fig1.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig2.pdf', bbox_inches='tight')
show()
