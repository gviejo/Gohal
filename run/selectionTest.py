#!/usr/bin/python
# encoding: utf-8
"""

to test keramati model

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

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------

testModel(selection)
selection.state = convertStimulus(np.array(selection.state))
selection.action = convertAction(np.array(selection.action))
selection.responses = np.array(selection.responses)
selection.rrate = np.array(selection.rrate)
data['keramati'] = extractStimulusPresentation(selection.responses, selection.state, selection.action, selection.responses) 
data['r'] = extractStimulusPresentation(selection.rrate[:,0:42], selection.state, selection.action, selection.responses)
# -----------------------------------
data['vpi'] = dict()
for s in cats.states:
    selection.vpi[s] = np.array(selection.vpi[s])[:,0:10,:]
    data['vpi'][s] = np.mean(selection.vpi[s], 0)

# -----------------------------------
#order data
# -----------------------------------
data['meg'] = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

fig = figure(figsize=(14, 5))
params = {'backend':'pdf',
          'axes.labelsize':9,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}
dashes = ['-', '--', ':']

for i in xrange(3):
    subplot(2,3,i+1)
    plot(range(1, len(data['keramati']['mean'][i])+1), data['keramati']['mean'][i], linewidth = 2, color = 'black')
    errorbar(range(1, len(data['keramati']['mean'][i])+1), data['keramati']['mean'][i], data['keramati']['sem'][i], linewidth = 2, color = 'black')
    plot(range(1, len(data['meg']['mean'][i])+1), data['meg']['mean'][i], linewidth = 2, color = 'black', linestyle = '--')
    errorbar(range(1, len(data['meg']['mean'][i])+1), data['meg']['mean'][i], data['meg']['sem'][i], linewidth = 2, color = 'black', linestyle = '--')
    legend()
    grid()
    title("Stimulus "+str(i+1))

for i,j in zip([4,5,6], xrange(3)):
    subplot(2,3,i)
    plot(data['r']['mean'][j])
    plot(data['vpi'][cats.states[j]])
    grid()

subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86, hspace = 0.35)

#fig1.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig2.pdf', bbox_inches='tight')
show()
