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

def testModel():
    for i in xrange(nb_blocs):
        sys.stdout.write("\r Testing model | Blocs : %i" % i); sys.stdout.flush()                        
        cats.reinitialize()
        selection.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = selection.chooseAction(state)
            reward = cats.getOutcome(state, action)
            selection.updateValue(reward)
    selection.state = convertStimulus(np.array(selection.state))
    selection.action = convertAction(np.array(selection.action))
    selection.responses = np.array(selection.responses)
    selection.rrate = np.array(selection.rrate)
    selection.vpi = np.array(selection.vpi)
    selection.model_used = np.array(selection.model_used)

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
gamma = 0.95     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.5    
noise_width = 0.05
length_memory = 7
sigma = 0.1
tau = 0.05

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]


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

testModel()

data['keramati'] = extractStimulusPresentation(selection.responses, selection.state, selection.action, selection.responses) 
data['used'] = extractStimulusPresentation(selection.model_used, selection.state, selection.action, selection.responses)
data['r'] = extractStimulusPresentation(selection.rrate, selection.state, selection.action, selection.responses)
# -----------------------------------
data['vpi'] = dict()
for i in xrange(len(cats.actions)):
    data['vpi'][cats.actions[i]] = extractStimulusPresentation(selection.vpi[:,:,i], selection.state, selection.action, selection.responses)


# -----------------------------------
#order data
# -----------------------------------
data['meg'] = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
ion()
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
    subplot(3,3,i+1)
    plot(range(1, len(data['keramati']['mean'][i])+1), data['keramati']['mean'][i], linewidth = 2, color = 'black')
    errorbar(range(1, len(data['keramati']['mean'][i])+1), data['keramati']['mean'][i], data['keramati']['sem'][i], linewidth = 2, color = 'black')
    plot(range(1, len(data['meg']['mean'][i])+1), data['meg']['mean'][i], linewidth = 2, color = 'black', linestyle = '--')
    errorbar(range(1, len(data['meg']['mean'][i])+1), data['meg']['mean'][i], data['meg']['sem'][i], linewidth = 2, color = 'black', linestyle = '--')
    legend()
    grid()
    ylim(0,1)
    xlim(0,10)
    title("Stimulus "+str(i+1))

for i,j in zip([4,5,6], xrange(3)):
    subplot(3,3,i)
    plot(range(1, len(data['r']['mean'][j])+1),data['r']['mean'][j], linewidth = 2, linestyle = '--')
    for a in cats.actions:
        plot(range(1, len(data['vpi'][a]['mean'][j])+1),data['vpi'][a]['mean'][j], linewidth = 2, label = a)
        errorbar(range(1, len(data['vpi'][a]['mean'][j])+1),data['vpi'][a]['mean'][j], data['vpi'][a]['sem'][j], linewidth = 2)
    grid()
    xlim(0,10)
    legend()

for i,j in zip([7,8,9], xrange(3)):
    subplot(3,3,i)
    plot(range(1, len(data['used']['mean'][j])+1),data['used']['mean'][j], 'o-', linewidth = 2)
    grid()
    xlim(0,10)
    legend()
    ylabel("$N^{Based}_a / N^{Free}_a$")



subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86, hspace = 0.35)

#fig1.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig2.pdf', bbox_inches='tight')
show()
