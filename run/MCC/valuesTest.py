#!/usr/bin/python
# encoding: utf-8
"""
New model selection for Brovelli task based on Keramati
Modification are :
      - reward rate depends on the stimulus
      - threshold for the number of inferences in the bayesian working memory
      
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
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

def testModels():
    for m in models.itervalues():
        m.initializeList()
        for i in xrange(nb_blocs):
            sys.stdout.write("\r Testing model | Blocs : %i" % i); sys.stdout.flush()                    
            cats.reinitialize()
            m.initialize()
            for j in xrange(nb_trials):
                state = cats.getStimulus(j)
                action = m.chooseAction(state)
                reward = cats.getOutcome(state, action)
                m.updateValue(reward)
        m.state = convertStimulus(np.array(m.state))
        m.action = convertAction(np.array(m.action))
        m.responses = np.array(m.responses)
        m.value = np.array(m.value)
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
gamma = 0.6        # discount factor
init_cov = 10       # initialisation of covariance matrice
kappa = 0.1         # unscentered transform parameters
beta = 1.5          # temperature for kalman soft-max
noise_width = 0.0  # variance of white noise for working memory
length_memory = 9  # size of working memory
threshold = 1.3     # entropy threshold

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]


cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, threshold)})

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
testModels()
# -----------------------------------

# -----------------------------------
#order data
# -----------------------------------
data = dict()

for m in models.iterkeys():
    data[m] = dict({i:dict() for i in [1,2,3]})
    state = extractStimulusPresentation2(models[m].state, models[m].state, models[m].action, models[m].responses)
    rs, ind = getRepresentativeSteps(models[m].action, models[m].state, models[m].action, models[m].responses)
    data[m][1] = dict({j:list() for j in [1,2,3]})
    data[m][2] = dict({j:list() for j in xrange(1,6)})
    data[m][3] = dict({j:list() for j in xrange(1,6)})
    for i in xrange(nb_blocs):
        ###################
        stim1 = state[1][i,0]
        tmp = np.transpose(models[m].value[i][models[m].state[i] == stim1])
        a1 = models[m].action[i][models[m].state[i] == stim1][ind[i][models[m].state[i] == stim1] == 1][0]
        a2 = models[m].action[i][models[m].state[i] == stim1][ind[i][models[m].state[i] == stim1] == 2][0]
        data[m][1][1].append(tmp[a1-1])
        data[m][1][2].append(tmp[a2-1])
        for j in set(range(1,6))-set([a1,a2]):
            data[m][1][3].append(tmp[j-1])
        ###################
        stim2 = state[2][i,0]
        tmp = np.transpose(models[m].value[i][models[m].state[i] == stim2])
        tmp2 = []
        for j in xrange(1,5):
            a = models[m].action[i][models[m].state[i] == stim2][ind[i][models[m].state[i] == stim2] == j][0]            
            data[m][2][j].append(tmp[a-1])
            tmp2.append(a)
        for j in set(range(1,6))-set(tmp2):
            data[m][2][5].append(tmp[j-1])
        ###################
        stim3 = state[3][i,0]
        tmp = np.transpose(models[m].value[i][models[m].state[i] == stim3])
        for j in xrange(1,6):
            a = models[m].action[i][models[m].state[i] == stim3][ind[i][models[m].state[i] == stim3] == j][0]         
            data[m][3][j].append(tmp[a-1])
    for i in data[m].iterkeys():
        for j in data[m][i].iterkeys():
            data[m][i][j] = np.array(data[m][i][j])[:,0:10]


# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
ion()
fig = figure(figsize=(14, 8))
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':10,
          'ytick.labelsize':10,
          'text.usetex':False}
dashes = ['-', '--', ':']
rcParams.update(params)

for i in [1,2,3]:
    subplot(2,3,i)
    for j in data['bmw'][i].iterkeys():
        m = np.mean(data['bmw'][i][j], axis = 0)
        v = np.var(data['bmw'][i][j], axis = 0)
        errorbar(range(1, len(m)+1), m, v, linestyle = 'o-', linewidth = 2)
        plot(range(1, len(m)+1), m, linewidth = 2)
    grid()
    ylim(0,1)
    
for i in [1,2,3]:
    subplot(2,3,i+3)
    for j in data['kalman'][i].iterkeys():
        m = np.mean(data['kalman'][i][j], axis = 0)
        v = np.var(data['kalman'][i][j], axis = 0)
        errorbar(range(1, len(m)+1), m, v, linestyle = 'o-', linewidth = 2)
        plot(range(1, len(m)+1), m, linewidth = 2)
    grid()
    ylim(0,1)


show()    


