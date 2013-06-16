#!/usr/bin/python
# encoding: utf-8
"""
modelsEvaluation.py

evaluation of all models on ColorAssociationTask.py
       -> Human Behaviour : <MEG>      <fMRI>
       -> ModelBased :      <No-noise> <Noise>
       -> KalmanQlearning : <Slow>     <Fast>
       -> QLearning :       <Slow>     <Fast>

For each model => Time Reaction + Accuracy
see 'Differential roles of caudate nucleus and putamen during instrumental learning.
     Brovelli & al, 2011'

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
from pylab import *
sys.path.append("../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import QLearning
from Models import KalmanQLearning
from Models import TreeConstruction
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
def iterationStep(iteration, qlearning, klearning, display = True):
    print iteration
    #get state
    state = cats.getStimulus(iteration)

    #choose best Action
    q_action = qlearning.chooseAction(state)
    k_action = klearning.chooseAction(state)
    t_action = tlearning.chooseAction(tlearning.g[state])

    #set reward 
    q_reward = cats.getOutcome(state, q_action)
    k_reward = cats.getOutcome(state, k_action)
    t_reward = cats.getOutcome(state, t_action)

    #update models
    qlearning.update(q_reward)
    klearning.update(k_reward)
    tlearning.updateTrees(state, t_reward)
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
gamma = 0.1 #discount facto
alpha = 1
beta = 1
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.9     # discount factor
sigma = 0.02    # updating rate of the average reward
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_trials = 42
nb_blocs = 100

cats = CATS()
qlearning = QLearning(cats.states, cats.actions, gamma, alpha, beta)
klearning = KalmanQLearning(cats.states, cats.actions, gamma, beta, eta, var_obs, sigma, init_cov, kappa)
tlearning = TreeConstruction(cats.states, cats.actions, alpha, beta, gamma)
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
for i in xrange(nb_blocs):
    cats.reinitialize(nb_trials, 'meg')
    for j in xrange(nb_trials):
        iterationStep(j, qlearning, klearning, False)
    qlearning.reinitialize()
    klearning.reinitialize()
    tlearning.reinitialize()
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../fMRI',39)}))
steps1, indice = getRepresentativeSteps(human.reaction['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
steps2, indice = getRepresentativeSteps(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
# -----------------------------------



# -----------------------------------
# Plot
# -----------------------------------
#HUMAN

#subplot(111)
#m,s = computeMeanRepresentativeSteps(steps1)
#errorbar(range(len(m)), m, s, linewidth = 2)
#xlabel('representative steps')
#ylabel('reaction time (s)')
#grid()

#show()











