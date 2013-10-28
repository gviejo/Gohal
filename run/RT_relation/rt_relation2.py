#!/usr/bin/python
# encoding: utf-8
"""
explore relation between reaction time, stimulus and action choice

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Models import BayesianWorkingMemory
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from time import time
from Models import BayesianWorkingMemory

# -----------------------------------
# FONCTIONS
# -----------------------------------
def computeDistanceMatrix(indice, s):
    m, n = indice.shape    
    distance = np.zeros((m,n))
    for i in xrange(m):        
        for j in xrange(n):
            if indice[i, j] > 1:
                distance[i,j] = j-np.where(s[i,0:j] == s[i,j])[0][-1]

    return distance

def testModel():
    bww.initializeList()
    for i in xrange(nb_blocs):
        sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                    
        cats.reinitialize()
        bww.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)

    bww.state = convertStimulus(np.array(bww.state))
    bww.action = convertAction(np.array(bww.action))
    bww.responses = np.array(bww.responses)
    bww.reaction = np.array(bww.reaction)
    

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
noise = 0.0
length_memory = 10
#threshold = 1.2
threshold = 1.0
case = 'meg'

nb_trials = 42
nb_blocs = 100
cats = CATS(nb_trials)

bww = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise, threshold)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
testModel()

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------


state = human.stimulus[case]
action = human.action[case]
responses = human.responses[case]
reaction = human.reaction[case]

hstep, hindice = getRepresentativeSteps(reaction, state, action, responses)
mstep, mindice = getRepresentativeSteps(bww.reaction, bww.state, bww.action, bww.responses)

rt_human = computeMeanRepresentativeSteps(hstep) 

distance_human = computeDistanceMatrix(hindice, state)
distance_model = computeDistanceMatrix(mindice, bww.state)
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (8, 4))
ion()
######################################
# Plot 1 : mean RT vs distance between correct

mean_plot_h = np.array([np.mean(reaction[np.where((distance_human == i) & (hindice > 5))]) for i in xrange(1, int(np.max(distance_human))+1)])
var_plot_h = np.array([sem(reaction[np.where((distance_human == i) & (hindice > 5))]) for i in xrange(1, int(np.max(distance_human))+1)])
mean_plot_m = np.array([np.mean(bww.reaction[np.where((distance_model == i) & (mindice > 5))]) for i in xrange(1, int(np.max(distance_model))+1)])
var_plot_m = np.array([sem(bww.reaction[np.where((distance_model == i) & (mindice > 5))]) for i in xrange(1, int(np.max(distance_model))+1)])
subplot(1,1,1)
plot(range(1, len(mean_plot_h)+1), mean_plot_h, linewidth = 3, linestyle = '-', color = 'blue')
errorbar(range(1, len(mean_plot_h)+1), mean_plot_h, var_plot_h, linewidth = 3, linestyle = '-', color = 'blue')    
ylim(0.3, 1.0)
ylabel("Reaction time (ms)")
twinx()
plot(range(1, len(mean_plot_m)+1), mean_plot_m, linewidth = 3, linestyle = '-', color = 'blue')
errorbar(range(1, len(mean_plot_m)+1), mean_plot_m, var_plot_m, linewidth = 3, linestyle = '-', color = 'blue')    
grid()
xlim(0, np.max(distance_human)+2)
xlabel("Distance")



show()