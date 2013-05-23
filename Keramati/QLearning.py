#!/usr/bin/python
# encoding: utf-8
"""
KelmanQLearning.py

Implementent simple Q-learning with the experiemnt from Keramati & al, 2011
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from copy import deepcopy
from optparse import OptionParser
import numpy as np
from fonctions import *
from pylab import plot, figure, show, subplot, legend, xlim
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
#parser.add_option("-o", "--output", action="store", help="The name of the output file to store the data", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
gamma = 0.95 #discount facto
alpha = 0.1
beta = 1

nb_iter_mod = 100
deval_mod_time = 40
nb_iter_ext = 300
deval_ext_time = 240

states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)

# -----------------------------------
# FONCTIONS
# -----------------------------------
def transitionRules(state, action):
    if state == 's0' and action == 'pl':
        return 's1'
    else:
        return 's0'
    
def iterationStep(state, values, rewards, display = False):
    #choose best action
    action = getBestActionSoftMax(state, values, beta)
    next_state = transitionRules(state, action)
    reward = rewards[0][rewards[(state, action)]]
    #print (state, action, next_state, reward)

    delta = reward + gamma*np.max(values[0][values[next_state]]) - values[0][values[(state,action)]]
    values[0][values[(state, action)]] = values[0][values[(state, action)]] + alpha*delta

    if display == True:
        displayQValues(states, actions, values)
    
    return next_state
# -----------------------------------

# -----------------------------------
# Moderate Training + devaluation
# -----------------------------------
state = 's0'
values_mod = createQValuesDict(states, actions)
data = np.zeros((nb_iter_mod, len(states)*len(actions)))
data[0,:] = values_mod[0].copy()
#Predevaluation training
print 'Predevaluation training'
rewards[0][rewards[('s1','em')]] = 1.0
for i in xrange(deval_mod_time-1):
    state = iterationStep(state, values_mod, rewards, display=True)
    data[i,:] = values_mod[0].copy()
#Devaluation
print 'Devaluation'
rewards[0][rewards[('s1','em')]] = -1.0
state = iterationStep(state, values_mod, rewards, display=True)
data[deval_mod_time-1] = values_mod[0].copy()
#Test in extinction
print 'Extinction'
rewards[0][rewards[('s1','em')]] = 0.0
for i in xrange(deval_mod_time, nb_iter_mod):
    state = iterationStep(state, values_mod,  rewards, display=True)
    data[i,:] = values_mod[0].copy()

# -----------------------------------

# -----------------------------------
# Extensive Training + devaluation
# -----------------------------------
state = 's0'
values_ext = createQValuesDict(states, actions)
data2 = np.zeros((nb_iter_ext, len(states)*len(actions)))
data2[0] = values_ext[0].copy()
# Predevaluation training
print 'Predevaluation training'
rewards[0][rewards[('s1','em')]] = 1.0
for i in xrange(deval_ext_time-1):
    state = iterationStep(state, values_ext, rewards, display=True)
    data2[i] = values_ext[0].copy()

#Devaluation
print 'Devaluation'
rewards[0][rewards[('s1','em')]] = -1.0
state = iterationStep(state, values_ext, rewards, display=True)
data2[deval_ext_time-1] = values_ext[0].copy()
#Test in extinction
print 'Extinction'
rewards[0][rewards[('s1','em')]] = 0.0
for i in xrange(deval_ext_time, nb_iter_ext):
    state = iterationStep(state, values_ext, rewards, display=True)
    data2[i] = values_ext[0].copy()

# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
figure()
subplot(211)
for s in states:
    for a in actions:
        plot(data[:,values_mod[(s, a)]], 'o-', label = s+","+a)
legend()

data2 = np.array(data2)

subplot(212)
for s in states:
    for a in actions:
        plot(data2[:,values_ext[(s, a)]], 'o-', label = s+","+a)
legend()

show()
# -----------------------------------









