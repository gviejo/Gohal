#!/usr/bin/python
# encoding: utf-8
"""
exp1.py

Reproduces the first experiment of Keramati & al, 2011
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
sys.path.append("../../src")
import os
from optparse import OptionParser
import numpy as np
from fonctions import *
from Selection import Keramati
from Models import KalmanQLearning
from pylab import plot, figure, show, subplot, legend, ylim, axvline

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-o", "--output", action="store", help="The name of the output file to store the data", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def transitionRules(state, action):
    if state == 's0' and action == 'pl':
        return 's1'
    else:
        return 's0'
    
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
beta = 1.0       # rate of exploration
gamma = 0.95     # discount factor
sigma = 0.02     # updating rate of the average reward
rau = 0.1        # update rate of the reward function
tau = 0.08       # time step for graph exploration

phi = 0.5        # update rate of the transition function
depth = 3        # depth of search when computing the goal value
init_cov = 1.0   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_iter_test = 100

nb_iter_mod = 100
deval_mod_time = 40
nb_iter_ext = 350
deval_ext_time = 240

states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)

# -----------------------------------
# Moderate Training + devaluation
# -----------------------------------
kalman = KalmanQLearning("moderate", states, actions, gamma, beta, eta, var_obs, init_cov, kappa)
selection = Keramati(kalman, depth, phi, rau, sigma, tau)
kalman.initialize()
state = 's0'

#Predevaluation training
rewards[0][rewards[('s1','em')]] = 1.0
for i in range(deval_mod_time-1):
    action = selection.chooseAction(state)
    selection.updateValues(rewards[0][rewards[(state, action)]])
    state = transitionRules(state, action)
#Devaluation
rewards[0][rewards[('s1','em')]] = -1.0
for i in range(2):
    action = selection.chooseAction(state)
    selection.updateValues(rewards[0][rewards[(state, action)]])
    state = transitionRules(state, action)
#Test in extinction
rewards[0][rewards[('s1','em')]] = 0.0
for i in range(deval_mod_time+1, nb_iter_mod):
    action = selection.chooseAction(state)
    selection.updateValues(rewards[0][rewards[(state, action)]])
    state = transitionRules(state, action)

"""
# -----------------------------------
# Saving
# -----------------------------------\

for i in data.iterkeys():
    data[i] = np.array(data[i])
    data2[i] = np.array(data2[i])

d = dict({'data':data,'data2':data2})
saveData(options.output, d)

"""
# -----------------------------------
# Plot
# -----------------------------------\
colors = {('s0','pl'):'green',('s0','em'):'red',('s1','pl'):'cyan',('s1','em'):'purple'}
figure()
subplot(521)
#for s in states:
for s in ['s0']:
    for a in actions:
        plot(data['vpi'][:,values_mod[(s,a)]], 'o-', color = colors[(s,a)], label = "VPI("+s+","+a+")")
plot(data['r'], 'o-', color = 'blue', label = "R*tau")
axvline(deval_mod_time-1, color='black')
legend()
ylim(0,0.1)
subplot(522)
#for s in states:
for s in ['s0']:
    for a in actions:
        plot(data2['vpi'][:,values_ext[(s,a)]], 'o-', color = colors[(s,a)], label = "VPI("+s+","+a+")")
plot(data2['r'], 'o-', color = 'blue', label = "R*tau")
axvline(deval_ext_time-1, color='black')
legend()
ylim(0,0.1)
subplot(523)
for s in ['s0']:
    for a in actions:
        plot(data['p'][:,values_mod[(s,a)]], 'o-', color = colors[(s,a)], label = "p("+s+","+a)
axvline(deval_mod_time-1, color='black')
ylim(0.3,0.7)
legend()
subplot(524)
for s in ['s0']:
    for a in actions:
        plot(data2['p'][:,values_ext[(s,a)]], 'o-', color = colors[(s,a)], label = "p("+s+","+a)
axvline(deval_ext_time-1, color='black')
ylim(0.3,0.7)
legend()
subplot(525)
plot((np.mean(data['vpi'], 1)-data['r'])>0, color = 'blue', label = 'deliberation time')
axvline(deval_mod_time-1, color='black')
ylim(0, 1.5)
legend()
subplot(526)
plot((np.mean(data2['vpi'], 1)-data2['r'])>0, color = 'blue', label = 'deliberation time')
axvline(deval_ext_time-1, color='black')
ylim(0, 1.5)
legend()
subplot(527)
plot(data['h'][:,0]-data['h'][:,1])
axvline(deval_mod_time-1, color='black')
ylim(0,0.5)
subplot(528)
plot(data2['h'][:,0]-data2['h'][:,1])
axvline(deval_ext_time-1, color='black')
ylim(0,0.5)

show()
# -----------------------------------








