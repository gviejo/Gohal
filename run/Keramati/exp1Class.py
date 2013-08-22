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

nb_iter_test = 500

nb_iter_mod = 100
deval_mod_time = 40
nb_iter_ext = 300
deval_ext_time = 240

states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)
data = dict({'mod':dict({'vpi':list(),
                         'r':list(),
                         'p':list(),
                         'q':list()}),
             'ext':dict({'vpi':list(),
                         'r':list(),
                         'p':list(),
                         'q':list()})})
# -----------------------------------
# Training + devaluation
# -----------------------------------
kalman = KalmanQLearning("", states, actions, gamma, beta, eta, var_obs, init_cov, kappa)
selection = Keramati(kalman, depth, phi, rau, sigma, tau)

for exp, nb_trials, deval_time in zip(['mod','ext'], [nb_iter_mod, nb_iter_ext], [deval_mod_time, deval_ext_time]):
    kalman.initialize()
    selection.initialize()
    state = 's0'
    rewards[0][rewards[('s1','em')]] = 1.0
    print exp, nb_trials, deval_time
    for i in xrange(nb_trials):
        print "TRIALS :", i
        #Setting Reward
        if i == deval_time:
            rewards[0][rewards[('s1','em')]] = 0.0
            selection.rfunction[0][selection.rfunction[('s1', 'em')]] = -1.0
        #Learning
        while True:
            action = selection.chooseAction(state)        
            next_state = transitionRules(state, action)
            selection.updateValues(rewards[0][rewards[(state, action)]], next_state)
            if state == 's1' and action == 'em':
                #Retrieving data
                data[exp]['vpi'].append(computeVPIValues(kalman.values[0][kalman.values['s0']],kalman.covariance['cov'].diagonal()[kalman.values['s0']]))
                data[exp]['r'].append(selection.rrate[-1])                                                       
                data[exp]['p'].append(testQValues(states, selection.values, kalman.beta, 0, nb_iter_test))
                data[exp]['q'].append(kalman.values[0][kalman.values[('s0','pl')]]-kalman.values[0][kalman.values[('s0','em')]])                
                state = next_state
                break
            else:
                state = next_state
        
    data[exp]['vpi'] = np.array(data[exp]['vpi'])
    data[exp]['r'] = np.array(data[exp]['r'])*selection.tau
    data[exp]['p'] = np.array(data[exp]['p'])
    data[exp]['q'] = np.array(data[exp]['q'])
# -----------------------------------
# Plot
# -----------------------------------\
colors = {('s0','pl'):'green',('s0','em'):'red',('s1','pl'):'cyan',('s1','em'):'purple'}
fig = figure()
subplot(321)
for s in ['s0']:
    for a in actions:
        plot(data['mod']['vpi'][:,selection.values[(s,a)]], 'o-', color = colors[(s,a)], label = "VPI("+s+","+a+")")
plot(data['mod']['r'], 'o-', color = 'blue', label = "R*tau")
axvline(deval_mod_time-1, color='black')
legend()
ylim(0,0.1)

subplot(323)
for s in ['s0']:
    for a in actions:
        plot(data['mod']['p'][:,selection.values[(s,a)]], 'o-', color = colors[(s,a)], label = "p("+s+","+a)
axvline(deval_mod_time-1, color='black')
ylim(0.3,0.7)
legend()

subplot(325)
for s in ['s0']:
    for a in actions:
        plot(data['mod']['q'], 'o-', color = colors[(s,a)], label = "Q("+s+","+a)
axvline(deval_mod_time-1, color='black')
ylim(0,0.5)

subplot(322)
for s in ['s0']:
    for a in actions:
        plot(data['ext']['vpi'][:,selection.values[(s,a)]], 'o-', color = colors[(s,a)], label = "VPI("+s+","+a+")")
plot(data['ext']['r'], 'o-', color = 'blue', label = "R*tau")
axvline(deval_ext_time-1, color='black')
legend()
ylim(0,0.1)


subplot(324)
for s in ['s0']:
    for a in actions:
        plot(data['ext']['p'][:,selection.values[(s,a)]], 'o-', color = colors[(s,a)], label = "p("+s+","+a)
axvline(deval_ext_time-1, color='black')
ylim(0.3,0.7)
legend()

subplot(326)
for s in ['s0']:
    for a in actions:
        plot(data['ext']['q'], 'o-', color = colors[(s,a)], label = "Q("+s+","+a)
axvline(deval_ext_time-1, color='black')
ylim(0,0.5)
legend()


show()
# -----------------------------------








