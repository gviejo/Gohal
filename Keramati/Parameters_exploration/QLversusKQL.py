#!/usr/bin/python
# encoding: utf-8
"""
QLversusKQL.py

Compare QL Learning and KQLearning in the evolution of Qvalues 
An arbitrary sequence of state action is generated
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from copy import deepcopy
from optparse import OptionParser
import numpy as np
sys.path.append("../")
from fonctions import *
from pylab import plot, figure, show, subplot, legend,ylim
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
# FONCTIONS
# -----------------------------------
def transitionRules(state, action):
    if state == 's0' and action == 'pl':
        return 's1'
    else:
        return 's0'

def QLearning(state, action, reward, values, rewards, display = False):
    next_state = transitionRules(state, action)
    delta = reward + gamma*np.max(values[0][values[next_state]]) - values[0][values[(state,action)]]
    values[0][values[(state, action)]] = values[0][values[(state, action)]] + alpha*delta
    if display <> True:
        displayQValues(states, actions, values)
    
def KelmanQLearning(state, action, reward, values, covariance, rewards, display = False):
    next_state = transitionRules(state, action)    
    #prediction step
    covariance[:,:] = covariance[:,:] + var_evo
    #sigma points computation
    sigma_points, weights = computeSigmaPoints(values[0], covariance)
    rewards_predicted = (sigma_points[:,values[(state, action)]]-gamma*np.max(sigma_points[:,values[next_state]], 1)).reshape(len(sigma_points), 1)
    #compute statistics of interest
    reward_predicted = np.dot(rewards_predicted.flatten(), weights.flatten())
    cov_values_rewards = np.sum(weights*(sigma_points-values[0])*(rewards_predicted-reward_predicted), 0)
    cov_rewards = np.sum(weights*(rewards_predicted-reward_predicted)**2) + var_obs
    #correction step
    kalman_gain = cov_values_rewards/cov_rewards
    values[0] = values[0] + kalman_gain*(reward-reward_predicted)
    covariance['cov'][:,:] = covariance['cov'][:,:] - cov_values_rewards.reshape(len(cov_values_rewards), 1)*cov_rewards*cov_values_rewards

    if display <> True:
        displayQValues(states, actions, values)
    


# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
var_evo = 0.0001 # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
beta = 1       # rate of exploration
gamma = 0.95     # discount factor
init_cov = 0.0001   # initialisation of covariance matrice
alpha = 0.1

niteration = 100
states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)
rewards[0][rewards[('s1','em')]] = 1.0
# -----------------------------------
# Sequence generation
# -----------------------------------
#from QValues
state = 's0'
values_Q = createQValuesDict(states, actions)
values_K = createQValuesDict(states, actions)
covariance = np.eye(len(states)*len(actions), len(states)*len(actions))*init_cov
data = dict({'K':[], 'Q':[]})
data['K'].append(values_K[0].copy())
data['Q'].append(values_Q[0].copy())
for i in xrange(niteration):
    action = getBestActionSoftMax(state, values_Q, beta)
    reward = rewards[0][rewards[(state, action)]]
    KelmanQLearning(state, action, reward, values_K, covariance, rewards, True)
    QLearning(state, action, reward, values_Q, rewards, True)
    data['K'].append(values_K[0].copy())
    data['Q'].append(values_Q[0].copy())
    state = transitionRules(state, action)
#from Kvalues
state = 's0'
values_Q = createQValuesDict(states, actions)
values_K = createQValuesDict(states, actions)
covariance = np.eye(len(states)*len(actions), len(states)*len(actions))*init_cov
data2 = dict({'K':[], 'Q':[]})
data2['K'].append(values_K[0].copy())
data2['Q'].append(values_Q[0].copy())
for i in xrange(niteration):
    action = getBestActionSoftMax(state, values_K, beta)
    reward = rewards[0][rewards[(state, action)]]
    KelmanQLearning(state, action, reward, values_K, covariance, rewards, True)
    QLearning(state, action, reward, values_Q, rewards, True)
    data2['K'].append(values_K[0].copy())
    data2['Q'].append(values_Q[0].copy())
    state = transitionRules(state, action)


# -----------------------------------
# Plot
# -----------------------------------
for i in data.iterkeys():
    data[i] = np.array(data[i])
    data2[i] = np.array(data2[i])
colors = {('s0','pl'):'green',('s0','em'):'red',('s1','pl'):'cyan',('s1','em'):'purple'}

figure()
subplot(211)
for s in states:
    for a in actions:
        plot(data['Q'][:,values_Q[(s,a)]], 'o-', color = colors[(s,a)], label = "Q("+s+","+a+")")
        #plot(data['K'][:,values_K[(s,a)]], '+-', color = colors[(s,a)], label = "K("+s+","+a+")")
legend()
subplot(212)
for s in states:
    for a in actions:
        #plot(data2['Q'][:,values_Q[(s,a)]], 'o-', color = colors[(s,a)], label = "Q("+s+","+a+")")
        plot(data2['K'][:,values_K[(s,a)]], '+-', color = colors[(s,a)], label = "K("+s+","+a+")")

legend()
show()
# -----------------------------------









