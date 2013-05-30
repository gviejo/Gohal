#!/usr/bin/python
# encoding: utf-8
"""
KelmanQLearning.py

Implementent Kelman Temporal Difference version Q-learning from 
Kalman Temporal Difference: the deterministic case
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
sys.path.append("../../src")
import os
from copy import deepcopy
from optparse import OptionParser
import numpy as np
from fonctions import *
from pylab import plot, figure, show, subplot, legend
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
    
def iterationStep(state, values, covariance, rewards, display = False):
    #prediction step
    covariance['cov'][:,:] = covariance['cov'][:,:] + covariance['noise']
    covariance['noise'] = eta*covariance['cov']
    
    #choose best action
    action = getBestActionSoftMax(state, values, beta)
    next_state = transitionRules(state, action)
    reward = rewards[0][rewards[(state, action)]]
    print (state, action, next_state, reward)

    #sigma points computation
    sigma_points, weights = computeSigmaPoints(values[0], covariance['cov'], kappa)
    rewards_predicted = (sigma_points[:,values[(state, action)]]-gamma*np.max(sigma_points[:,values[next_state]], 1)).reshape(len(sigma_points), 1)

    #compute statistics of interest
    reward_predicted = np.dot(rewards_predicted.flatten(), weights.flatten())
    cov_values_rewards = np.sum(weights*(sigma_points-values[0])*(rewards_predicted-reward_predicted), 0)
    cov_rewards = np.sum(weights*(rewards_predicted-reward_predicted)**2) + var_obs

    #correction step
    kalman_gain = cov_values_rewards/cov_rewards
    values[0] = values[0] + kalman_gain*(reward-reward_predicted)
    covariance['cov'][:,:] = covariance['cov'][:,:] - kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards*kalman_gain

    print 'sigma_points'
    print sigma_points
    print 'rewards_predicted ',rewards_predicted.flatten()
    print 'reward_predicted ',reward_predicted
    print 'c_vr', cov_values_rewards
    print 'c_r', cov_rewards
    print 'kalman_gain', kalman_gain
    print 'delta ', kalman_gain*(reward-reward_predicted)
    print 'delta cov'
    print cov_values_rewards.reshape(len(cov_values_rewards), 1)*cov_rewards*cov_values_rewards
    print 'cov '
    print covariance['cov']
    if display == True:
        displayQValues(states, actions, values)
    
    return next_state
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001 # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
beta = 1       # rate of exploration
gamma = 0.95     # discount factor
kappa = 0.1
init_cov = 0.9   # initialisation of covariance matrice

nb_iter_mod = 1000
deval_mod_time = 40
nb_iter_ext = 3000
deval_ext_time = 240
states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)
# -----------------------------------

# -----------------------------------
# Moderate Training + devaluation
# -----------------------------------
state = 's0'
values_mod = createQValuesDict(states, actions)
covariance_mod = createCovarianceDict(len(states)*len(actions), init_cov, eta)
data = []
data.append(values_mod[0])
#Predevaluation training
print 'Predevaluation training'
rewards[0][rewards[('s1','em')]] = 1.0
for i in xrange(deval_mod_time-1):
    state = iterationStep(state, values_mod, covariance_mod, rewards, display=True)
    data.append(values_mod[0])
#Devaluation
print 'Devaluation'
rewards[0][rewards[('s1','em')]] = -1.0
state = iterationStep(state, values_mod, covariance_mod, rewards, display=True)
data.append(values_mod[0])
#Test in extinction
print 'Extinction'
rewards[0][rewards[('s1','em')]] = 0.0
for i in xrange(deval_mod_time+1, nb_iter_mod):
    state = iterationStep(state, values_mod, covariance_mod, rewards, display=True)
    data.append(values_mod[0])
# -----------------------------------

# -----------------------------------
# Extensive Training + devaluation
# -----------------------------------
state = 's0'
values_ext = createQValuesDict(states, actions)
covariance_ext = createCovarianceDict(len(states)*len(actions), init_cov, eta)
data2 = []
data2.append(values_ext[0])
# Predevaluation training
print 'Predevaluation training'
rewards[0][rewards[('s1','em')]] = 1.0
for i in range(deval_ext_time-1):
    state = iterationStep(state, values_ext, covariance_ext, rewards, display=True)
    data2.append(values_ext[0])
#Devaluation
print 'Devaluation'
rewards[0][rewards[('s1','em')]] = -1.0
state = iterationStep(state, values_ext, covariance_ext, rewards, display=True)
data2.append(values_ext[0])
#Test in extinction
print 'Extinction'
rewards[0][rewards[('s1','em')]] = 0.0
for i in xrange(deval_mod_time+1, nb_iter_mod):
    state = iterationStep(state, values_ext, covariance_ext, rewards, display=True)
    data2.append(values_ext[0])

# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
figure()
subplot(211)
data = np.array(data)
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









