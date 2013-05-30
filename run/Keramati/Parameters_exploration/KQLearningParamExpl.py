#!/usr/bin/python
# encoding: utf-8
"""
KQLearningParamsExpl.py

Scripts to explore the parameters of Kelman TD learning
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from copy import deepcopy
from optparse import OptionParser
import numpy as np
sys.path.append("../")
from fonctions import *
from pylab import plot, figure, show, subplot, legend, ylim
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
eta = 0.0001    # variance of evolution noise v
var_obs = 0.05  # variance of observation noise n
beta = 1        # rate of exploration
gamma = 0.95    # discount factor
kappa = 0.5
init_cov = 0.1  # initialisation of covariance matrice

niterations = 100
states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)
# -----------------------------------

# -----------------------------------
# 
# -----------------------------------
state = 's0'
values = createQValuesDict(states, actions)
covariance = dict({'cov':np.eye(len(states)*len(actions), len(states)*len(actions))*init_cov,
                   'noise':np.eye(len(states)*len(actions), len(states)*len(actions))*init_cov*eta})
data = dict({'val':[],'cov':[],'vpi':[]})
data['val'].append(values[0].copy())
data['cov'].append(covariance['cov'].diagonal())
data['vpi'].append(list(computeVPIValues(values[0][values['s0']],covariance['cov'].diagonal()[values['s0']]))+list(computeVPIValues(values[0][values['s1']],covariance['cov'].diagonal()[values['s1']])))

print 'Predevaluation training'
rewards[0][rewards[('s1','em')]] = 1.0
for i in xrange(niterations):
    state = iterationStep(state, values, covariance, rewards, display=True)
    data['val'].append(values[0].copy())
    data['cov'].append(covariance['cov'].diagonal())
    data['vpi'].append(list(computeVPIValues(values[0][values['s0']],covariance['cov'].diagonal()[values['s0']]))+list(computeVPIValues(values[0][values['s1']],covariance['cov'].diagonal()[values['s1']])))

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
for i in data.iterkeys():
    data[i] = np.array(data[i])
colors = {('s0','pl'):'green',('s0','em'):'red',('s1','pl'):'cyan',('s1','em'):'purple'}

figure()
subplot(411)
for s in states:
    for a in actions:
        plot(data['val'][:,values[(s,a)]], '-', color = colors[(s,a)], label = "Q("+s+","+a+")")
legend()

subplot(412)
for s in states:
    for a in actions:
        plot(data['cov'][:,values[(s,a)]], '-', color = colors[(s,a)], label = "cov("+s+","+a+")")
legend()

subplot(413)
plot(data['val'][:,values[('s0','pl')]]-data['val'][:,values[('s0','em')]], '-', color = colors[('s0','pl')], label = "Q(s0,pl)-Q(s0,em)")
plot(data['val'][:,values[('s1','em')]]-data['val'][:,values[('s1','pl')]], '-', color = colors[('s1','em')], label = "Q(s1,em)-Q(s1,pl)")
legend()

subplot(414)
for s in states:
    for a in actions:
        plot(data['vpi'][:,values[(s,a)]], '-', color = colors[(s,a)], label = "vpi("+s+","+a+")")
legend()
show()
# -----------------------------------






