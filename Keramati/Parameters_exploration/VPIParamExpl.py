#!/usr/bin/python
# encoding: utf-8
"""
exp1.py

Exploration of the parameters from the 
first experiment of Keramati & al, 2011
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../")
from fonctions import *
from pylab import plot, figure, show, subplot, legend, ylim, axvline

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
    
def iterationStep(state, values, covariance, rewards, reward_rate, display = False):
    #prediction step
    covariance['noise'] = covariance['cov']*eta
    covariance['cov'][:,:] = covariance['cov'][:,:] + covariance['noise']

    #display tables
    if display <> True:
        displayQValues(states, actions, values, 1)
        
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
    covariance['cov'][:,:] = covariance['cov'][:,:] - (kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards)*kalman_gain

    return next_state            
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
beta = 1.0       # rate of exploration
gamma = 0.95     # discount factor

init_cov = 1   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

ntrials = 300

states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)
vpi = np.zeros((len(states)*len(actions)))


# -----------------------------------
# Training
# -----------------------------------
state = 's0'
values = createQValuesDict(states, actions)
covariance = createCovarianceDict(len(states)*len(actions), init_cov, eta)
reward_rate = []
data = {}
data['h'] = [values[0].copy()]
data['vpi'] = [computeVPIValues(values[0][values['s0']], covariance['cov'].diagonal()[values['s0']])]
              
rewards[0][rewards[('s1','em')]] = 1.0

for i in xrange(ntrials):
    state = 's0'
    data['vpi'].append(computeVPIValues(values[0][values['s0']], covariance['cov'].diagonal()[values['s0']]))
    data['h'].append(values[0].copy())
    for j in xrange(2):
        state = iterationStep(state, values, covariance, rewards, reward_rate, display=True)

# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------\
for i in data.iterkeys():
    data[i] = np.array(data[i])

colors = {('s0','pl'):'green',('s0','em'):'red',('s1','pl'):'cyan',('s1','em'):'purple'}
figure()
subplot(311)
for s in ['s0']:
    for a in actions:
        plot(data['vpi'][:,values[(s,a)]], 'o-', color = colors[(s,a)], label = "VPI("+s+","+a+")")
legend()
ylim(0,0.1)
subplot(312)
for s in states:
    for a in actions:
        plot(data['h'][:,values[(s,a)]], 'o-', color = colors[(s,a)], label = "Q("+s+","+a)
legend()
subplot(313)
plot(data['h'][:,0]-data['h'][:,1])
ylim(0,0.5)
show()

# -----------------------------------









