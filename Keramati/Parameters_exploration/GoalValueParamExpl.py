#!/usr/bin/python
# encoding: utf-8
"""
exp1.py

scripts to explore the goal directed value computation
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
    
    #calculate values for softmaxx from current state
    reward_rate['rate'] = updateRewardRate(reward_rate, sigma)                
    reward_rate['tau'] = computeExplorationCost(tau, depth, len(states), len(actions))
    vpi = computeVPIValues(values[0][values[state]], covariance['cov'].diagonal()[values[state]])

    #for i in range(len(vpi)):
        #if vpi[i] >= reward_rate['rate']*reward_rate['tau']:
            #values[1][values[state][i]] = computeGoalValue(values, state, values[(state, i)], rewards, gamma, depth, phi, rau)
        #else:
            #values[1][values_mod[state][i]] = values[0][values_mod[state][i]]
    for s in states:
        for a in actions:
            values[1][values[(s, a)]] = computeGoalValue(values, s, a, rewards, gamma, depth, phi, rau)
    
    
    #display tables
    if display <> True:
        displayQValues(states, actions, values, 1)
        
    #choose best action
    action = getBestActionSoftMax(state, values, beta, 1)
    next_state = transitionRules(state, action)
    reward = rewards[0][rewards[(state, action)]]
    reward_rate['reward'] = reward
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

    #if  state == 's1' and action == 'em':
    if next_state == 's0':
        return None
    else:
        return next_state            
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

phi = 0.1        # update rate of the transition function
depth = 3        # depth of search when computing the goal value
init_cov = 1.1     # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_iter_test = 500
nb_iter_mod = 300
deval_mod_time = 240

states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)

# -----------------------------------
# Training
# -----------------------------------
values_mod = createQValuesDict(states, actions)
values_mod[1] = np.zeros((len(states)*len(actions)))
covariance_mod = createCovarianceDict(len(states)*len(actions), init_cov, eta)
reward_rate_mod = createRewardRateDict()
data = {}
data['g'] = [values_mod[1].copy()]
data['h'] = [values_mod[0].copy()]
data['p'] = [testQValues(states, values_mod, beta, 1, nb_iter_test).copy()]
data['r'] = [reward_rate_mod['rate']*reward_rate_mod['tau']]
#Predevaluation training
rewards[0][rewards[('s1','em')]] = 1.0
for i in range(deval_mod_time-1):
    state = 's0'
    data['g'].append(values_mod[1].copy())
    data['h'].append(values_mod[0].copy())
    data['p'].append(testQValues(states, values_mod, beta, 1, nb_iter_test).copy())
    data['r'].append(reward_rate_mod['rate']*reward_rate_mod['tau'])
    #while state <> None:
    for j in xrange(2):
        if state <> None:
            state = iterationStep(state, values_mod, covariance_mod, rewards, reward_rate_mod, display=True)
        else: 
            break
#Devaluation
rewards[0][rewards[('s1','em')]] = -1.0
state = 's0'
data['g'].append(values_mod[1].copy())
data['h'].append(values_mod[0].copy())
data['p'].append(testQValues(states, values_mod, beta, 1, nb_iter_test).copy())
data['r'].append(reward_rate_mod['rate']*reward_rate_mod['tau'])
#while state <> None:
for j in xrange(2):
    if state <> None:
        state = iterationStep(state, values_mod, covariance_mod, rewards, reward_rate_mod, display=True)
    else: 
        break
#Test in extinction
rewards[0][rewards[('s1','em')]] = 0.0
for i in range(deval_mod_time, nb_iter_mod):
    state = 's0'
    data['g'].append(values_mod[1].copy())
    data['h'].append(values_mod[0].copy())
    data['p'].append(testQValues(states, values_mod, beta, 1, nb_iter_test).copy())
    data['r'].append(reward_rate_mod['rate']*reward_rate_mod['tau'])
    #while state <> None:
    for j in xrange(2):
        if state <> None:
            state = iterationStep(state, values_mod, covariance_mod, rewards, reward_rate_mod, display=True)
        else: 
            break

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
        plot(data['p'][:,values_mod[(s,a)]], 'o-', color = colors[(s,a)], label = "p("+s+","+a)
axvline(deval_mod_time-1, color='black')
ylim(0.3,0.7)
legend()
subplot(312)
plot(data['h'][:,0]-data['h'][:,1])
axvline(deval_mod_time-1, color='black')
ylim(0,0.5)
subplot(313)
for s in ['s0']:
    for a in actions:
        plot(data['g'][:,values_mod[(s,a)]], 'o-',  color = colors[(s,a)], label = "G("+s+","+a+")")
        plot(data['h'][:,values_mod[(s,a)]], '+-',  color = colors[(s,a)], label = "H("+s+","+a+")")
axvline(deval_mod_time-1, color='black')
legend()
show()
# -----------------------------------

figure()
plot(data['r'])
ylim(0, 0.1)
show()







