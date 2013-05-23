#!/usr/bin/python
# encoding: utf-8
"""
KLearning.py

performs a Kelman Qlearning over CATS task

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../Plot")
from plot import plotCATS
sys.path.append("../LearningAnalysis")
from LearningAnalysis import SSLearning
sys.path.append("../Fonctions")
from fonctions import *
import scipy.io as sio
from ColorAssociationTasks import CATS
from pylab import *
from scipy import stats

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument => -i reaction_time_.mat")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="Load reaction time data 'reaction_time.dat'", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------    
def KelmanQlearning(iteration, values, covariance, reward_rate, display = False):
    #prediction step
    covariance['noise'] = covariance['cov']*eta
    covariance['cov'][:,:] = covariance['cov'][:,:] + covariance['noise']

    #compute VPI values over all possible state
    vpi.append(np.array([computeVPIValues(values[0][values[s]], covariance['cov'].diagonal()[values[s]]) for s in cats.states]).flatten())
            
    #display current stimulus
    state = cats.getStimulus(iteration)
        
    #choose best action
    action = getBestActionSoftMax(state, values, beta, 0)

    # set response + get outcome
    reward = cats.getOutcome(state, action, iteration)
    answer.append((reward==1)*1)

    #update reward rate
    best_action = getBestAction(state, values, 0)
    if action == best_action:
        reward_rate['rate'] = (1-sigma)*reward_rate['rate']+sigma*reward
    reward_rate['tau'] = np.random.normal(reaction['mean'][iteration], reaction['sem'][iteration])
    reward_rate['r_list'].append(reward_rate['rate'])
    reward_rate['r*tau'].append(reward_rate['tau']*reward_rate['rate'])

    #sigma points computation
    sigma_points, weights = computeSigmaPoints(values[0], covariance['cov'], kappa)
    rewards_predicted = (sigma_points[:,values[(state, action)]]-gamma*np.max(sigma_points[:,values[state]], 1)).reshape(len(sigma_points), 1)

    #compute statistics of interest
    reward_predicted = np.dot(rewards_predicted.flatten(), weights.flatten())
    cov_values_rewards = np.sum(weights*(sigma_points-values[0])*(rewards_predicted-reward_predicted), 0)
    cov_rewards = np.sum(weights*(rewards_predicted-reward_predicted)**2) + var_obs

    #correction step
    kalman_gain = cov_values_rewards/cov_rewards
    values[0] = values[0] + kalman_gain*(reward-reward_predicted)
    covariance['cov'][:,:] = covariance['cov'][:,:] - (kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards)*kalman_gain

    if display == True:
        print (state, action, reward)
        print 'R predicted', reward_predicted
        print cats.correct
        displayQValues(cats.states, cats.actions, values)
        print '\n'

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
beta = 3.0       # rate of exploration
gamma = 0.9     # discount factor
sigma = 0.02    # updating rate of the average reward
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_trials = 40
nb_blocs = 100
cats = CATS()
case = 'meg'
# -----------------------------------

# -----------------------------------
# Loading human reaction time
# -----------------------------------
reaction = sio.loadmat(options.input)
# -----------------------------------

# -----------------------------------
#Kelman Learning session
# -----------------------------------
blocvpi = dict({0:[],1:[],2:[],-1:dict({0:[], 1:[], 2:[]})})
rate = []
ratetau = []
responses = []
for i in xrange(nb_blocs):
    vpi = []
    answer = []
    values = createQValuesDict(cats.states, cats.actions)
    reward_rate = createRewardRateDict()
    covariance = createCovarianceDict(len(cats.states)*len(cats.actions), init_cov, eta)
    cats.reinitialize(nb_trials, case)
    for j in xrange(nb_trials):
         KelmanQlearning(j, values, covariance, reward_rate, True)         
    responses.append(np.array(answer))
    ratetau.append(np.array(reward_rate['r*tau']))
    rate.append(np.array(reward_rate['r_list'][1:]))
    vpi = np.array(vpi)
    for j in xrange(len(cats.correct)):
        state = cats.correct[j].split(" => ")[0]; action = cats.correct[j].split(" => ")[1]
        tmp = list(values[state]); tmp.pop(tmp.index(values[(state, action)]))
        blocvpi[j].append(vpi[:,values[(state, action)]])
responses = np.array(responses)
rate = np.array(rate)
ratetau = np.array(rate)
for k in xrange(len(cats.states)):
    blocvpi[k] = np.transpose(np.array(blocvpi[k]))
# -----------------------------------



# -----------------------------------
#Plot 
# -----------------------------------
ticks_size = 15
legend_size = 15
title_size = 22
label_size = 19

figure()
rc('legend',**{'fontsize':legend_size})
tick_params(labelsize = ticks_size)

subplot(212)
for k in xrange(3):
    m = np.mean(blocvpi[k], 1)
    s = stats.sem(blocvpi[k], 1)
    plot(m, linewidth = 2, label = 'VPI('+str(k)+')')    
    fill_between(range(nb_trials), m+s, m-s, alpha = 0.2)
m = np.mean(ratetau, 0)
s = stats.sem(ratetau, 0)
plot(m, 'black', linewidth = 2, label = 'R*tau')
fill_between(range(nb_trials), m+s, m-s, alpha = 0.2)
axvline(3, 0, 1)
axvline(9, 0, 1)
axvline(15, 0, 1)
title('Value of Precise Information', fontsize = label_size)
xlabel('trials', fontsize = label_size)

grid()
legend()

subplot(211)
m = reaction['mean'][0:nb_trials].flatten()
s = reaction['sem'][0:nb_trials].flatten()
errorbar(range(nb_trials), m, s, label = 'reaction time (tau)')
#m = np.mean(rate, 0)
#s = stats.sem(rate, 0)
#errorbar(range(nb_trials), m, s, label = 'Rate')
axvline(3, 0, 1)
axvline(9, 0, 1)
axvline(15, 0, 1)
ylabel('(s)', fontsize = label_size)
title('Reaction time', fontsize = label_size)
grid()
legend()
ylim(0.3, 0.7)

show()
