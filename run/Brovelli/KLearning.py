#!/usr/bin/python
# encoding: utf-8
"""
KLearning.py

performs a Kalman Qlearning over CATS task
python KLearning

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
sys.path.append("../../src")
import os
from optparse import OptionParser
import numpy as np
from LearningAnalysis import SSLearning
from fonctions import *
import scipy.io as sio
from ColorAssociationTasks import CATS
from pylab import *
from scipy import stats

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument => -i reaction_time_.mat")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="Load reaction time data 'reaction_time.dat'", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------    
def KalmanQlearning(iteration, values, covariance, display = False):
    #prediction step
    covariance['noise'] = covariance['cov']*eta
    covariance['cov'][:,:] = covariance['cov'][:,:] + covariance['noise']
            
    #display current stimulus
    state = cats.getStimulus(iteration)
    stimulus[-1].append(state)
    #choose best action
    action = getBestActionSoftMax(state, values, beta, 0)
    action_list[-1].append(action)
    reaction[-1].append(computeEntropy(values[0][values[state]], beta))

    # set response + get outcome
    reward = cats.getOutcome(state, action)
    responses[-1].append((reward==1)*1)
    
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

responses = []
stimulus = []
action_list = []
reaction = []
# -----------------------------------

# -----------------------------------
#Kalman Learning session
# -----------------------------------

for i in xrange(nb_blocs):
    values = createQValuesDict(cats.states, cats.actions)
    covariance = createCovarianceDict(len(cats.states)*len(cats.actions), init_cov, eta)
    cats.reinitialize()
    action_list.append([])
    stimulus.append([])
    responses.append([])
    reaction.append([])
    for j in xrange(nb_trials):
        KalmanQlearning(j, values, covariance, True)         
        

responses = np.array(responses)
stimulus = convertStimulus(np.array(stimulus))
action = convertAction(np.array(action_list))
reaction = np.array(reaction)

# -----------------------------------

data = extractStimulusPresentation(responses, stimulus, action, responses)
step, indice = getRepresentativeSteps(reaction, stimulus, action, responses)


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

subplot(211)

for m, s in zip(data['mean'],data['sem']):
    errorbar(range(1, len(m)+1), m, s)
grid()
xlim(0,16)
title("Performance")

m, s = computeMeanRepresentativeSteps(step)

subplot(212)
errorbar(range(1, len(m)+1), m, s)
grid()
title("Entropy")

show()
