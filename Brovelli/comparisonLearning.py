#!/usr/bin/python
# encoding: utf-8
"""
HumanLearning.py

scripts to load and analyze data from Brovelli

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../Fonctions")
from fonctions import *
sys.path.append("../Plot")
from plot import plotCATS
sys.path.append("../LearningAnalysis")
from LearningAnalysis import SSLearning
import scipy.io
from ColorAssociationTasks import CATS
from pylab import *
from scipy import stats
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------    
def QLearning(iteration, values, display = True):    
    #display current stimulus
    state = cats.getStimulus(iteration)

    #choose best action 
    action = getBestActionSoftMax(state, values, beta)
    
    # set response + get outcome
    reward = cats.getOutcome(state, action, iteration)
    answer.append((reward==1)*1)

    # QLearning
    delta = reward + gamma*np.max(values[0][values[state]]) - values[0][values[(state,action)]]
    values[0][values[(state, action)]] = values[0][values[(state, action)]] + alpha*delta

    if display == True:
        print (state, action, reward)
        print cats.correct
        displayQValues(cats.states, cats.actions, values)
        print '\n'

def KelmanQlearning(iteration, values, covariance, display = False):
    #prediction step
    covariance['noise'] = covariance['cov']*eta
    covariance['cov'][:,:] = covariance['cov'][:,:] + covariance['noise']

    #display current stimulus
    state = cats.getStimulus(iteration)
        
    #choose best action
    action = getBestActionSoftMax(state, values, beta, 0)

    # set response + get outcome
    reward = cats.getOutcome(state, action, iteration)
    answer.append((reward==1)*1)

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
gamma = 0.9      # discount factor
alpha = 0.8
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_trials = 42
nStepEm = 2000
pOutset = 0.2
cats = CATS()
Responses = dict()


# -----------------------------------
#Kelman Learning session
# -----------------------------------
Kdata = []
for i in xrange(200):
    answer = []
    values = createQValuesDict(cats.states, cats.actions)
    covariance = createCovarianceDict(len(cats.states)*len(cats.actions), init_cov, eta)
    cats.reinitialize(nb_trials, 'meg')
    for j in xrange(nb_trials):
        KelmanQlearning(j, values, covariance, False)
    Kdata.append(list(answer))
Responses['K'] = np.array(Kdata)
# -----------------------------------

# -----------------------------------
# QLearning session
# -----------------------------------
Qdata = []
for i in xrange(200):
    answer = []
    values = createQValuesDict(cats.states, cats.actions)
    cats.reinitialize(nb_trials, 'meg')
    for j in xrange(nb_trials):
        QLearning(j, values, False)
    Qdata.append(list(answer))
Responses['Q'] = np.array(Qdata)
# -----------------------------------

# -----------------------------------
# Loading human data
# -----------------------------------
data = loadDirectoryEEG(options.input)
Hdata = []
for i in data.iterkeys():
    for j in xrange(1,5):
        Hdata.append(data[i][j]['sar'][0:nb_trials,2])
Responses['H'] = np.array(Hdata)
# -----------------------------------

# -----------------------------------
#fitting state space model
# -----------------------------------
p = dict()
ss = SSLearning(nb_trials, pOutset)
for l in Responses.iterkeys():
    p[l] = []
    for r in Responses[l]:
        ss.runAnalysis(r, nStepEm)
        p[l].append(ss.pmode)
    p[l] = np.array(p[l])
# -----------------------------------

# -----------------------------------
# Index of performance
# -----------------------------------
Iperf = dict()
for l in p.iterkeys():
    Iperf[l] = np.log2(p[l]/pOutset)

# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
ticks_size = 15
legend_size = 15
title_size = 22
label_size = 19

lab = dict({'H':'Human Learning','Q':'Q-Learning','K':'Kelman Learning'})
col = dict({'H':'r','Q':'b','K':'g'})
figure()
rc('legend',**{'fontsize':legend_size})
tick_params(labelsize = ticks_size)

subplot(211)
for i in Responses.iterkeys():
    m = np.mean(Responses[i], 0)
    s = stats.sem(Responses[i], 0)
    plot(m, col[i], label = lab[i], linewidth = 2)
    fill_between(range(nb_trials), m+s, m-s, alpha = 0.2)

axvline(3, 0, 1)
axvline(9, 0, 1)
axvline(15, 0, 1)
title("Accuracy", fontsize = title_size)
xlabel("Trials", fontsize = label_size)
ylabel("%", fontsize  = label_size)
legend()
ylim(0,1)
grid()

subplot(212)
for i in p.iterkeys():
    m = np.mean(p[i], 0)
    s = stats.sem(p[i], 0)
    plot(m, col[i], label = lab[i], linewidth = 2)
    fill_between(range(nb_trials), m+s, m-s, alpha = 0.2)

axvline(3, 0, 1)
axvline(9, 0, 1)
axvline(15, 0, 1)
title("Probability of attaining the goal", fontsize = title_size)
xlabel("Trials", fontsize = label_size)
ylabel("p", fontsize = label_size)
legend()
grid()
ylim(0, 1)


figure()
m = np.mean(Iperf['Q'], 0)
v = np.sqrt(np.var(Iperf['Q'], 0))
plot(m, linewidth = 2, label = 'QLearning')
#fill_between(range(nb_trials), m+v/2., m-v/2., alpha = 0.1) 
m = np.mean(Iperf['K'], 0)
v = np.sqrt(np.var(Iperf['H'], 0))
plot(m, linewidth = 2, label = 'KelmanLearning')
#fill_between(range(nb_trials), m+v/2., m-v/2., alpha = 0.1) 
m = np.mean(Iperf['H'], 0)
v = np.sqrt(np.var(Iperf['H'], 0))
plot(m, linewidth = 2, label = 'Human')
#fill_between(range(nb_trials), m+v/2., m-v/2., alpha = 0.1) 
ylabel('Performance')
xlabel('Trials')
legend()


show()

# -----------------------------------
