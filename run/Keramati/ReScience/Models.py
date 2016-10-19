#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models.py
Class of all models
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *

class KalmanQLearning():
    """ Class that implement a KalmanQLearning : 
    Kalman Temporal Differences : The deterministic case, Geist & al, 2009
    """

    def __init__(self, name, states, actions, gamma, beta, eta, var_obs, init_cov, kappa):
        self.name=name
        self.gamma=gamma;self.beta=beta;self.eta=eta;self.var_obs=var_obs;self.init_cov=init_cov;self.kappa=kappa

        self.values = createQValuesDict(states, actions)
        self.covariance = createCovarianceDict(len(states)*len(actions), self.init_cov, self.eta)
        self.states = states
        self.actions = actions
        self.action = list()
        self.state = list()
        self.responses = list()
        self.reaction = list()
        
    def getParameter(self, name):
        if name == 'gamma':
            return self.gamma
        elif name == 'beta':
            return self.beta
        elif name == 'eta':
            return self.eta
        else:
            print "Parameters not found"
            sys.exit(0)     

    def setParameter(self, name, value):
        if name == 'gamma':
            self.gamma = value
        elif name == 'beta':
            self.beta = value
        elif name == 'eta':
            self.eta = value
        else:
            print "Parameters not found"
            sys.exit(0)

    def initialize(self):
        self.responses.append([])
        self.values = createQValuesDict(self.states, self.actions)
        self.covariance = createCovarianceDict(len(self.states)*len(self.actions), self.init_cov, self.eta)
        self.action.append([])
        self.state.append([])
        self.reaction.append([])

    def initializeList(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.covariance = createCovarianceDict(len(self.states)*len(self.actions), self.init_cov, self.eta)
        self.action = list()
        self.state = list()
        self.responses = list()
        self.reaction = list()

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.covariance['noise'] = self.covariance['cov']*self.eta
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']
        self.action[-1].append(getBestActionSoftMax(state, self.values, self.beta, 0))
        self.reaction[-1].append(computeEntropy(self.values[0][self.values[state]], self.beta))
        return self.action[-1][-1]

    def updateValue(self, reward):
        self.responses[-1].append((reward==1)*1)
        s = self.state[-1][-1]; a = self.action[-1][-1]
        sigma_points, weights = computeSigmaPoints(self.values[0], self.covariance['cov'], self.kappa)
        rewards_predicted = (sigma_points[:,self.values[(s,a)]]-self.gamma*np.max(sigma_points[:,self.values[s]], 1)).reshape(len(sigma_points), 1)
        reward_predicted = np.dot(rewards_predicted.flatten(), weights.flatten())
        cov_values_rewards = np.sum(weights*(sigma_points-self.values[0])*(rewards_predicted-reward_predicted), 0)
        cov_rewards = np.sum(weights*(rewards_predicted-reward_predicted)**2) + self.var_obs
        kalman_gain = cov_values_rewards/cov_rewards
        self.values[0] = self.values[0] + kalman_gain*(reward-reward_predicted)
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] - (kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards)*kalman_gain
