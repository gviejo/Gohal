#!/usr/bin/python
# encoding: utf-8
"""
Models.py

Class of all models

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *

class QLearning():
    """Class that implement a Qlearning
    """
    
    def __init__(self, states, actions, gamma, alpha, beta):
        self.gamma=gamma;self.alpha=alpha;self.beta=beta
        self.values = createQValuesDict(states, actions)
        self.answer = []
        self.responses = []
        self.states = states
        self.actions = actions
        self.action = None
        self.state = None

    def reinitialize(self):
        if len(self.answer) <> 0:
            self.responses.append(list(self.answer))
            self.answer = list()
        self.values = createQValuesDict(self.states, self.actions)
        
    def chooseAction(self, state):
        self.state = state
        self.action = getBestActionSoftMax(state, self.values, self.beta)
        return self.action
    
    def update(self, reward):
        self.answer.append((reward==1)*1)
        delta = reward+self.gamma*np.max(self.values[0][self.values[self.state]])-self.values[0][self.values[(self.state,self.action)]]
        self.values[0][self.values[(self.state,self.action)]] = self.values[0][self.values[(self.state,self.action)]]+self.alpha*delta
    

class KalmanQLearning():
    """ Class that implement a KalmanQLearning : 
    Kalman Temporal Differences : The deterministic case, Geist & al, 2009
    """

    def __init__(self, states, actions, gamma, beta, eta, var_obs, sigma, init_cov, kappa):
        self.gamma=gamma;self.beta=beta;self.eta=eta;self.var_obs=var_obs;self.sigma=sigma;self.init_cov=init_cov;self.kappa=kappa
        self.values = createQValuesDict(states, actions)
        self.reward_rate = createRewardRateDict()
        self.covariance = createCovarianceDict(len(states)*len(actions), self.init_cov, self.eta)
        self.answer = []
        self.responses = []
        self.states = states
        self.actions = actions
        self.action = None
        self.state = None

    def reinitialize(self):
        if len(self.answer) <> 0:
            self.responses.append(list(self.answer))
            self.answer = list()
        self.values = createQValuesDict(self.states, self.actions)

    def chooseAction(self, state):
        self.state = state
        self.covariance['noise'] = self.covariance['cov']*self.eta
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']
        self.action = getBestActionSoftMax(state, self.values, self.beta, 0)
        return self.action

    def update(self, reward):
        self.answer.append((reward==1)*1)
        sigma_points, weights = computeSigmaPoints(self.values[0], self.covariance['cov'], self.kappa)
        rewards_predicted = (sigma_points[:,self.values[(self.state, self.action)]]-self.gamma*np.max(sigma_points[:,self.values[self.state]], 1)).reshape(len(sigma_points), 1)
        reward_predicted = np.dot(rewards_predicted.flatten(), weights.flatten())
        cov_values_rewards = np.sum(weights*(sigma_points-self.values[0])*(rewards_predicted-reward_predicted), 0)
        cov_rewards = np.sum(weights*(rewards_predicted-reward_predicted)**2) + self.var_obs
        kalman_gain = cov_values_rewards/cov_rewards
        self.values[0] = self.values[0] + kalman_gain*(reward-reward_predicted)
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] - (kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards)*kalman_gain

class TreeConstruction():
    """Class that implement a trees construction based on 
    Color Association Experiments from Brovelli & al 2011
    """

    def __init__(self, states, actions, alpha, beta, gamma):
        self.alpha=alpha;self.beta=beta;self.gamma=gamma
        self.states=states
        self.actions_list=actions
        self.n_action=len(actions)
        self.g, self.action = self.initializeTree(states, actions)
        self.mental_path = []
        self.state=None
        self.answer = []
        self.responses = []

    def initializeTree(self, state, action):
        g = dict()
        dict_action = dict()
        for s in state:
            g[s] = dict()            
            g[s][0] = np.ones(len(action))*(1/float(len(action)))
            for a in range(1, len(action)+1):
                g[s][a] = dict()
                dict_action[a] = action[a-1]
                dict_action[action[a-1]] = a
        return g, dict_action

    def reinitialize(self):
        self.g, self.actions = self.initializeTree(self.states, self.actions_list)
        self.mental_path = []
        if len(self.answer) <> 0:
            self.responses.append(list(self.answer))
            self.answer = list()
        
    def chooseAction(self, ptr_trees):
        id_action = ptr_trees.keys()[self.sample(ptr_trees[0])]
        if id_action == 0:
            sys.stderr.write("End of trees\n")
            sys.exit()
        self.mental_path.append(id_action)
        if len(ptr_trees[id_action]):
            return self.chooseAction(ptr_trees[id_action])
        else:
            return self.action[id_action]

    def updateTrees(self, state, reward):        
        self.answer.append((reward==1)*1)
        if reward <> 1:
            self.extendTrees(self.mental_path, self.mental_path, self.g[state])
        elif reward == 1:
            self.reinforceTrees(self.mental_path, self.mental_path, self.g[state])

    def reinforceTrees(self, path, position, ptr_trees):
        if len(position) > 1:
            self.reinforceTrees(path, position[1:], ptr_trees[position[0]])
        elif len(position) == 1:
            ptr_trees[0] = np.zeros(len(ptr_trees.keys())-1)
            ptr_trees[0][ptr_trees.keys().index(position[0])-1] = 1
            self.mental_path = []

    def extendTrees(self, path, position, ptr_trees):
        if len(position) > 1:
            self.extendTrees(path, position[1:], ptr_trees[position[0]])
        elif len(position) == 1:
            ptr_trees[0] = np.zeros(len(ptr_trees.keys())-1)
            ptr_trees[0][ptr_trees.keys().index(position[0])-1] = 1
            self.extendTrees(path, position[1:], ptr_trees[position[0]])            
        else:
            new_nodes = set(range(1,self.n_action+1))-set(path)
            ptr_trees[0] = np.ones(len(new_nodes))*1/float(len(new_nodes))
            for i in new_nodes:
                ptr_trees[i] = {}
            self.mental_path = []

    def sample(self, values):
        #WARNING return 1 not 0 for indicing
        # values are probability
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())
