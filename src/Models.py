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
    
    def __init__(self, name, states, actions, gamma, alpha, beta):
        self.name = name
        self.gamma=gamma;self.alpha=alpha;self.beta=beta
        self.values = createQValuesDict(states, actions)
        self.responses = list()
        self.states = states
        self.actions = actions
        self.action = list()
        self.state = list()
        self.reaction = list()

    def initialize(self):
        self.responses.append([])
        self.values = createQValuesDict(self.states, self.actions)
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        
    def chooseAction(self, state):
        self.state[-1].append(state)         
        self.action[-1].append(getBestActionSoftMax(state, self.values, self.beta))
        self.reaction[-1].append(computeEntropy(self.values[0][self.values[state]], self.beta))
        return self.action[-1][-1]
    
    def updateValue(self, reward):
        self.responses[-1].append((reward==1)*1)
        delta = reward+self.gamma*np.max(self.values[0][self.values[self.state[-1][-1]]])-self.values[0][self.values[(self.state[-1][-1],self.action[-1][-1])]]
        self.values[0][self.values[(self.state[-1][-1],self.action[-1][-1])]] = self.values[0][self.values[(self.state[-1][-1],self.action[-1][-1])]]+self.alpha*delta
    

class KalmanQLearning():
    """ Class that implement a KalmanQLearning : 
    Kalman Temporal Differences : The deterministic case, Geist & al, 2009
    """

    def __init__(self, name, states, actions, gamma, beta, eta, var_obs, sigma, init_cov, kappa):
        self.name=name
        self.gamma=gamma;self.beta=beta;self.eta=eta;self.var_obs=var_obs;self.sigma=sigma;self.init_cov=init_cov;self.kappa=kappa
        self.values = createQValuesDict(states, actions)
        self.covariance = createCovarianceDict(len(states)*len(actions), self.init_cov, self.eta)
        self.states = states
        self.actions = actions
        self.action = list()
        self.state = list()
        self.responses = list()
        self.reaction = list()

    def initialize(self):
        self.responses.append([])
        self.values = createQValuesDict(self.states, self.actions)
        self.action.append([])
        self.state.append([])
        self.reaction.append([])

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

class TreeConstruction():
    """Class that implement a trees construction based on 
    Color Association Experiments from Brovelli & al 2011
    """

    def __init__(self, name, states, actions, noise = 0.0):
        self.name = name
        self.noise = noise
        self.states=states
        self.actions=actions
        self.n_action=len(actions)
        self.initializeTree(states, actions)
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.mental_path=list()
        self.action=list()
        self.reaction=list()
        self.time_step=0.08

    def initializeTree(self, state, action):
        self.g = dict()
        self.dict_action = dict()
        for s in state:
            self.g[s] = dict()            
            self.g[s][0] = np.ones(len(action))*(1/float(len(action)))
            for a in range(1, len(action)+1):
                self.g[s][a] = dict()
                self.dict_action[a] = action[a-1]
                self.dict_action[action[a-1]] = a

    def initialize(self):
        self.initializeTree(self.states, self.actions)
        self.mental_path = []
        self.responses.append([])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
    
    def chooseAction(self, state):
        self.state[-1].append(state)        
        self.action[-1].append(self.branching(self.g[state], 0))
        return self.action[-1][-1]

    def branching(self, ptr_trees, edge_count):
        id_action = ptr_trees.keys()[self.sample(ptr_trees[0])]
        if id_action == 0:
            sys.stderr.write("End of trees\n")
            sys.exit()
        self.mental_path.append(id_action)
        if len(ptr_trees[id_action]):
            return self.branching(ptr_trees[id_action], edge_count+1)
        else:
            self.reaction[-1].append(edge_count*self.time_step)
            return self.dict_action[id_action]

    def updateTrees(self, state, reward):        
        self.responses[-1].append((reward==1)*1)
        if reward <> 1:
            self.extendTrees(self.mental_path, self.mental_path, self.g[state])
        elif reward == 1:
            self.reinforceTrees(self.mental_path, self.mental_path, self.g[state])
        #TO ADD NOISE TO OTHERS STATE
        if self.noise:
            for s in set(self.states)-set([state]):
                self.addNoise(self.g[s])

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

    def addNoise(self, ptr_tree):
        if 0 in ptr_tree.keys():
            tmp = np.abs(np.random.normal(ptr_tree[0], np.ones(len(ptr_tree[0]))*self.noise, len(ptr_tree[0])))
            ptr_tree[0] = tmp/np.sum(tmp)
            for k in ptr_tree.iterkeys():
                if type(ptr_tree[k]) == dict and len(ptr_tree[k].values()) > 0:
                    self.addNoise(ptr_tree[k])
