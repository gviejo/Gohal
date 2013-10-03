#!/usr/bin/python
#encoding: utf-8
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
    
    def initializeList(self):
        self.responses = list()
        self.values = createQValuesDict(self.states, self.actions)
        self.action = list()
        self.state = list()
        self.reaction = list()

    def getParameter(self, name):
        if name == 'alpha':
            return self.alpha
        elif name == 'gamma':
            return self.gamma
        else:
            print "Unknow parameter"
            sys.exit(0)

    def setParameter(self, name, value):
        if name == 'alpha':
            self.alpha = value
        elif name == 'gamma':
            self.gamma = value
        else:
            print "Unknow parameter"
            sys.exit(0)
    
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
        self.value = list()
        
    def getAllParameters(self):        
        return dict({'gamma':[0.1,self.gamma,0.9],
                     'beta':[1,self.beta,5]})
                     #'eta':[1.0e-6,self.eta,0.001],
                     #'var_obs':[0.01,self.var_obs,0.07]})
                     #'init_cov':[5,self.init_cov,15]})

    def setAllParameters(self, dict_p):
        for i in dict_p.iterkeys():
            self.setParameter(i,dict_p[i][1])

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
        elif name == 'var_obs':
            self.var_obs = value
        elif name == 'init_cov':
            self.init_cov = value
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
        self.value.append([])

    def initializeList(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.covariance = createCovarianceDict(len(self.states)*len(self.actions), self.init_cov, self.eta)
        self.action = list()
        self.state = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.covariance['noise'] = self.covariance['cov']*self.eta
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']
        self.value[-1].append(SoftMaxValues(self.values[0][self.values[state]], self.beta))
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

    def predictionStep(self):
        self.covariance['noise'] = self.covariance['cov']*self.eta
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']

    def updatePartialValue(self, s, a, n_s, reward):
        #FOr keramati model, action selection is made outside the class
        self.state[-1].append(s)
        self.action[-1].append(a)
        #self.reaction[-1].append(computeEntropy(self.values[0][self.values[s]], self.beta))
        self.responses[-1].append((reward==1)*1)
        sigma_points, weights = computeSigmaPoints(self.values[0], self.covariance['cov'], self.kappa)
        rewards_predicted = (sigma_points[:,self.values[(s,a)]]-self.gamma*np.max(sigma_points[:,self.values[n_s]], 1)).reshape(len(sigma_points), 1)
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

    def initializeList(self):
        self.initializeTree(self.states, self.actions)
        self.mental_path = []
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.mental_path=list()
        self.action=list()
        self.reaction=list()
        self.time_step=0.08

    def getParameter(self, name):
        if name == 'noise':
            return self.noise
        else:
            print("Parameters not found")
            sys.exit(0)
    
    def setParameter(self, name, value):
        if name == 'noise':
            self.noise = value
        else:
            print("Parameters not found")
            sys.exit(0)

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


class BayesianWorkingMemory():
    """Class that implement a bayesian working memory based on 
    Color Association Experiments from Brovelli & al 2011
    choose Action based on p(A = i/ R = 1,S = j)
    """

    def __init__(self, name, states, actions, lenght_memory = 7, noise = 0.0, beta = 1.0):
        # State Action Space        
        self.states=states
        self.actions=actions        
        #Parameters
        self.name = name
        self.lenght_memory = lenght_memory
        self.beta = beta
        self.threshold = 1.0
        self.noise = noise
        self.n_action=float(len(actions))
        self.n_state=float(len(states))
        # Probability Initialization
        self.initializeBMemory(states, actions)
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))        
        self.values = np.ones(self.n_action)*(1./self.n_action)
        self.p = None        
        # Various Init
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None        
        self.initial_entropy = -np.log2(1./self.n_action)
        #List Init
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()
        self.value=list()

    def setEntropyEvolution(self, nb_blocs, nb_trials):
        ## USE to study evolution of entropy##
        self.entropy = np.zeros((nb_blocs, nb_trials, nb_trials+1))
        self.bloc = -1
        self.trial = 0
        self.entropy[0,:,0] = self.initial_entropy
        #####################################
        
    def getAllParameters(self):        
        return dict({'lenght':[3, self.lenght_memory,15],
                     'noise':[1.0e-8,self.noise,0.5]})
                     #'beta':[1.0,self.beta,5.0]})

    def setAllParameters(self, dict_p):
        for i in dict_p.iterkeys():
            self.setParameter(i,dict_p[i][1])

    def setParameter(self, name, value):
        if name == 'lenght':
            self.lenght_memory = int(value)
        elif name == 'noise':
            self.noise = value
        elif name == 'beta':
            self.beta = value
        else:
            print("Parameters not found")
            sys.exit(0)

    def initializeBMemory(self, state, action):        
        #self.p_s = [np.ones((self.n_state))*(1/self.n_state)]
        #self.p_a_s = [np.ones((self.n_state, self.n_action))*(1/self.n_action)]
        #self.p_r_as = [np.ones((self.n_state, self.n_action, 2))*0.5]
        self.p_s = list() 
        self.p_a_s = list()
        self.p_r_as = list()

    def initialize(self):
        self.initializeBMemory(self.states, self.actions)
        self.responses.append([])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.value.append([])
        self.bloc+=1
        self.trial = 0

    def initializeList(self):
        self.initializeBMemory(self.states, self.actions)
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()
        self.value=list()
        self.trial = 0
        self.bloc = 0

    def normalize(self):
        for i in xrange(len(self.p_s)):
            self.p_s[i] = self.p_s[i]/np.sum(self.p_s[i])
            self.p_a_s[i] = self.p_a_s[i]/np.vstack(np.sum(self.p_a_s[i], axis = 1))
            self.p_r_as[i] = self.p_r_as[i]/np.reshape(np.repeat(np.sum(self.p_r_as[i], axis=2),2),self.p_r_as[i].shape)
            
    def sample(self, values):
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())-1

    def inferenceModule(self):        
        tmp = self.p_a_s[self.nb_inferences] * np.vstack(self.p_s[self.nb_inferences])
        self.p += self.p_r_as[self.nb_inferences] * np.reshape(np.repeat(tmp, 2, axis = 1), self.p_r_as[self.nb_inferences].shape)
        self.p = self.p/np.sum(self.p)        
        self.nb_inferences+=1

    def evaluationModule(self):
        p_ra_s = self.p[self.current_state]/np.sum(self.p[self.current_state])
        p_r_s = np.sum(p_ra_s, axis = 0)
        p_a_rs = p_ra_s/p_r_s
        self.values = p_a_rs[:,1]/p_a_rs[:,0]
        self.values = self.values/np.sum(self.values)        
        #self.entropy = -np.sum(self.values*np.log2(self.values))
        self.entropy[self.bloc, self.trial, self.nb_inferences] = -np.sum(self.values*np.log2(self.values))

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform
        #self.entropy = self.initial_entropy
        self.nb_inferences = 0        
        #while self.entropy > self.threshold:
        for i in xrange(len(self.p_s)):
            self.inferenceModule()
            self.evaluationModule()
        self.current_action = self.sample(self.values)            
        self.value[-1].append(self.values)        
        self.action[-1].append(self.actions[self.current_action])
        self.trial+=1
        return self.action[-1][-1]

    def updateValue(self, reward):
        r = int((reward==1)*1)
        self.responses[-1].append(r)
        #Shifting memory            
        self.p_s.insert(0, np.zeros((self.n_state)))
        self.p_a_s.insert(0, np.ones((self.n_state, self.n_action))*(1/float(self.n_action)))
        self.p_r_as.insert(0, np.ones((self.n_state, self.n_action, 2))*0.5)        
        #Adding last choice         
        self.p_s[0][self.current_state] = 1.0    
        self.p_a_s[0][self.current_state] = 0.0        
        self.p_a_s[0][self.current_state, self.current_action] = 1.0
        self.p_r_as[0][self.current_state, self.current_action] = 0.0
        self.p_r_as[0][self.current_state, self.current_action, int(r)] = 1.0
        #Length of memory allowed
        while len(self.p_a_s) > self.lenght_memory:
            self.p_s.pop(-1)
            self.p_a_s.pop(-1)
            self.p_r_as.pop(-1)
        #Adding noise
        if self.noise:
            for i in xrange(1, len(self.p_s)):
                self.p_s[i] = np.abs(np.random.normal(self.p_s[i], np.ones(self.p_s[i].shape)*self.noise, self.p_s[i].shape))
                self.p_a_s[i] = np.abs(np.random.normal(self.p_a_s[i], np.ones(self.p_a_s[i].shape)*self.noise, self.p_a_s[i].shape))
                self.p_r_as[i] = np.abs(np.random.normal(self.p_r_as[i], np.ones(self.p_r_as[i].shape)*self.noise,self.p_r_as[i].shape))
            self.normalize()        

    def updatePartialValue(self, state, action, reward):
        r = (reward==1)*1
        self.action[-1].append(action)
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.current_action = convertAction(action)-1
        self.responses[-1].append(r)
        #Shifting memory            
        self.p_s.insert(0, np.zeros((self.n_state)))
        self.p_a_s.insert(0, np.ones((self.n_state, self.n_action))*(1/float(self.n_action)))
        self.p_r_as.insert(0, np.ones((self.n_state, self.n_action, 2))*0.5)        
        #Adding last choice         
        self.p_s[0][self.current_state] = 1.0
        self.p_a_s[0][self.current_state] = 0.0        
        self.p_a_s[0][self.current_state, self.current_action] = 1.0
        self.p_r_as[0][self.current_state, self.current_action] = 0.0
        self.p_r_as[0][self.current_state, self.current_action, int(r)] = 1.0
        #Length of memory allowed
        while len(self.p_a_s) > self.lenght_memory:
            self.p_s.pop(-1)
            self.p_a_s.pop(-1)
            self.p_r_as.pop(-1)
        #Adding noise
        if self.noise:
            for i in xrange(1, len(self.p_s)):
                self.p_s[i] = np.abs(np.random.normal(self.p_s[i], np.ones(self.p_s[i].shape)*self.noise, self.p_s[i].shape))
                self.p_a_s[i] = np.abs(np.random.normal(self.p_a_s[i], np.ones(self.p_a_s[i].shape)*self.noise, self.p_a_s[i].shape))
                self.p_r_as[i] = np.abs(np.random.normal(self.p_r_as[i], np.ones(self.p_r_as[i].shape)*self.noise,self.p_r_as[i].shape))
            self.normalize()        

    

    # def computeBayesianInference(self, i):
    #     tmp = self.p_a_s[i] * np.vstack(self.p_s[i])
    #     return self.p_r_as[i] * np.reshape(np.repeat(tmp, 2, axis = 1), self.p_r_as[i].shape)

    # def computeDecisionValues(self, p):
    #     p_ra_s = p/np.sum(p)
    #     p_r_s = np.sum(p_ra_s, axis = 0)
    #     p_a_rs = p_ra_s/p_r_s
    #     value = p_a_rs[:,1]/p_a_rs[:,0]
    #     return value                                

    # def computeValue(self, state):
    #     self.state[-1].append(state)
    #     self.current_state = convertStimulus(state)-1
    #     #Bayesian Inference
    #     p = np.zeros((self.n_state,self.n_action,2))
    #     for i in xrange(len(self.p_a_s)):
    #         p += self.computeBayesianInference(i) 
    #         p = p/np.sum(p)
    #         value = self.computeDecisionValues(p[self.current_state])            
    #         value = value/np.sum(value)
    #         self.entropy[self.bloc, self.trial, i] = computeEntropy(value, 1.0)        
    #     self.p = p # for Collins model
    #     #Current state        
    #     return value

    # def chooseAction2(self, state):
    #     value = self.computeValue(state)                
    #     self.reaction[-1].append(computeEntropy(value, 1.0))
    #     #Sample according to p(A/R,S)        
    #     self.value[-1].append(value)
    #     self.current_action = self.sample(value)
    #     #self.current_action = SoftMax(value, self.beta)
    #     self.action[-1].append(self.actions[self.current_action])
    #     self.trial+=1        
    #     return self.action[-1][-1]

