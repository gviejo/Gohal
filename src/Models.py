#!/usr/bin/python
#encoding: utf-8
"""
Models.py

Class of all models

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
from fonctions import *

class QLearning():
    """Class that implement a Qlearning
    """
    
    def __init__(self, name, states, actions, gamma, alpha, beta):
        # State action space
        self.states=states
        self.actions=actions
        #Parameters
        self.name = name
        self.gamma=gamma;self.alpha=alpha;self.beta=beta
        self.n_action=len(actions)
        self.n_state=len(states)
        self.bounds = dict({"gamma":[0.0, 1.0], "beta":[1.0, 10.0], "alpha":[0.0, 1.0]})
        #Values Initialization
        self.values = np.zeros((self.n_state, self.n_action))        
        #Valrious Init
        self.current_state = None
        self.current_action = None
        # List Init
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()

    def getAllParameters(self):
        return dict({'gamma':[self.bounds['gamma'][0],self.gamma,self.bounds['gamma'][1]],
                     'beta':[self.bounds['beta'][0],self.beta,self.bounds['beta'][1]],
                     'alpha':[self.bounds['alpha'][0],self.beta,self.bounds['alpha'][1]]})   

    def setAllParameters(self):
        for i in dict_p.iterkeys():
            self.setParameter(i, dict_p[i][1])

    def getParameter(self, name):
        if name == 'alpha':
            return self.alpha
        elif name == 'beta':
            return self.beta
        elif name == 'gamma':
            return self.gamma
        else:
            print "Unknow parameter"
            sys.exit(0)

    def setParameter(self, name, value):
        if name == 'gamma':
            if value < self.bounds['gamma'][0]:
                self.gamma = self.bounds['gamma'][0]
            elif value > self.bounds['gamma'][1]:
                self.gamma = self.bounds['gamma'][1]
            else:
                self.gamma = value                
        elif name == 'beta':
            if value < self.bounds['beta'][0]:
                self.beta = self.bounds['beta'][0]
            elif value > self.bounds['beta'][1]:
                self.beta = self.bounds['beta'][1]
            else :
                self.beta = value        
        elif name == 'alpha':
            if value < self.bounds['alpha'][0]:
                self.alpha = self.bounds['alpha'][0]
            elif value > self.bounds['alpha'][1]:
                self.alpha = self.bounds['alpha'][1]
            else:
                self.alpha = value
        else:
            print "Parameters not found"
            sys.exit(0)    

    
    def initializeList(self):
        self.responses = list()
        self.values = np.zeros((self.n_state, self.n_action))
        self.action = list()
        self.state = list()
        self.reaction = list()
        self.value = list()

    def initialize(self):
        self.responses.append([])
        self.values = np.zeros((self.n_state, self.n_action))
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.value.append([])

    def sampleSoftMax(self, values):
        tmp = np.exp(values*float(self.beta))
        tmp = tmp/float(np.sum(tmp))
        tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
        return np.sum(np.array(tmp) < np.random.rand())-1        

    def computeValue(self, state):
        self.current_state = convertStimulus(state)-1
        self.value[-1].append(SoftMaxValues(self.values[self.current_state], self.beta))
        return self.value[-1][-1]

    def chooseAction(self, state):        
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.current_action = self.sampleSoftMax(self.values[self.current_state])
        self.action[-1].append(self.actions[self.current_action])
        self.reaction[-1].append(computeEntropy(self.values[self.current_state], self.beta))
        self.value[-1].append(self.values[self.current_state])
        return self.action[-1][-1]
    
    def updateValue(self, reward):
        r = int((reward==1)*1)
        self.responses[-1].append(r)
        delta = reward+self.gamma*np.max(self.values[self.current_state])-self.values[self.current_state, self.current_state]
        self.values[self.current_state, self.current_action] = self.values[self.current_state, self.current_action]+self.alpha*delta
    

class KalmanQLearning():
    """ Class that implement a KalmanQLearning : 
    Kalman Temporal Differences : The deterministic case, Geist & al, 2009
    """

    def __init__(self, name, states, actions, gamma, beta, eta, var_obs, init_cov, kappa):
        #State Action Space
        self.states=states
        self.actions=actions
        #Parameters
        self.name=name
        self.gamma=gamma;self.beta=beta;self.eta=eta;self.var_obs=var_obs;self.init_cov=init_cov;self.kappa=kappa
        self.n_action=len(actions)
        self.n_state=len(states)
        self.bounds = dict({"gamma":[0.0, 1.0], "beta":[0.1, 10.0], "eta":[0.00001, 0.001]})
        #Values Initialization                
        self.values = np.zeros((self.n_state,self.n_action))
        self.covariance = createCovarianceDict(len(states)*len(actions), self.init_cov, self.eta)
        #Various Init
        self.current_state=None
        self.current_action=None
        self.point = None
        self.weights = None
        #List Init
        self.state = list()
        self.action = list()        
        self.responses = list()
        self.reaction = list()        
        self.value = list()        

    def getAllParameters(self):        
        return dict({'gamma':[self.bounds['gamma'][0],self.gamma,self.bounds['gamma'][1]],
                     'beta':[self.bounds['beta'][0],self.beta,self.bounds['beta'][1]],
                     'eta':[self.bounds['eta'][0],self.eta,self.bounds['eta'][1]]})                

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
            if value < self.bounds['gamma'][0]:
                self.gamma = self.bounds['gamma'][0]
            elif value > self.bounds['gamma'][1]:
                self.gamma = self.bounds['gamma'][1]
            else:
                self.gamma = value                
        elif name == 'beta':
            if value < self.bounds['beta'][0]:
                self.beta = self.bounds['beta'][0]
            elif value > self.bounds['beta'][1]:
                self.beta = self.bounds['beta'][1]
            else :
                self.beta = value        
        elif name == 'eta':
            if value < self.bounds['eta'][0]:
                self.eta = self.bounds['eta'][0]
            elif value > self.bounds['eta'][1]:
                self.eta = self.bounds['eta'][1]
            else:
                self.eta = value
        else:
            print "Parameters not found"
            sys.exit(0)    

    def initialize(self):
        self.responses.append([])        
        self.values = np.zeros((self.n_state, self.n_action))
        self.covariance = createCovarianceDict(self.n_state*self.n_action, self.init_cov, self.eta)
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.value.append([])

    def initializeList(self):
        self.values = np.zeros((self.n_state, self.n_action))
        self.covariance = createCovarianceDict(len(self.states)*len(self.actions), self.init_cov, self.eta)
        self.action = list()
        self.state = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()

    def sampleSoftMax(self, values):
        tmp = np.exp(values*float(self.beta))
        tmp = tmp/float(np.sum(tmp))
        tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
        return np.sum(np.array(tmp) < np.random.rand())-1

    def computeValue(self, state):        
        self.current_state = convertStimulus(state)-1        
        self.covariance['noise'] = self.covariance['cov']*self.eta
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']        
        self.value[-1].append(SoftMaxValues(self.values[self.current_state], self.beta))
        return self.value[-1][-1]

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1        
        self.covariance['noise'] = self.covariance['cov']*self.eta
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']        
        self.current_action = self.sampleSoftMax(self.values[self.current_state])
        self.value[-1].append(SoftMaxValues(self.values[self.current_state], self.beta))        
        self.action[-1].append(self.actions[self.current_action])        
        self.reaction[-1].append(computeEntropy(self.values[self.current_state], self.beta))
        return self.action[-1][-1]

    def updateValue(self, reward):
        #r = int((reward==0)*-1+(reward==1)*1)        
        r = int((reward==1)*1)
        self.responses[-1].append(r)                
        self.computeSigmaPoints()                
        t =self.n_action*self.current_state+self.current_action
        rewards_predicted = (self.point[:,t]-self.gamma*np.max(self.point[:,self.n_action*self.current_state:self.n_action*self.current_state+self.n_action], 1)).reshape(len(self.point), 1)
        reward_predicted = np.dot(rewards_predicted.flatten(), self.weights.flatten())        
        cov_values_rewards = np.sum(self.weights*(self.point-self.values.flatten())*(rewards_predicted-reward_predicted), 0)
        cov_rewards = np.sum(self.weights*(rewards_predicted-reward_predicted)**2) + self.var_obs
        kalman_gain = cov_values_rewards/cov_rewards
        self.values = (self.values.flatten() + kalman_gain*(r-reward_predicted)).reshape(self.n_state, self.n_action)
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] - (kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards)*kalman_gain

    def computeSigmaPoints(self):
        n = self.n_state*self.n_action
        self.point = np.zeros((2*n+1, n))
        self.point[0] = self.values.flatten()
        c = np.linalg.cholesky((n+self.kappa)*self.covariance['cov'])
        self.point[range(1,n+1)] = self.values.flatten()+np.transpose(c)
        self.point[range(n+1, 2*n+1)] = self.values.flatten()-np.transpose(c)
        self.weights = np.zeros((2*n+1,1))
        self.weights[1:2*n+1] = 1/(2*n+self.kappa)

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
        if reward != 1:
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


    """

    def __init__(self, name, states, actions, lenght_memory = 7, noise = 0.0, threshold = 1.0):
        # State Action Space        
        self.states=states
        self.actions=actions        
        #Parameters
        self.name = name
        self.lenght_memory = lenght_memory
        self.threshold = threshold
        self.w = noise
        self.n_action=int(len(actions))
        self.n_state=int(len(states))
        # Probability Initialization        
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        if "v1" in self.name:
            self.values = np.ones(self.n_action)*(1./self.n_action)
        elif "v2" in self.name:
            self.values = np.ones((self.n_state, self.n_action))*(1./self.n_action)
        self.p = None        
        # Various Init
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None        
        self.initial_entropy = -np.log2(1./self.n_action)
        self.entropy = self.initial_entropy        
        self.n_element = 0
        self.p_choice = 0.0
        self.bounds = dict({"lenght":[5, 100], "threshold":[0.0, 2.5], "noise":[0.0, 0.1]})
        # Optimization init
        self.p_s = np.zeros((self.lenght_memory, self.n_state))
        self.p_a_s = np.zeros((self.lenght_memory, self.n_state, self.n_action))
        self.p_r_as = np.zeros((self.lenght_memory, self.n_state, self.n_action, 2))
        #self.p_s_var = np.ones(self.n_state)*self.noise
        #self.p_a_s_var = np.ones((self.n_state, self.n_action))*self.noise
        #self.p_r_as_var = np.ones((self.n_state, self.n_action, 2))*self.noise
        #List Init
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()
        self.value=list()
        self.entropies=list()
        self.choice=list()
        self.sampleChoice=list()
        self.sampleEntropy=list()

    def getAllParameters(self):        
        return dict({'lenght':[self.bounds["lenght"][0],self.lenght_memory,self.bounds["lenght"][1]],
                     'noise':[self.bounds["noise"][0],self.w,self.bounds["noise"][1]],
                     'threshold':[self.bounds["threshold"][0],self.threshold,self.bounds["threshold"][1]]})

    def setAllParameters(self, dict_p):
        for i in dict_p.iterkeys():
            self.setParameter(i,dict_p[i][1])

    def setParameter(self, name, value):        
        if name == 'lenght':
            if value < self.bounds["lenght"][0]:
                self.lenght_memory = self.bounds["lenght"][0]
            elif value > self.bounds["lenght"][1]:
                self.lenght_memory = self.bounds["lenght"][1]
            else:
                self.lenght_memory = int(value)
        elif name == 'noise':
            if value < self.bounds["noise"][0]:
                self.w = self.bounds["noise"][0]
            elif value > self.bounds["noise"][1]:
                self.w = self.bounds["noise"][1]
            else:
                self.w = value
        elif name == 'threshold':
            if value < self.bounds["threshold"][0]:
                self.threshold = self.bounds["threshold"][0]
            elif value > self.bounds["threshold"][1]:
                self.threshold = self.bounds["threshold"][1]
            else:                
                self.threshold = value
        else:
            print("Parameters not found")
            sys.exit(0)

    def initialize(self):
        self.n_element = 0
        self.p_s = np.zeros((self.lenght_memory, self.n_state))
        self.p_a_s = np.zeros((self.lenght_memory, self.n_state, self.n_action))
        self.p_r_as = np.zeros((self.lenght_memory, self.n_state, self.n_action, 2))
        self.responses.append([])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.value.append([])
        self.entropies.append([])
        self.choice.append([])
        self.sampleChoice.append([])
        self.sampleEntropy.append([])

    def initializeList(self):
        self.n_element = 0
        self.p_s = np.zeros((self.lenght_memory, self.n_state))
        self.p_a_s = np.zeros((self.lenght_memory, self.n_state, self.n_action))
        self.p_r_as = np.zeros((self.lenght_memory, self.n_state, self.n_action, 2))
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()
        self.value=list()
        self.entropies=list()
        self.choice=list()
        self.sampleChoice=list()
        self.sampleEntropy=list()
        
    def sample(self, values):
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())-1

    def inferenceModule(self):        
        tmp = self.p_a_s[self.nb_inferences] * np.vstack(self.p_s[self.nb_inferences])
        self.p = self.p + self.p_r_as[self.nb_inferences] * np.reshape(np.repeat(tmp, 2, axis = 1), self.p_r_as[self.nb_inferences].shape)
        self.nb_inferences+=1

    def evaluationModule(self):
        tmp = self.p/np.sum(self.p)
        p_ra_s = tmp[self.current_state]/np.sum(tmp[self.current_state])
        p_r_s = np.sum(p_ra_s, axis = 0)
        p_a_rs = p_ra_s/p_r_s
        if "v1" in self.name:
            self.values = p_a_rs[:,1]/p_a_rs[:,0]
            self.values = self.values/np.sum(self.values)
            self.entropy = -np.sum(self.values*np.log2(self.values))
        elif "v2" in self.name:
            self.values[self.current_state] = p_a_rs[:,1]/p_a_rs[:,0]
            self.values[self.current_state] = self.values[self.current_state]/np.sum(self.values[self.current_state])
            self.entropy = -np.sum(self.values[self.current_state]*np.log2(self.values[self.current_state]))

    def decisionModule(self):                
        #self.p_choice = np.exp(-self.threshold*self.entropy)        
        self.p_choice = np.exp(-self.threshold*self.entropy[self.current_state])    
        self.sampleChoice[-1][-1].append(self.p_choice)
        self.sampleEntropy[-1][-1].append(self.entropy.copy())    

    def computeValue(self, state):        
        self.state[-1].append(state)
        self.sampleChoice[-1].append([])
        self.sampleEntropy[-1].append([])
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        if "v1" in self.name:
            self.entropy = self.initial_entropy
        elif "v2" in self.name:
            self.entropy = -np.sum(self.values[self.current_state]*np.log2(self.values[self.current_state]))        
        self.nb_inferences = 0     
        #self.decisionModule()   
        while self.entropy > self.threshold and self.nb_inferences < self.n_element:        
            self.inferenceModule()
            self.evaluationModule()        
            #self.decisionModule()
        if "v1" in self.name:
            return self.values
        elif "v2" in self.name:            
            return self.values[self.current_state]

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.sampleChoice[-1].append([])
        self.sampleEntropy[-1].append([])
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        if "v1" in self.name:
            self.entropy = self.initial_entropy
        elif "v2" in self.name:
            self.entropy = -np.sum(self.values[self.current_state]*np.log2(self.values[self.current_state]))
        self.nb_inferences = 0                 
        #self.decisionModule()        
        while self.entropy > self.threshold and self.nb_inferences < self.n_element:
        #for i in xrange(self.n_element):
        #while np.random.uniform(0,1) > self.p_choice and self.nb_inferences < self.n_element:        
            self.inferenceModule()
            self.evaluationModule()
            #self.decisionModule()
        if "v1" in self.name: 
            self.current_action = self.sample(self.values)            
            self.value[-1].append(self.values)
        elif "v2" in self.name:
            self.current_action = self.sample(self.values[self.current_state])             
            self.value[-1].append(self.values[self.current_state])        
        self.choice[-1].append(self.p_choice)        
        self.action[-1].append(self.actions[self.current_action])
        self.reaction[-1].append(self.nb_inferences)
        self.entropies[-1].append(self.entropy)
        return self.action[-1][-1]

    def updateValue(self, reward):
        r = int((reward==1)*1)
        self.responses[-1].append(r)
        #Adding noise
        # if self.noise:
        #     #self.p_s = self.p_s + np.random.beta(self.noise, 5, self.p_s.shape)
        #     #self.p_s = self.p_s + np.abs(np.random.normal(0, self.noise, self.p_s.shape))
        #     self.p_s = self.p_s + self.noise
        #     self.p_s[0:self.n_element] = self.p_s[0:self.n_element]/np.sum(self.p_s[0:self.n_element], axis = 1, keepdims = True)
        #     #self.p_a_s = self.p_a_s + np.random.beta(self.noise, 5, self.p_a_s.shape)            
        #     #self.p_a_s = self.p_a_s + np.abs(np.random.normal(0, self.noise, self.p_a_s.shape))
        #     self.p_a_s = self.p_a_s + self.noise
        #     self.p_a_s[0:self.n_element] = self.p_a_s[0:self.n_element]/np.sum(self.p_a_s[0:self.n_element], axis = 2, keepdims = True)
        #     #self.p_r_as = self.p_r_as + np.random.beta(self.noise, 5, self.p_r_as.shape)
        #     #self.p_r_as = self.p_r_as + np.abs(np.random.normal(0, self.noise, self.p_r_as.shape))
        #     self.p_r_as = self.p_r_as + self.noise
        #     self.p_r_as[0:self.n_element] = self.p_r_as[0:self.n_element]/np.sum(self.p_r_as[0:self.n_element], axis = 3, keepdims = True)            
        if self.w:
            self.p_s = self.p_s*(1-self.w)+self.w*(1.0/self.n_state*np.ones(self.p_s.shape))
            self.p_a_s = self.p_a_s*(1-self.w)+self.w*(1.0/self.n_action*np.ones(self.p_a_s.shape))
            self.p_r_as = self.p_r_as*(1-self.w)+self.w*(0.5*np.ones(self.p_r_as.shape))
        #Shifting memory            
        if self.n_element < self.lenght_memory:
            self.n_element+=1
        self.p_s[1:self.n_element] = self.p_s[0:self.n_element-1]
        self.p_a_s[1:self.n_element] = self.p_a_s[0:self.n_element-1]
        self.p_r_as[1:self.n_element] = self.p_r_as[0:self.n_element-1]
        self.p_s[0] = 0.0
        self.p_a_s[0] = np.ones((self.n_state, self.n_action))*(1/float(self.n_action))
        self.p_r_as[0] = np.ones((self.n_state, self.n_action, 2))*0.5
        #Adding last choice                 
        self.p_s[0, self.current_state] = 1.0        
        self.p_a_s[0, self.current_state] = 0.0
        self.p_a_s[0, self.current_state, self.current_action] = 1.0
        self.p_r_as[0, self.current_state, self.current_action] = 0.0
        self.p_r_as[0, self.current_state, self.current_action, int(r)] = 1.0        
        #Predicting Q-Value
        if "v2" in self.name:
            tmp = self.p_a_s[0] * np.vstack(self.p_s[0])
            self.p = self.p + self.p_r_as[0] * np.reshape(np.repeat(tmp, 2, axis = 1), self.p_r_as[0].shape)
            self.evaluationModule()

    def updatePartialValue(self, state, action, reward):
        r = (reward==1)*1
        self.action[-1].append(action)
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.current_action = convertAction(action)-1
        self.responses[-1].append(r)
        #Shifting memory            
        if self.n_element < self.lenght_memory:
            self.n_element+=1
        self.p_s[1:self.n_element] = self.p_s[0:self.n_element-1]
        self.p_a_s[1:self.n_element] = self.p_a_s[0:self.n_element-1]
        self.p_r_as[1:self.n_element] = self.p_r_as[0:self.n_element-1]
        self.p_s[0] = 0.0
        self.p_a_s[0] = np.ones((self.n_state, self.n_action))*(1/float(self.n_action))
        self.p_r_as[0] = np.ones((self.n_state, self.n_action, 2))*0.5
        #Adding last choice                 
        self.p_s[0, self.current_state] = 1.0        
        self.p_a_s[0, self.current_state] = 0.0
        self.p_a_s[0, self.current_state, self.current_action] = 1.0
        self.p_r_as[0, self.current_state, self.current_action] = 0.0
        self.p_r_as[0, self.current_state, self.current_action, int(r)] = 1.0
        #Adding noise
        if self.noise:
            self.p_s = self.p_s + np.random.beta(self.noise, 5, self.p_s.shape)
            self.p_s = self.p_s/np.sum(self.p_s, axis = 1, keepdims = True)
            self.p_a_s = self.p_a_s + np.random.beta(self.noise, 5, self.p_a_s.shape)
            self.p_a_s = self.p_a_s/np.sum(self.p_a_s, axis = 2, keepdims = True)
            self.p_r_as = self.p_r_as + np.random.beta(self.noise, 5, self.p_r_as.shape)
            self.p_r_as = self.p_r_as/np.sum(self.p_r_as, axis = 3, keepdims = True)  

    def computeInformationGain(self):
        tmp = self.p_a_s[0] * np.vstack(self.p_s[0])
        self.p = self.p + self.p_r_as[0] * np.reshape(np.repeat(tmp, 2, axis = 1), self.p_r_as[0].shape)
        tmp = self.p/np.sum(self.p)
        p_ra_s = tmp[self.current_state]/np.sum(tmp[self.current_state])
        p_r_s = np.sum(p_ra_s, axis = 0)
        p_a_rs = p_ra_s/p_r_s
        values = p_a_rs[:,1]/p_a_rs[:,0]
        values = values/np.sum(values)        
        return -np.sum(values*np.log2(values))


