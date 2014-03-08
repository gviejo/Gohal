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
    
    def __init__(self, states, actions, parameters={'length':0}):
        # State action space
        self.states=states
        self.actions=actions
        #Parameters
        self.parameters = parameters
        self.n_action=len(actions)
        self.n_state=len(states)
        self.bounds = dict({"gamma":[0.0, 1.0],
                            "beta":[1.0, 5.0],
                            "alpha":[0.0, 1.0],
                            "sigma":[0.000001, 0.3]})
        
        #Values Initialization
        self.values = np.zeros((self.n_state, self.n_action))        
        #Various Init
        self.current_state = None
        self.current_action = None
        # List Init
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()
        self.sigma = list()
        self.pdf = list()

    def setParameters(self, name, value):            
        if value < self.bounds[name][0]:
            self.parameters[name] = self.bounds[name][0]
        elif value > self.bounds[name][1]:
            self.parameters[name] = self.bounds[name][1]
        else:
            self.parameters[name] = value                

    def setAllParameters(self, parameters):
        for i in parameters.iterkeys():
            if i in self.bounds.keys():
                self.setParameters(i, parameters[i])

    def startBloc(self):
        self.responses.append([])
        self.values = np.zeros((self.n_state, self.n_action))
        self.action.append([])
        self.state.append([])
        self.reaction.append([])

    def startExp(self):
        self.values = np.zeros((self.n_state, self.n_action))
        self.responses = list()
        self.action = list()
        self.state = list()
        self.reaction = list()
        self.value = list()
        self.sigma = list()      
        self.pdf = list()  

    def sampleSoftMax(self, values):
        tmp = np.exp(values*float(self.parameters['beta']))
        tmp = tmp/float(np.sum(tmp))
        tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
        return np.sum(np.array(tmp) < np.random.rand())-1        

    def computeValue(self, s, a):
        self.current_state = s
        self.current_action = a

        value = SoftMaxValues(self.values[self.current_state], self.parameters['beta'])
        self.value.append([float(value[self.current_action])])
        self.pdf.append(np.ones(1))
        self.sigma.append([self.parameters['sigma']])
        
    def chooseAction(self, state):        
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.current_action = self.sampleSoftMax(self.values[self.current_state])
        self.action[-1].append(self.actions[self.current_action])
        #self.reaction[-1].append(computeEntropy(self.values[self.current_state], self.parameters['beta']))
        self.value[-1].append(list(self.values[self.current_state]))
        self.reaction[-1].append(0.0)
        return self.action[-1][-1]
    
    def updateValue(self, reward):
        self.responses[-1].append(int((reward==1)*1))
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0        
        delta = r+self.parameters['gamma']*np.max(self.values[self.current_state])-self.values[self.current_state, self.current_action]        
        self.values[self.current_state, self.current_action] = self.values[self.current_state, self.current_action]+self.parameters['alpha']*delta
    

class KalmanQLearning():
    """ Class that implement a KalmanQLearning : 
    Kalman Temporal Differences : The deterministic case, Geist & al, 2009
    """

    def __init__(self, states, actions, parameters, var_obs = 0.05, init_cov = 10, kappa = 0.1):
        #State Action Space
        self.states=states
        self.actions=actions        
        #Parameters
        self.parameters = parameters
        self.var_obs = var_obs
        self.init_cov = init_cov
        self.kappa = kappa
        self.n_action=len(actions)
        self.n_state=len(states)
        self.bounds = dict({"gamma":[0.0, 1.0],
                            "beta":[0.1, 30.0],
                            "eta":[0.00001, 0.001]})
        #Values Initialization                
        self.values = np.zeros((self.n_state,self.n_action))
        self.covariance = createCovarianceDict(len(states)*len(actions), self.init_cov, self.parameters['eta'])
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

    def setParameters(self, name, value):            
        if value < self.bounds[name][0]:
            self.parameters[name] = self.bounds[name][0]
        elif value > self.bounds[name][1]:
            self.parameters[name] = self.bounds[name][1]
        else:
            self.parameters[name] = value                

    def setAllParameters(self, dict_p):
        for i in dict_p.iterkeys():
            self.setParameter(i,dict_p[i][1])

    def startBloc(self):
        self.responses.append([])        
        self.values = np.zeros((self.n_state, self.n_action))
        self.covariance = createCovarianceDict(self.n_state*self.n_action, self.init_cov, self.parameters['eta'])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.value.append([])

    def startExp(self):
        self.values = np.zeros((self.n_state, self.n_action))
        self.covariance = createCovarianceDict(len(self.states)*len(self.actions), self.init_cov, self.parameters['eta'])
        self.action = list()
        self.state = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()

    def sampleSoftMax(self, values):
        tmp = np.exp(values*float(self.parameters['beta']))
        tmp = tmp/float(np.sum(tmp))
        tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
        return np.sum(np.array(tmp) < np.random.rand())-1

    def computeValue(self, state):        
        self.current_state = convertStimulus(state)-1        
        self.predictionStep()
        self.value[-1].append(SoftMaxValues(self.values[self.current_state], self.parameters['beta']))
        self.reaction[-1].append(computeEntropy(self.values[self.current_state], self.parameters['beta']))
        return self.value[-1][-1]

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1        
        self.predictionStep()
        self.current_action = self.sampleSoftMax(self.values[self.current_state])
        self.value[-1].append(SoftMaxValues(self.values[self.current_state], self.parameters['beta']))        
        self.action[-1].append(self.actions[self.current_action])        
        self.reaction[-1].append(computeEntropy(self.values[self.current_state], self.parameters['beta']))
        return self.action[-1][-1]

    def updateValue(self, reward):
        self.responses[-1].append(int((reward==1)*1))
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0
        self.computeSigmaPoints()                
        t =self.n_action*self.current_state+self.current_action
        rewards_predicted = (self.point[:,t]-self.parameters['gamma']*np.max(self.point[:,self.n_action*self.current_state:self.n_action*self.current_state+self.n_action], 1)).reshape(len(self.point), 1)
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
        self.covariance['noise'] = self.covariance['cov']*self.parameters['eta']
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']    



class BayesianWorkingMemory():
    """Class that implement a bayesian working memory based on 
    Color Association Experiments from Brovelli & al 2011

    """

    def __init__(self, states, actions, parameters={'length':1}):
        # State Action Space        
        self.states=states
        self.actions=actions        
        #Parameters
        self.parameters = parameters
        self.n_action=int(len(actions))
        self.n_state=int(len(states))
        self.initial_entropy = -np.log2(1./self.n_action)
        self.bounds = dict({"length":[6, 11], 
                            "threshold":[0.0, self.initial_entropy], 
                            "noise":[0.0, 0.1],
                            "sigma":[0.000001, 0.3]})
        # Probability Initialization        
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        self.values = np.ones(self.n_action)*(1./self.n_action)    
        self.p = None        
        # Various Init
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None        
        self.entropy = self.initial_entropy        
        self.n_element = 0
        # Optimization init
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_r_s = np.ones(2)*0.5
        #List Init
        self.state=list()        
        self.action=list()
        self.responses=list()        
        self.reaction=list()
        self.value=list()
        self.pdf = list()
        self.sigma = list()

    def setParameters(self, name, value):            
        if value < self.bounds[name][0]:
            self.parameters[name] = self.bounds[name][0]
        elif value > self.bounds[name][1]:
            self.parameters[name] = self.bounds[name][1]
        else:
            self.parameters[name] = value                

    def setAllParameters(self, parameters):
        for i in parameters.iterkeys():
            if i in self.bounds.keys():
                self.setParameters(i, parameters[i])

    def startBloc(self):
        self.state.append([])
        self.action.append([])
        self.responses.append([])
        self.reaction.append([])
        self.n_element = 0
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.values = np.ones(self.n_action)*(1./self.n_action)
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None
                
    def startExp(self):
        self.n_element = 0
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.state=list()
        self.action=list()
        self.reaction=list()
        self.responses=list()
        self.value=list()
        self.values = np.ones(self.n_action)*(1./self.n_action)
        self.sigma = list()
        self.pdf = list()

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
        self.values = p_a_rs[:,1]/p_a_rs[:,0]
        self.values = self.values/np.sum(self.values)
        self.entropy = -np.sum(self.values*np.log2(self.values))

    def computeValue(self, s, a):
        self.current_state = s
        self.current_action = a
        self.p = self.uniform[:,:,:]
        self.entropy = self.initial_entropy
        self.nb_inferences = 0     

        value = np.zeros(int(self.parameters['length']+1))
        pdf = np.zeros(int(self.parameters['length'])+1)
        sigma = np.zeros(int(self.parameters['length'])+1)

        d = (self.entropy > self.parameters['threshold'] and self.nb_inferences < self.n_element)*1.0
        value[self.nb_inferences] = 1./float(self.n_action)
        pdf[self.nb_inferences] = 1.0-float(d)
        sigma[self.nb_inferences] = float(self.parameters['sigma'])        
        while self.nb_inferences < self.n_element:                    
            self.inferenceModule()
            self.evaluationModule()                    
            d = (self.entropy > self.parameters['threshold'] and self.nb_inferences < self.n_element)*1.0
            pdf[self.nb_inferences] = 1.0-float(d)
            value[self.nb_inferences] = float(self.values[self.current_action])
            sigma[self.nb_inferences] = float(self.parameters['sigma'])

        pdf = np.array(pdf)
        pdf[1:] = pdf[1:]*np.cumprod(1-pdf)[0:-1]

        self.pdf.append(pdf)
        self.value.append(value)
        self.sigma.append(sigma)        

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        self.entropy = self.initial_entropy
        self.nb_inferences = 0                 
        while self.entropy > self.parameters['threshold'] and self.nb_inferences < self.n_element:
            self.inferenceModule()
            self.evaluationModule()
        self.current_action = self.sample(self.values)            
        self.value[-1].append(self.values)
        self.action[-1].append(self.actions[self.current_action])
        self.reaction[-1].append(float(self.nb_inferences))
        return self.action[-1][-1]

    def updateValue(self, reward):
        r = int((reward==1)*1)
        self.responses[-1].append(r)
        if self.parameters['noise']:
            self.p_s = self.p_s*(1-self.parameters['noise'])+self.parameters['noise']*(1.0/self.n_state*np.ones(self.p_s.shape))
            self.p_a_s = self.p_a_s*(1-self.parameters['noise'])+self.parameters['noise']*(1.0/self.n_action*np.ones(self.p_a_s.shape))
            self.p_r_as = self.p_r_as*(1-self.parameters['noise'])+self.parameters['noise']*(0.5*np.ones(self.p_r_as.shape))
        #Shifting memory            
        if self.n_element < int(self.parameters['length']):
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

# class TreeConstruction():
#     """Class that implement a trees construction based on 
#     Color Association Experiments from Brovelli & al 2011
#     """

#     def __init__(self, name, states, actions, noise = 0.0):
#         self.name = name
#         self.noise = noise
#         self.states=states
#         self.actions=actions
#         self.n_action=len(actions)
#         self.initializeTree(states, actions)
#         self.state=list()
#         self.answer=list()
#         self.responses=list()
#         self.mental_path=list()
#         self.action=list()
#         self.reaction=list()
#         self.time_step=0.08

#     def initializeTree(self, state, action):
#         self.g = dict()
#         self.dict_action = dict()
#         for s in state:
#             self.g[s] = dict()            
#             self.g[s][0] = np.ones(len(action))*(1/float(len(action)))
#             for a in range(1, len(action)+1):
#                 self.g[s][a] = dict()
#                 self.dict_action[a] = action[a-1]
#                 self.dict_action[action[a-1]] = a

#     def initialize(self):
#         self.initializeTree(self.states, self.actions)
#         self.mental_path = []
#         self.responses.append([])
#         self.action.append([])
#         self.state.append([])
#         self.reaction.append([])

#     def initializeList(self):
#         self.initializeTree(self.states, self.actions)
#         self.mental_path = []
#         self.state=list()
#         self.answer=list()
#         self.responses=list()
#         self.mental_path=list()
#         self.action=list()
#         self.reaction=list()
#         self.time_step=0.08

#     def getParameter(self, name):
#         if name == 'noise':
#             return self.noise
#         else:
#             print("Parameters not found")
#             sys.exit(0)
    
#     def setParameter(self, name, value):
#         if name == 'noise':
#             self.noise = value
#         else:
#             print("Parameters not found")
#             sys.exit(0)

#     def chooseAction(self, state):
#         self.state[-1].append(state)        
#         self.action[-1].append(self.branching(self.g[state], 0))
#         return self.action[-1][-1]

#     def branching(self, ptr_trees, edge_count):
#         id_action = ptr_trees.keys()[self.sample(ptr_trees[0])]
#         if id_action == 0:
#             sys.stderr.write("End of trees\n")
#             sys.exit()
#         self.mental_path.append(id_action)
#         if len(ptr_trees[id_action]):
#             return self.branching(ptr_trees[id_action], edge_count+1)
#         else:
#             self.reaction[-1].append(edge_count*self.time_step)
#             return self.dict_action[id_action]

#     def updateTrees(self, state, reward):        
#         self.responses[-1].append((reward==1)*1)
#         if reward != 1:
#             self.extendTrees(self.mental_path, self.mental_path, self.g[state])
#         elif reward == 1:
#             self.reinforceTrees(self.mental_path, self.mental_path, self.g[state])
#         #TO ADD NOISE TO OTHERS STATE
#         if self.noise:
#             for s in set(self.states)-set([state]):
#                 self.addNoise(self.g[s])

#     def reinforceTrees(self, path, position, ptr_trees):
#         if len(position) > 1:
#             self.reinforceTrees(path, position[1:], ptr_trees[position[0]])
#         elif len(position) == 1:
#             ptr_trees[0] = np.zeros(len(ptr_trees.keys())-1)
#             ptr_trees[0][ptr_trees.keys().index(position[0])-1] = 1
#             self.mental_path = []

#     def extendTrees(self, path, position, ptr_trees):
#         if len(position) > 1:
#             self.extendTrees(path, position[1:], ptr_trees[position[0]])
#         elif len(position) == 1:
#             ptr_trees[0] = np.zeros(len(ptr_trees.keys())-1)
#             ptr_trees[0][ptr_trees.keys().index(position[0])-1] = 1
#             self.extendTrees(path, position[1:], ptr_trees[position[0]])            
#         else:
#             new_nodes = set(range(1,self.n_action+1))-set(path)
#             ptr_trees[0] = np.ones(len(new_nodes))*1/float(len(new_nodes))
#             for i in new_nodes:
#                 ptr_trees[i] = {}
#             self.mental_path = []

#     def sample(self, values):
#         #WARNING return 1 not 0 for indicing
#         # values are probability
#         tmp = [np.sum(values[0:i]) for i in range(len(values))]
#         return np.sum(np.array(tmp) < np.random.rand())

#     def addNoise(self, ptr_tree):
#         if 0 in ptr_tree.keys():
#             tmp = np.abs(np.random.normal(ptr_tree[0], np.ones(len(ptr_tree[0]))*self.noise, len(ptr_tree[0])))
#             ptr_tree[0] = tmp/np.sum(tmp)
#             for k in ptr_tree.iterkeys():
#                 if type(ptr_tree[k]) == dict and len(ptr_tree[k].values()) > 0:
#                     self.addNoise(ptr_tree[k])

