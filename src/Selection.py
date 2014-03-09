#!/usr/bin/python
# encoding: utf-8
"""
Selection.py

Class of for strategy selection when training

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *
from Models import *

class FSelection():
    """ fusion strategy
    Specially tuned for Brovelli experiment so beware

    """
    def __init__(self, states, actions, parameters={"length":1}):
        #State Action Spaces
        self.states=states
        self.actions=actions
        #Parameters
        self.parameters = parameters
        self.n_action = int(len(actions))
        self.n_state = int(len(states))
        self.bounds = dict({"gamma":[0.0, 1.0],
                            "beta":[1.0, 7.0],
                            "alpha":[0.0, 1.0],
                            "length":[6, 11],
                            #"threshold":[0.00001, 100.0], 
                            "noise":[0.0, 0.1],
                            "gain":[0.00001, 10.0],
                            "sigma_bwm":[0.00001, 0.1],
                            "sigma_ql":[0.00001, 0.1]})
                            

        #Probability Initialization
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_a_mf = None
        self.p_a_mb = None
        self.p = None
        self.p_a = None
        self.pA = None
        # QValues model free
        self.values_mf = np.zeros((self.n_state, self.n_action))
        self.values_net = None
        # Control initialization
        self.nb_inferences = 0
        self.n_element= 0
        self.current_state = None
        self.current_action = None
        self.max_entropy = -np.log2(1./self.n_action)
        self.Hb = self.max_entropy
        self.Hf = self.max_entropy
        # List Init
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()
        self.pdf = list()
        self.sigma = list()
        self.sigma_test = list()

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
        self.sigma_test.append([])
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.values_mf = np.zeros((self.n_state, self.n_action))
        self.nb_inferences = 0
        self.n_element = 0
        self.current_state = None
        self.current_action = None
        self.Hb = self.max_entropy
        self.Hf = self.max_entropy        

    def startExp(self):
        self.n_element = 0                
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()        
        self.pdf = list()
        self.sigma = list()
        self.sigma_test = list()

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
        self.p_a_mb = p_a_rs[:,1]/p_a_rs[:,0]
        p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)
        self.Hb = -np.sum(p_a_mb*np.log2(p_a_mb))

    def sigmoideModule(self):
        x = 2*self.max_entropy-self.Hb-self.Hf
        #self.pA = 1/(1+(self.n_element-self.nb_inferences)*np.exp(-x))
        #self.pA = 1/(1+((self.n_element-self.nb_inferences)*self.parameters['threshold'])*np.exp(-x*self.parameters['gain']))
        #self.pA = 1/(1+((self.n_element-self.nb_inferences)/self.parameters['threshold'])*np.exp(-x/self.parameters['gain']))
        self.pA = 1/(1+(self.n_element-self.nb_inferences)*np.exp(-x*self.parameters['gain']))        
        #self.pA = 1/(1+((self.n_element-self.nb_inferences)/self.parameters['threshold'])*np.exp(-x/self.parameters['gain']))        
        return np.random.uniform(0,1) > self.pA
    

    def fusionModule(self):
        np.seterr(invalid='ignore')
        #w = (self.max_entropy-self.Hb)/self.max_entropy
        #self.values_net = w*self.p_a_mb+self.values_mf[self.current_state]
        self.values_net = self.p_a_mb+self.values_mf[self.current_state]
        tmp = np.exp(self.values_net*float(self.parameters['beta']))
        self.p_a = tmp/np.sum(tmp)        
        # if True in np.isnan(self.p_a):
        #     self.p_a = np.isnan(self.p_a)*0.995+0.001

    def computeValue(self, s, a):        
        self.current_state = s
        self.current_action = a
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.Hf = computeEntropy(self.values_mf[self.current_state], self.parameters['beta'])
        self.nb_inferences = 0
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
        
        value = np.zeros(int(self.parameters['length']+1))        
        pdf = np.zeros(int(self.parameters['length'])+1)
        sigma = np.zeros(int(self.parameters['length']+1))

        d = self.sigmoideModule()
        pdf[self.nb_inferences] = float(self.pA)
        self.fusionModule()
        value[self.nb_inferences] = float(self.p_a[self.current_action])        
        sigma[self.nb_inferences] = float(self.parameters['sigma_ql'])

        while self.nb_inferences < self.n_element:
            self.inferenceModule()
            self.evaluationModule()
            self.fusionModule()
            value[self.nb_inferences] = float(self.p_a[self.current_action])        
            sigma[self.nb_inferences] = float(self.parameters['sigma_bwm'])
            d = self.sigmoideModule()
            pdf[self.nb_inferences] = float(self.pA)
        
        pdf = np.array(pdf)
        pdf[1:] = pdf[1:]*np.cumprod(1-pdf)[0:-1]
        
        self.pdf.append(pdf)
        self.value.append(value)
        self.sigma.append(sigma)        
                
    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.Hf = computeEntropy(self.values_mf[self.current_state], self.parameters['beta'])
        self.nb_inferences = 0
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
        pdf = np.zeros(int(self.parameters['length'])+1)
        pdf[self.nb_inferences] = 1.0
        while self.sigmoideModule():
            self.inferenceModule()
            self.evaluationModule()
            pdf[self.nb_inferences] = 1.0 - float(self.pA)            
        self.fusionModule()
        self.current_action = self.sample(self.p_a)        
        self.action[-1].append(self.actions[self.current_action])                
        self.reaction[-1].append(self.nb_inferences)
        self.sigma_test[-1].append([self.parameters['sigma_ql'], self.parameters['sigma_bwm']][int(self.nb_inferences != 0)])

        while self.nb_inferences < self.n_element:
            d = self.sigmoideModule()
            self.inferenceModule()
            self.evaluationModule()
            pdf[self.nb_inferences] = 1.0 - float(self.pA)            
        pdf = np.cumprod(pdf)
        self.pdf.append(pdf)
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
        # Updating model free
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0        
        delta = float(r)+self.parameters['gamma']*np.max(self.values_mf[self.current_state])-self.values_mf[self.current_state, self.current_action]        
        #delta = float(r)-self.values_mf[self.current_state, self.current_action]        
        self.values_mf[self.current_state, self.current_action] = self.values_mf[self.current_state, self.current_action]+self.parameters['alpha']*delta
        #self.values_mf[self.current_state, self.current_action] = self.values_mf[self.current_state, self.current_action]+0.9*delta
        

class KSelection():
    """Class that implement Keramati models for action selection
    Specially tuned for Brovelli experiment so beware
    """
    def __init__(self, states, actions, parameters={"length":1,"eta":0.0001}, var_obs = 0.05, init_cov = 10, kappa = 0.1):
        #State Action Spaces
        self.states=states
        self.actions=actions
        #Parameters
        self.parameters = parameters
        self.n_action = int(len(actions))
        self.n_state = int(len(states))
        self.bounds = dict({"gamma":[0.0, 1.0],
                            "beta":[1.0, 7.0],
                            "eta":[0.00001, 0.001],
                            "length":[6, 11],
                            "threshold":[0.0, -np.log2(1./self.n_action)], 
                            "noise":[0.0, 0.1],
                            "sigma":[0.0,1.0],
                            "sigma_bwm":[0.00001, 0.3],
                            "sigma_ql":[0.00001, 0.3]})        
        self.var_obs = var_obs
        self.init_cov = init_cov
        self.kappa = kappa
        #Probability Initialization
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_a_mf = None
        self.p_a_mb = None
        self.p = None
        self.p_a = None
        self.pA = None
        # QValues model free
        self.values_mf = np.zeros((self.n_state, self.n_action))
        self.covariance = createCovarianceDict(self.n_state*self.n_action, self.init_cov, self.parameters['eta'])
        self.point = None
        self.weights = None
        # Control initialization
        self.nb_inferences = 0
        self.n_element= 0
        self.current_state = None
        self.current_action = None
        self.max_entropy = -np.log2(1./self.n_action)
        self.Hb = self.max_entropy
        self.reward_rate = np.zeros(self.n_state)
        # List Init
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()
        self.vpi = list()
        self.rrate = list()
        self.sigma = list()
        self.sigma_test = list()

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
        self.vpi.append([])
        self.rrate.append([])
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.values_mf = np.zeros((self.n_state, self.n_action))
        self.nb_inferences = 0
        self.n_element = 0
        self.values = None
        self.current_state = None
        self.current_action = None
        self.Hb = self.max_entropy
        self.Hf = self.max_entropy

    def startExp(self):                
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()   
        self.vpi = list()
        self.rrate = list() 
        self.sigma = list()    
        self.sigma_test = list()
        self.pdf = list()

    def sampleSoftMax(self, values):
        tmp = np.exp(values*float(self.parameters['beta']))
        tmp = tmp/float(np.sum(tmp))
        tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
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
        self.p_a_mb = p_a_rs[:,1]/p_a_rs[:,0]
        p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)
        self.Hb = -np.sum(p_a_mb*np.log2(p_a_mb))
        self.values = p_a_rs[:,1]/p_a_rs[:,0]
        self.values = self.values/np.sum(self.values)

    def predictionStep(self):
        self.covariance['noise'] = self.covariance['cov']*self.parameters['eta']        
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']    

    def computeSigmaPoints(self):
        n = self.n_state*self.n_action
        self.point = np.zeros((2*n+1, n))
        self.point[0] = self.values_mf.flatten()
        c = np.linalg.cholesky((n+self.kappa)*self.covariance['cov'])
        self.point[range(1,n+1)] = self.values_mf.flatten()+np.transpose(c)
        self.point[range(n+1, 2*n+1)] = self.values_mf.flatten()-np.transpose(c)
        self.weights = np.zeros((2*n+1,1))
        self.weights[1:2*n+1] = 1/(2*n+self.kappa)

    def updateRewardRate(self, reward, delay = 0.0):
        self.reward_rate[self.current_state] = (1.0-self.parameters['sigma'])*self.reward_rate[self.current_state]+self.parameters['sigma']*reward
        self.rrate[-1].append(self.reward_rate[self.current_state])

    def softMax(self, values):
        tmp = np.exp(values*float(self.parameters['beta']))
        return tmp/float(np.sum(tmp))

    def sample(self, values):
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())-1        
        
    def computeValue(self, s, a):
        self.current_state = s
        self.current_action = a
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy            
        self.nb_inferences = 0
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
        self.predictionStep()        
        t = self.n_action*self.current_state
        vpi = computeVPIValues(self.values_mf[self.current_state], self.covariance['cov'].diagonal()[t:t+self.n_action])
        
        value = np.zeros(int(self.parameters['length']+1))
        pdf = np.zeros(int(self.parameters['length'])+1)
        sigma = np.zeros(int(self.parameters['length']+1))
        
        # 1ere transition should be made outside the while
        d = (np.sum(vpi > self.reward_rate[self.current_state])>0)*(self.nb_inferences < self.n_element)*1.0
        values = self.softMax(self.values_mf[self.current_state])
        value[self.nb_inferences] = float(values[self.current_action])
        pdf[self.nb_inferences] = 1.0-float(d)
        sigma[self.nb_inferences] = float(self.parameters['sigma_ql'])
                
        while self.nb_inferences < self.n_element:
            self.inferenceModule()
            self.evaluationModule()
            d = (self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element)*1.0
            pdf[self.nb_inferences] = 1.0-float(d)
            value[self.nb_inferences] = float(self.values[self.current_action])
            sigma[self.nb_inferences] = float(self.parameters['sigma_ql'])

        pdf = np.array(pdf)
        pdf[1:] = pdf[1:]*np.cumprod(1-pdf)[0:-1]

        self.pdf.append(pdf)
        self.value.append(value)
        self.sigma.append(sigma)

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.nb_inferences = 0
        self.predictionStep()
        values = self.softMax(self.values_mf[self.current_state])
        t =self.n_action*self.current_state
        vpi = computeVPIValues(self.values_mf[self.current_state], self.covariance['cov'].diagonal()[t:t+self.n_action])
        self.vpi[-1].append(vpi)
        if np.sum(vpi > self.reward_rate[self.current_state]):
            self.p = self.uniform[:,:,:]
            self.Hb = self.max_entropy            
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:
                self.inferenceModule()
                self.evaluationModule()
            values = self.p_a_mb/np.sum(self.p_a_mb)
        self.current_action = self.sample(values)
        
        self.action[-1].append(self.actions[self.current_action])
        self.reaction[-1].append(self.nb_inferences)
        #self.sigma[-1].append([self.parameters['sigma_ql'], self.parameters['sigma_bwm']][int(self.nb_inferences != 0)])
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
        # Updating model free
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0        
        self.computeSigmaPoints()                
        t =self.n_action*self.current_state+self.current_action
        rewards_predicted = (self.point[:,t]-self.parameters['gamma']*np.max(self.point[:,self.n_action*self.current_state:self.n_action*self.current_state+self.n_action], 1)).reshape(len(self.point), 1)
        reward_predicted = np.dot(rewards_predicted.flatten(), self.weights.flatten())        
        cov_values_rewards = np.sum(self.weights*(self.point-self.values_mf.flatten())*(rewards_predicted-reward_predicted), 0)
        cov_rewards = np.sum(self.weights*(rewards_predicted-reward_predicted)**2) + self.var_obs
        kalman_gain = cov_values_rewards/cov_rewards
        self.values_mf = (self.values_mf.flatten() + kalman_gain*(r-reward_predicted)).reshape(self.n_state, self.n_action)
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] - (kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards)*kalman_gain
        # Updating selection 
        self.updateRewardRate(r)


            
class CSelection():
    """Class that implement Collins models for action selection
    Model-based must be provided
    Specially tuned for Brovelli experiment so beware
    """
    def __init__(self, free, based, w_0):
        self.w0 = w_0
        self.C = float(based.lenght_memory)
        self.n_s = float(len(free.states))
        self.n_a = float(len(free.actions))
        self.free = free
        self.based = based
        self.actions = free.actions; 
        self.states = free.states        
        self.values = createQValuesDict(self.states, self.actions)
        self.w = {i:self.w0*np.min([1,(self.based.lenght_memory/float(len(self.states)))]) for i in self.states}
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.weight = list()
        self.model_based_values = None
        self.model_free_values = None
        self.p_r_based = list()
        self.p_r_free = list()

    def initialize(self):
        self.free.initialize()
        self.based.initialize()
        self.responses.append([])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.weight.append([])
        self.values = createQValuesDict(self.states, self.actions)
        self.w = {i:self.w0*np.min([1,self.C/self.n_s]) for i in self.states}
        self.p_r_based.append([])
        self.p_r_free.append([])

    def initializeList(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.w = {i:self.w0*np.min([1,self.C/self.n_s]) for i in self.states}
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()
        self.weight=list()
        self.p_r_based = list()
        self.p_r_free = list()

    def computeRewardLikelihood(self, s, reward):
        tmp = np.min([1.0, self.C/self.n_s])
        if reward == 1:
            p_r_bwm = tmp*self.model_based_values + (1-tmp)/float(len(self.actions))
            p_r_rl = self.free.values[0][self.free.values[self.states[s]]]
        elif reward == 0:
            p_r_bwm = tmp*(1-self.model_based_values) + (1-tmp)/float(len(self.actions))
            p_r_rl = 1.0 - self.free.values[0][self.free.values[self.states[s]]]
        p_r_bwm = p_r_bwm/np.sum(p_r_bwm)
        p_r_rl = np.exp(p_r_rl)/np.sum(np.exp(p_r_rl))
        return p_r_bwm, p_r_rl

    def updateWeight(self, s, a, reward):
        assert reward == 0 or reward == 1
        #print reward, self.free.values[0][self.free.values[(self.states[s],self.actions[a])]]
        (p_r_bwm,p_r_rl) = self.computeRewardLikelihood(s, reward)
        #print p_r_rl[a]
        self.w[self.states[s]] = (p_r_bwm[a]*self.w[self.states[s]])/(p_r_bwm[a]*self.w[self.states[s]]+p_r_rl[a]*(1-self.w[self.states[s]]))
        self.p_r_based[-1].append(p_r_bwm[a])
        self.p_r_free[-1].append(p_r_rl[a])
    
    def chooseAction(self, state):
        self.state[-1].append(state)
        self.weight[-1].append(self.w[state])
        self.free.predictionStep()
        
        self.model_based_values = self.based.computeValue(state) 
        self.model_based_values = self.model_based_values/float(np.sum(self.model_based_values))
        self.model_free_values = np.exp(self.free.values[0][self.free.values[state]]*float(self.free.beta))
        self.model_free_values =  self.model_free_values/float(np.sum(self.model_free_values))

        self.values[0][self.values[state]] = (1-self.w[state])*self.model_free_values + self.w[state]*self.model_based_values

        action = getBestAction(state, self.values)
        self.action[-1].append(action)
        return action

    def updateValue(self, reward):
        self.responses[-1].append((reward==1)*1)
        self.updateWeight(self.states.index(self.state[-1][-1]), self.actions.index(self.action[-1][-1]), (reward==1)*1)        
        self.free.updatePartialValue(self.state[-1][-1], self.action[-1][-1], self.state[-1][-1], reward)
        self.based.updatePartialValue(self.state[-1][-1], self.action[-1][-1], reward)

    def getAllParameters(self):
        tmp = dict({'w0':[0.0, self.w0, 1.0]})
        tmp.update(self.free.getAllParameters())
        tmp.update(self.based.getAllParameters())
        return tmp


class Keramati():
    """Class that implement Keramati models for action selection
    Use to replicate exp 1 from Keramati & al, 2011
    """
    
    def __init__(self, kalman,depth,phi, rau, sigma, tau):
        self.kalman = kalman
        self.depth = depth; self.phi = phi; self.rau = rau;self.sigma = sigma; self.tau = tau
        self.actions = kalman.actions; self.states = kalman.states
        self.values = createQValuesDict(kalman.states, kalman.actions)
        self.rfunction = createQValuesDict(kalman.states, kalman.actions)
        self.vpi = dict.fromkeys(self.states,list())
        self.rrate = [0.0]
        self.state = None
        self.action = None
        self.transition = createTransitionDict(['s0','s0','s1','s1'],
                                               ['pl','em','pl','em'],
                                               ['s1','s0','s0','s0'], 's0') #<====VERY BAD==============    NEXT_STATE = TRANSITION[(STATE, ACTION)]        
    def initialize(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.rfunction = createQValuesDict(self.states, self.actions)
        self.vpi = dict.fromkeys(self.states,list())
        self.rrate = [0.0]
        self.state = None
        self.action = None
        self.transition = createTransitionDict(['s0','s0','s1','s1'],
                                               ['pl','em','pl','em'],
                                               ['s1','s0','s0','s0'], 's0')
                
    def chooseAction(self, state):
        self.state = state
        self.kalman.predictionStep()
        vpi = computeVPIValues(self.kalman.values[0][self.kalman.values[self.state]], self.kalman.covariance['cov'].diagonal()[self.kalman.values[self.state]])
        
        for i in range(len(vpi)):
            if vpi[i] >= self.rrate[-1]*self.tau:
                depth = self.depth
                self.values[0][self.values[(self.state, self.actions[i])]] = self.computeGoalValue(self.state, self.actions[i], depth)
            else:
                self.values[0][self.values[(self.state, self.actions[i])]] = self.kalman.values[0][self.kalman.values[(self.state,self.actions[i])]]

        self.action = getBestActionSoftMax(state, self.values, self.kalman.beta)
        return self.action

    def updateValues(self, reward, next_state):
        self.updateRewardRate(reward, delay = 0.0)
        self.kalman.updatePartialValue(self.state, self.action, next_state, reward)
        self.updateRewardFunction(self.state, self.action, reward)
        self.updateTransitionFunction(self.state, self.action)

    def updateRewardRate(self, reward, delay = 0.0):
        self.rrate.append(((1-self.sigma)**(1+delay))*self.rrate[-1]+self.sigma*reward)

    def updateRewardFunction(self, state, action, reward):
        self.rfunction[0][self.rfunction[(state, action)]] = (1-self.rau)*self.rfunction[0][self.rfunction[(state, action)]]+self.rau*reward

    def updateTransitionFunction(self, state, action):
        #This is cheating since the transition is known inside the class
        #Plus assuming the transition are deterministic
        nextstate = self.transition[(state, action)]
        for i in [nextstate]:
            if i == nextstate:
                self.transition[(state, action, nextstate)] = (1-self.phi)*self.transition[(state, action, nextstate)]+self.phi
            else:
                self.transition[(state, action, i)] = (1-self.phi)*self.transition[(state, action, i)]
        
    def computeGoalValue(self, state, action, depth):
        next_state = self.transition[(state, action)]
        tmp = np.max([self.computeGoalValueRecursive(next_state, a, depth-1) for a in self.values[next_state]])
        value =  self.rfunction[0][self.rfunction[(state, action)]] + self.kalman.gamma*self.transition[(state, action, next_state)]*tmp
        return value

    def computeGoalValueRecursive(self, state, a, depth):
        action = self.values[(state, self.values[state].index(a))]
        next_state = self.transition[(state, action)]
        if depth:
            tmp = np.max([self.computeGoalValueRecursive(next_state, a, depth-1) for a in self.values[next_state]])
            return self.rfunction[0][self.rfunction[(state, action)]] + self.kalman.gamma*self.transition[(state, action, next_state)]*tmp
        else:
            return self.rfunction[0][self.rfunction[(state, action)]] + self.kalman.gamma*self.transition[(state, action, next_state)]*np.max(self.kalman.values[0][self.kalman.values[(state, action)]])        
        
