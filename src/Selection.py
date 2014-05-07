#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Selection.py

Class of for strategy selection when training

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
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
        self.bounds = dict({"gamma":[0.1, 1.0],
                            "beta":[1.0, 10.0],
                            "alpha":[0.1, 1.0],
                            "length":[6, 10],
                            "threshold":[0.1, 20.0],
                            "noise":[0.01, 1.0],
                            "gain":[0.01, 10.0],
                            "sigma":[0.01, 1.0]})                            

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
        self.pdf.append([])
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
        self.pA = 1/(1+((self.n_element-self.nb_inferences)*self.parameters['threshold'])*np.exp(-x*self.parameters['gain']))
        #self.pA = 1/(1+((self.n_element-self.nb_inferences)*self.parameters['gain'])*np.exp(-x))
        #self.pA = 1/(1+(self.n_element-self.nb_inferences)*np.exp(-x*self.parameters['gain']))        
        #self.pA = 1/(1+(self.n_element-self.nb_inferences)*np.exp(-x))
        #self.pA = 1/(1+((self.n_element-self.nb_inferences)/self.parameters['threshold'])*np.exp(-x/self.parameters['gain']))        
        return np.random.uniform(0,1) > self.pA
    
    def fusionModule(self):
        np.seterr(invalid='ignore')
        #w = (self.max_entropy-self.Hb)/self.max_entropy
        #self.values_net = w*self.p_a_mb+(1-w)*self.values_mf[self.current_state]
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

        while self.sigmoideModule():
            self.inferenceModule()
            self.evaluationModule()
        self.fusionModule()
        
        self.value.append(float(self.p_a[self.current_action]))
        H = -(self.p_a*np.log2(self.p_a)).sum()
        N = float(self.nb_inferences+1)
        
        self.reaction[-1].append(H*self.parameters['sigma']+np.log2(N))
        
        #self.reaction[-1].append((2*self.max_entropy-self.Hf-self.Hb)/float(self.nb_inferences+1))        
        #self.reaction[-1].append((2*self.max_entropy-self.Hf-self.Hb))
        #self.reaction[-1].append(float(self.nb_inferences+1))

        #rt = 1+((2*self.max_entropy-self.Hf-self.Hb)/(2*self.max_entropy))**0.2

        #self.reaction[-1].append(rt)


        #value = np.zeros(int(self.parameters['length']+1))        
        #pdf = np.zeros(int(self.parameters['length'])+1)

        #d = self.sigmoideModule()
        #pdf[self.nb_inferences] = float(self.pA)
        #self.fusionModule()
        #value[self.nb_inferences] = float(self.p_a[self.current_action])        

        #while self.nb_inferences < self.n_element:
        #     self.inferenceModule()
        #     self.evaluationModule()
        #     self.fusionModule()
        #     value[self.nb_inferences] = float(self.p_a[self.current_action])        
        #     d = self.sigmoideModule()
        #     pdf[self.nb_inferences] = float(self.pA)
        
        # pdf = np.array(pdf)
        # pdf[1:] = pdf[1:]*np.cumprod(1-pdf)[0:-1]
        #pdf = pdf/pdf.sum()
        
        # self.pdf.append(pdf)
        # self.value.append(value)
                
    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.Hf = computeEntropy(self.values_mf[self.current_state], self.parameters['beta'])
        self.nb_inferences = 0
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)

        #pdf = np.zeros(int(self.parameters['length'])+1)
        
        while self.sigmoideModule():
            #pdf[self.nb_inferences] = float(self.pA)
            self.inferenceModule()
            self.evaluationModule()
        #pdf[self.nb_inferences] = float(self.pA)            
        self.fusionModule()
        self.current_action = self.sample(self.p_a)
        self.value.append(float(self.p_a[self.current_action]))
        self.action[-1].append(self.current_action)                
        
        H = -(self.p_a*np.log2(self.p_a)).sum()
        N = float(self.nb_inferences+1)
        self.reaction[-1].append(H*self.parameters['sigma']+np.log2(N))
        
        # while self.nb_inferences < self.n_element:            
        #     self.inferenceModule()
        #     self.evaluationModule()
        #     d = self.sigmoideModule()
        #     pdf[self.nb_inferences] = float(self.pA)            
        
        # pdf[1:] = pdf[1:]*np.cumprod(1-pdf)[0:-1]
        # self.pdf[-1].append(pdf)
        return self.actions[self.current_action]

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
        self.bounds = dict({"gamma":[0.1, 1.0],
                            "beta":[1.0, 10.0],
                            "eta":[0.00001, 0.001],
                            "length":[6, 10],
                            "threshold":[0.01, -np.log2(1./self.n_action)], 
                            "noise":[0.01,1.0],
                            "sigma":[0.0,1.0],
                            "sigma_rt":[0.001, 1.0]})
                            #"sigma_ql":[0.00001, 1.0]})        
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
        #self.sigma = list()
        #self.sigma_test = list()

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
        #self.sigma_test.append([])
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
        #self.sigma = list()    
        #self.sigma_test = list()
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
        self.nb_inferences = 0
        self.predictionStep()
        values = self.softMax(self.values_mf[self.current_state])        
        t = self.n_action*self.current_state
        vpi = computeVPIValues(self.values_mf[self.current_state], self.covariance['cov'].diagonal()[t:t+self.n_action])

        if np.sum(vpi > self.reward_rate[self.current_state]):
            self.p = self.uniform[:,:,:]
            self.Hb = self.max_entropy            
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:
                self.inferenceModule()
                self.evaluationModule()
            values = self.p_a_mb/np.sum(self.p_a_mb)
        
        self.value.append(float(values[self.current_action]))
        H = -(values*np.log2(values)).sum()
        N = float(self.nb_inferences+1)
        self.reaction[-1].append(H*self.parameters['sigma_rt']+np.log2(N))
        
        
    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.nb_inferences = 0
        self.predictionStep()
        values = self.softMax(self.values_mf[self.current_state])
        t =self.n_action*self.current_state
        vpi = computeVPIValues(self.values_mf[self.current_state], self.covariance['cov'].diagonal()[t:t+self.n_action])
        self.vpi[-1].append(vpi)

        #pdf = np.zeros(int(self.parameters['length'])+1)
        #d = (np.sum(vpi > self.reward_rate[self.current_state])>0)*(self.nb_inferences < self.n_element)*1.0
        #pdf[self.nb_inferences] = 1.0-float(d)

        if np.sum(vpi > self.reward_rate[self.current_state]):
            self.p = self.uniform[:,:,:]
            self.Hb = self.max_entropy            
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:
                self.inferenceModule()
                self.evaluationModule()
                #d = (self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element)*1.0
                #pdf[self.nb_inferences] = 1.0-float(d)
            values = self.p_a_mb/np.sum(self.p_a_mb)
        self.current_action = self.sample(values)
        self.value.append(float(values[self.current_action]))
        self.action[-1].append(self.current_action)
        H = -(values*np.log2(values)).sum()
        N = float(self.nb_inferences+1)
        self.reaction[-1].append(H*self.parameters['sigma_rt']+np.log2(N))
        #self.sigma_test[-1].append([self.parameters['sigma_ql'], self.parameters['sigma_bwm']][int(self.nb_inferences != 0)])
        
        # while self.nb_inferences < self.n_element:            
        #     self.inferenceModule()
        #     self.evaluationModule()
        #     d = (self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element)*1.0
        #     pdf[self.nb_inferences] = 1.0-float(d)            

        # pdf = pdf/pdf.sum()
        # self.pdf.append(pdf)
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
    def __init__(self, states, actions, parameters={'length':1, 'w0':0.5}):
        # State Action Space        
        self.states=states
        self.actions=actions        
        #Parameters
        self.parameters = parameters
        self.n_action=int(len(actions))
        self.n_state=int(len(states))
        self.initial_entropy = -np.log2(1./self.n_action)
        self.bounds = dict({"length":[6, 10], 
                            "threshold":[0.01, self.initial_entropy], 
                            "noise":[0.01, 1.0],
                            "alpha":[0.01, 1.0],
                            "beta":[0.01, 7.0],
                            "gamma":[0.01, 1.0],                            
                            "sigma":[0.001, 1.0], 
                            "w0":[0.001, 1.0]})

        # Probability Initialization        
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)    
        self.p = None        
        self.p_a = None
        # Specific to collins
        self.min_1_C_ns = np.min([1.0, self.parameters['length']/float(self.n_state)])
        self.inv_n_a = 1.0/float(self.n_action)
        self.w = np.ones(self.n_state)*self.parameters['w0']*self.min_1_C_ns
        # Q-values model free
        self.values_mf = np.zeros((self.n_state, self.n_action))
        self.p_a_mf = None
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
        #self.sigma = list()

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
        self.p_a = np.ones(self.n_action)*(1./self.n_action)        
        self.w = np.ones(self.n_state)*self.parameters['w0']*self.min_1_C_ns
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
        self.p_a = np.ones(self.n_action)*(1./self.n_action)        
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
        self.p_a_mb = p_a_rs[:,1]/p_a_rs[:,0]
        self.p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)
        self.entropy = -np.sum(self.p_a_mb*np.log2(self.p_a_mb))

    def fusionModule(self):
        np.seterr(invalid='ignore')
        self.p_a_mf = np.exp(self.values_mf[self.current_state]*float(self.parameters['beta']))
        self.p_a_mf = self.p_a_mf/np.sum(self.p_a_mf)
        self.p_a = (1.0-self.w[self.current_state])*self.p_a_mf + self.w[self.current_state]*self.p_a_mb

    def updateWeight(self, r):
        if r:
            p_wmc = self.min_1_C_ns * self.p_a_mb[self.current_action] + (1.0 - self.min_1_C_ns)*self.inv_n_a
            p_rl = self.p_a_mf[self.current_action]
        else:
            p_wmc = self.min_1_C_ns * (1.0 - self.p_a_mb[self.current_action]) + (1.0 - self.min_1_C_ns)*self.inv_n_a
            p_rl = 1.0 - self.p_a_mf[self.current_action]
        self.w[self.current_state] = (p_wmc*self.w[self.current_state])/(p_wmc*self.w[self.current_state] + p_rl * (1.0 - self.w[self.current_state]))            

    def computeValue(self, s, a):
        self.current_state = s
        self.current_action = a
        self.p = self.uniform[:,:,:]
        self.entropy = self.initial_entropy
        self.nb_inferences = 0     

        while self.entropy > self.parameters['threshold'] and self.nb_inferences < self.n_element:                    
            self.inferenceModule()
            self.evaluationModule()                    

        self.fusionModule()

        self.value.append(self.p_a[self.current_action])
        H = -(self.p_a*np.log2(self.p_a)).sum()
        N = float(self.nb_inferences+1)
        self.reaction[-1].append(H*self.parameters['sigma']+np.log2(N))

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        self.entropy = self.initial_entropy
        self.nb_inferences = 0             

        while self.entropy > self.parameters['threshold'] and self.nb_inferences < self.n_element:
            self.inferenceModule()
            self.evaluationModule()

        self.fusionModule()

        self.current_action = self.sample(self.p_a)
        self.value.append(float(self.p_a[self.current_action]))
        self.action[-1].append(self.current_action)

        H = -(self.p_a*np.log2(self.p_a)).sum()
        N = float(self.nb_inferences+1)
        self.reaction[-1].append(H*self.parameters['sigma']+np.log2(N))

        return self.actions[self.current_action]

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
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0        
        delta = float(r)+self.parameters['gamma']*np.max(self.values_mf[self.current_state])-self.values_mf[self.current_state, self.current_action]                
        self.values_mf[self.current_state, self.current_action] = self.values_mf[self.current_state, self.current_action]+self.parameters['alpha']*delta
        # Specific to Collins model
        self.updateWeight(r)


class Keramati():
    """Class that implement Keramati models for action selection
    Use to replicate exp 1 from Keramati & al, 2011
    """
    
    def __init__(self, kalman,depth,phi, rau, sigma, tau):
        self.kalman = kalman
        self.depth = depth
        self.phi = phi
        self.rau = rau
        self.sigma = sigma
        self.tau = tau
        self.gamma = self.kalman.parameters['gamma']
        self.beta = self.kalman.parameters['beta']
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
        n = self.kalman.states.index(self.state)
        t = len(self.actions)*n
        vpi = computeVPIValues(self.kalman.values[n], self.kalman.covariance['cov'].diagonal()[t:t+len(self.actions)])        
        
        for i in range(len(vpi)):
            if vpi[i] >= self.rrate[-1]*self.tau:
                depth = self.depth
                self.values[0][self.values[(self.state, self.actions[i])]] = self.computeGoalValue(self.state, self.actions[i], depth)
            else:
                self.values[0][self.values[(self.state, self.actions[i])]] = self.kalman.values[n, i]

        self.action = getBestActionSoftMax(state, self.values, self.beta)
        return self.action

    def updateValues(self, reward, next_state):
        self.updateRewardRate(reward, delay = 0.0)
        self.kalman.current_state = self.kalman.states.index(self.state)
        self.kalman.current_action = self.kalman.actions.index(self.action)        
        self.kalman.updateValue(reward)
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
        tmp = np.max([self.computeGoalValueRecursive(next_state, a, depth-1) for a in xrange(len(self.actions))])
        value =  self.rfunction[0][self.rfunction[(state, action)]] + self.gamma*self.transition[(state, action, next_state)]*tmp
        return value

    def computeGoalValueRecursive(self, state, a, depth):
        action = self.actions[a]
        next_state = self.transition[(state, action)]
        if depth:
            tmp = np.max([self.computeGoalValueRecursive(next_state, a, depth-1) for a in xrange(len(self.actions))])
            return self.rfunction[0][self.rfunction[(state, action)]] + self.gamma*self.transition[(state, action, next_state)]*tmp
        else:
            return self.rfunction[0][self.rfunction[(state, action)]] + self.gamma*self.transition[(state, action, next_state)]*np.max(self.kalman.values[self.kalman.states.index(state)])        
        
