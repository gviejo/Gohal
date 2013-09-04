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

class KSelection():
    """Class that implement Keramati models for action selection
    Model-based must be provided
    Specially tuned for Brovelli experiment so beware
    """
    def __init__(self, kalman, based, sigma, tau):
        self.kalman = kalman
        self.based = based
        self.sigma = sigma; self.tau = tau
        self.actions = kalman.actions; 
        self.states = kalman.states        
        self.values = createQValuesDict(self.states, self.actions)
        self.rfunction = createQValuesDict(kalman.states, kalman.actions)
        #self.vpi = {i:list() for i in self.states}
        self.vpi = list()
        self.rrate = list()
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()


    def initialize(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.kalman.initialize()
        self.based.initialize()
        self.responses.append([])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.rrate.append([0.0])
        self.vpi.append([])
        #[self.vpi[s].append([]) for s in self.vpi.iterkeys()]

    def initializeList(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.rfunction = createQValuesDict(self.states, self.actions)
        #self.vpi = {i:list() for i in self.states}
        self.vpi = list()
        self.rrate = list()
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()


    def chooseAction(self, state):
        self.state[-1].append(state)
        self.kalman.predictionStep()
        vpi = computeVPIValues(self.kalman.values[0][self.kalman.values[state]], self.kalman.covariance['cov'].diagonal()[self.kalman.values[state]])
        model_based_value = self.based.computeValue(state) #WRONG since Q^G(s,a) should be computed after the decision is made
        self.vpi[-1].append(vpi)
        for i in range(len(vpi)):
            if vpi[i] >= self.rrate[-1][-1]*self.tau:
                #use Model-based                
                self.values[0][self.values[(state,self.actions[i])]] = model_based_value[i]
            else:
                #use Model-free
                self.values[0][self.values[(state, self.actions[i])]] = self.kalman.values[0][self.kalman.values[(state,self.actions[i])]]

        action = getBestActionSoftMax(state, self.values, self.kalman.beta)                
        self.action[-1].append(action)
        return action

    def updateValue(self, reward):
        self.responses[-1].append((reward==1)*1)
        self.updateRewardRate((reward==1)*1, delay = 0.0)
        self.kalman.updatePartialValue(self.state[-1][-1], self.action[-1][-1], self.state[-1][-1], reward)
        self.based.updatePartialValue(self.state[-1][-1], self.action[-1][-1], reward)

    def updateRewardRate(self, reward, delay = 0.0):
        self.rrate[-1].append(((1-self.sigma)**(1+delay))*self.rrate[-1][-1]+self.sigma*reward)

    def getAllParameters(self):
        tmp = dict({'tau':[0.0, self.tau, 1.0],
                    'sigma':[0.0, self.sigma, 1.0]})
        tmp.update(self.kalman.getAllParameters())
        tmp.update(self.based.getAllParameters())
        return tmp

            
class CSelection():
    """Class that implement Collins models for action selection
    Model-based must be provided
    Specially tuned for Brovelli experiment so beware
    """
    def __init__(self, kalman, based):
        self.kalman = kalman
        self.based = based
        self.actions = kalman.actions; 
        self.states = kalman.states        
        self.values = createQValuesDict(self.states, self.actions)
        self.weight = createQValuesDict(self.states, self.actions)
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()

    def initialize(self):
        self.kalman.initialize()
        self.based.initialize()
        self.responses.append([])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.values = createQValuesDict(self.states, self.actions)
        self.weight = createQValuesDict(self.states, self.actions)

    def initializeList(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.weight = createQValuesDict(self.states, self.actions)
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()


    def computeRewardLikelihood(self, p):
        # p is p(a,s,r) from B-WM
        p_as = np.sum(p, axis = 2)
        p_as = p_as/np.sum(p_as)
        p_r_bwm = p/np.reshape(np.repeat(p_as, 2), p.shape)
        p_r_bwm = p_r_bwm/np.reshape(np.repeat(np.sum(p_r_bwm, 2), 2), p_r_bwm.shape)
        p_r_rl = np.zeros((3,5,2))
        p_r_rl[:,:,0] = np.reshape(self.kalman.values[0], (len(self.states), len(self.actions)))
        p_r_rl[:,:,1] = 1-np.reshape(self.kalman.values[0], (len(self.states), len(self.actions)))
        p_r_rl = p_r_rl/np.reshape(np.repeat(np.sum(p_r_rl, 2), 2), p_r_rl.shape)
        return p_r_bwm, p_r_rl

    def computeWeight(self, reward):
        (p_r_bwm,p_r_rl) = self.computeRewardLikelihood(self.based.p)
        tmp = np.reshape(self.weight[0], (len(self.states), len(self.actions)))
        self.weight = np.flatten((p_r_bwm[:,:,reward]*tmp)/(p_r_bwm[:,:,reward]*tmp + p_r_rl[:,:,reward]*(1-tmp)))        
            
    def chooseAction(self, state):
        self.state[-1].append(state)
        self.kalman.predictionStep()
        
        model_based_values = self.based.computeValue(state) #WRONG since Q^G(s,a) should be computed after the decision is made
        model_based_values = model_based_values/float(np.sum(model_based_values))
        tmp = np.exp(self.kalman.values[0][self.kalman.values[state]]*float(self.kalman.beta))
        model_free_values =  tmp/float(np.sum(tmp))
        
        self.values[0][self.values[state]] = (1-self.weight[self.weight[state]])*model_free_values + self.weight[self.weight[state]]*model_based_values
        action = getBestAction(state, values)
        self.action[-1].append(action)
        return action

    def updateValue(self, reward):
        self.responses[-1].append((reward==1)*1)
        self.computeWeight((reward==1)*1)        
        self.kalman.updatePartialValue(self.state[-1][-1], self.action[-1][-1], self.state[-1][-1], reward)
        self.based.updatePartialValue(self.state[-1][-1], self.action[-1][-1], reward)

    def getAllParameters(self):
        tmp = dict({'tau':[0.0, self.tau, 1.0],
                    'sigma':[0.0, self.sigma, 1.0]})
        tmp.update(self.kalman.getAllParameters())
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
        
