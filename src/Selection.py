#!/usr/bin/python
# encoding: utf-8
"""
Selection.py

Class of for model selection when training

first model <- Keramati et al, 2011

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *

class Keramati():
    """Class that implement Keramati models for action selection
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
        
