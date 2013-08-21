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
        self.transition = createTransitionDict(['s0','s0','s1','s1'],
                                               ['pl','em','pl','em'],
                                               ['s1','s0','s0','s0'], 's0') #<====VERY BAD==============    NEXT_STATE = TRANSITION[(STATE, ACTION)]
        
        self.vpi = list()
        self.rrate = [0.0]
        self.responses = list()
        self.state = None
        self.action = None
                
    def chooseAction(self, state):
        self.state = state
        vpi = computeVPIValues(self.kalman.values[0][self.kalman.values[state]], self.kalman.covariance['cov'].diagonal()[self.kalman.values[state]])
        for i,j in zip(vpi, self.actions):
            if i < self.rrate[-1]*self.tau:
                self.values[0][self.values[(state, j)]] = self.kalman.values[0][self.kalman.values[(state,j)]]
            else:
                depth = self.depth #TO CHECK IF PYTHON IS REF OR NOT
                self.values[0][self.values[(state, j)]] = self.computeGoalValue(state, j, depth-1)
                
        self.action = getBestActionSoftMax(state, self.values, self.kalman.beta)
        return self.action

    def updateValues(self, reward):
        self.kalman.updatePartialValue(self.state, self.action, reward)
        self.updateRewardRate(reward, delay = 0.0)
        self.updateRewardFunction(self.state, self.action, reward)
        self.updateTransitionFunction(self.state, self.action)

    def updateRewardRate(self, reward, delay = 0.0):
        self.rrate.append(((1-self.sigma)**(1+delay))*self.rrate[-1]+self.sigma*reward)

    def updateRewardFunction(self, state, action, reward):
        print state, action
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
        #Cheating again. Only one s' is assumed to simplify
        nextstate = self.transition[(state, action)]
        if depth:        
            tmp = self.transition[(state, action, nextstate)]*np.max([self.computeGoalValue(state, self.values[(state,a)], depth-1) for a in range(len(self.values[nextstate]))])
            return self.rfunction[0][self.rfunction[(state, action)]]+self.kalman.gamma*tmp
        else:
            tmp = self.transition[(state, action, nextstate)]*np.max(self.kalman.values[0][self.kalman.values[nextstate]])
            return self.rfunction[0][self.rfunction[(state, action)]]+self.kalman.gamma*tmp

        
