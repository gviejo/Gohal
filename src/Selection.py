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
    def __init__(self, free, based, sigma, tau):
        self.free = free
        self.based = based
        self.sigma = sigma; self.tau = tau
        self.actions = free.actions; 
        self.states = free.states        
        self.values = createQValuesDict(self.states, self.actions)
        self.rrfunc = {i:0.0 for i in self.states}
        self.vpi = list()
        self.rrate = list()
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.model_used = list()
        self.n_inf = list()

    def initialize(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.free.initialize()
        self.based.initialize()
        self.responses.append([])
        self.action.append([])
        self.state.append([])
        self.reaction.append([])
        self.rrate.append([])
        self.vpi.append([])
        self.model_used.append([])
        self.rrfunc = {i:0.0 for i in self.states}
        self.n_inf.append([])

    def initializeList(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.rrfunc = {i:0.0 for i in self.states}
        self.vpi = list()
        self.rrate = list()
        self.state=list()
        self.answer=list()
        self.responses=list()
        self.action=list()
        self.reaction=list()
        self.model_used = list()
        self.n_inf = list()

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.free.predictionStep()
        vpi = computeVPIValues(self.free.values[0][self.free.values[state]], self.free.covariance['cov'].diagonal()[self.free.values[state]])
        self.vpi[-1].append(vpi)
        
        # Decision True => Model based | False => Model free
        model_used = vpi > self.rrfunc[state]
        
        # copy Model-free value
        self.values[0][self.values[state]] = self.free.values[0][self.free.values[state]]
        
        # replace with Model-based value only if needed
        if True in model_used:            
            model_based_value = self.based.computeValue(state)
            self.values[0][np.array(self.values[state])[model_used]] = model_based_value[model_used]
            
        # choose Action 
        action = getBestActionSoftMax(state, self.values, self.free.beta)
        
        self.action[-1].append(action)
        self.model_used[-1].append(float(np.sum(model_used))/len(self.actions))
        self.n_inf[-1].append(len(self.based.p_s)*(float(np.sum(model_used))/len(self.actions)))
        return action

    def updateValue(self, reward):
        self.responses[-1].append((reward==1)*1)
        self.updateRewardRate((reward==1)*1, delay = 0.0)
        self.free.updatePartialValue(self.state[-1][-1], self.action[-1][-1], self.state[-1][-1], reward)
        self.based.updatePartialValue(self.state[-1][-1], self.action[-1][-1], reward)

    def updateRewardRate(self, reward, delay = 0.0):
        #self.rrate[-1].append(((1-self.sigma)**(1+delay))*self.rrate[-1][-1]+self.sigma*reward)
        self.rrfunc[self.state[-1][-1]] = ((1-self.sigma)**(1+delay))*self.rrfunc[self.state[-1][-1]]+self.sigma*reward
        self.rrate[-1].append(self.rrfunc[self.state[-1][-1]])

    def getAllParameters(self):
        tmp = dict({'tau':[0.0, self.tau, 1.0],
                    'sigma':[0.0, self.sigma, 1.0]})
        tmp.update(self.free.getAllParameters())
        tmp.update(self.based.getAllParameters())
        return tmp

            
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
        
