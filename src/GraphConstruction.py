#!/usr/bin/python
# encoding: utf-8
"""
GraphConstruction.py

Class that implement a trees construction based on
Color Association Experiments from Brovelli & al 2011
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
sys.path.append("../../src")
import os
import numpy as np
from fonctions import *
import matplotlib.pyplot as plt

class ModelBased():
    """Class that implement a trees construction based on
    Color Association Experiments from Brovelli & al 2011
    """
    
    def __init__(self, state, action):
        self.state = state
        self.g, self.action = self.initializeTree(state, action)
            
    def reinitialize():
        self.__init__(self.state, self.action)

    def initializeTree(self, state, action):
        g = dict()
        dict_action = dict()
        for s in state:
            g[s] = dict()            
            g[s][0] = np.ones(len(action))*(1/float(len(action)))
            for a in range(1, len(action)+1):
                g[s][a] = dict()
                dict_action[a] = action[a-1]
                dict_action[action[a-1]] = a
        return g, dict_action

    def SoftMax(self, values, beta=1):
        #WARNING return 1 not 0 for indicing
        tmp = np.exp(values*float(beta))
        tmp = tmp/float(np.sum(tmp))
        tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
        return np.sum(np.array(tmp) < np.random.rand())

    def chooseAction(self, state, beta):
        position = self.g[state]
        ind_action = self.SoftMax(position[0], beta)
        self.mental_path = [ind_action]
        if len(position[ind_action]) == 0:
            return self.action[ind_action]
        else:
            return self.chooseActionRecursive(position[ind_action])

    def chooseActionRecursive(self, position):
        ind_action = self.SoftMax(position[0], beta)
        self.mental_path.append(ind_action)
        if len(position[ind_action]) == 0:
            return self.action[ind_action]
        else:
            return self.chooseActionRecursively(position[ind_action])        

    def updateTrees(self, state, action, reward):
        self.g[state][0] = self.g[state]*0.0
        id_action = self.action[action]

    def branch(
                                        
