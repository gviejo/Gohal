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
    
    def __init__(self, state, action, noise):
        self.noise = noise
        self.state = state
        self.n_action = len(action)
        self.g, self.action = self.initializeTree(state, action)
        self.mental_path = []

    def reinitialize(self, state, action):
        self.__init__(state, action, self.noise)

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

    def sample(self, values):
        #WARNING return 1 not 0 for indicing
        # values are probability
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())

    def chooseAction(self, ptr_trees):
        id_action = ptr_trees.keys()[self.sample(ptr_trees[0])]
        if id_action == 0:
            sys.stderr.write("End of trees\n")
            sys.exit()
        self.mental_path.append(id_action)
        if len(ptr_trees[id_action]):
            return self.chooseAction(ptr_trees[id_action])
        else:
            return self.action[id_action]
            
    def updateTrees(self, state, reward):        
        if reward <> 1:
            self.extendTrees(self.mental_path, self.mental_path, self.g[state])
        elif reward == 1:
            self.reinforceTrees(self.mental_path, self.mental_path, self.g[state])
        #TO ADD NOISE TO OTHERS STATE
        if self.noise:
            for s in set(self.state)-set([state]):
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

    def addNoise(self, ptr_tree):
        if 0 in ptr_tree.keys():
            #sys.stdin.read(1)
            tmp = np.random.normal(ptr_tree[0], np.ones(len(ptr_tree[0]))*self.noise, len(ptr_tree[0]))
            ptr_tree[0] = tmp/np.sum(tmp)
            for k in ptr_tree.iterkeys():
                if type(ptr_tree[k]) == dict and len(ptr_tree[k].values()) > 0:
                    self.addNoise(ptr_tree[k])
