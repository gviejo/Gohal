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
import networkx as nx
import matplotlib.pyplot as plt

class ModelBased():
    """Class that implement a trees construction based on
    Color Association Experiments from Brovelli & al 2011
    """
    
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.g = dict()
        for s in state:
            self.g[s] = nx.Graph()
        
        self.g = nx.Graph()
        self.g.add_nodes_from(state)
        for i in range(3):
            self.g.add_nodes_from(action)
        self.g.add_edges_from(self.computeEdgesBunch(state, action))
    
    def reinitialize():
        self.__init__(self.state, self.action)

    def computeEdgesBunch(self, state, action):
        tmp = []
        for i in state:
            for j in action:
                tmp.append((i, j,{'weight':1.0/len(action)}))
        return tmp

    def display(self):        
        nx.draw(self.g)
        plt.show()

    def chooseAction(self):
        
        
    def updateTrees(self):
        
