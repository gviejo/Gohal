#!/usr/bin/python
# encoding: utf-8
"""
Sweep.py

Class to explore parameters

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *
from scipy.stats import chi2_contingency




class Sweep_performances():
    """
    class to explore parameters by comparing between human peformances 
    and models performances for different range of data
    Initialize class with human responses
    
    """
    
    def __init__(self, human, ptr_cats, nb_trials, nb_blocs):
        self.human = human
        self.cats = ptr_cats
        self.nb_trials = nb_trials
        self.nb_blocs = nb_blocs

    def exploreParameters(self, ptr_model, param, values):
        tmp = dict()
        true_value = ptr_model.getParameter(param)
        for v in values:
            #parameters
            ptr_model.setParameter(param, v)
            #Learning
            ptr_model.initializeList()
            for i in xrange(self.nb_blocs):
                self.cats.reinitialize()
                ptr_model.initialize()
                for j in xrange(self.nb_trials):
                    self.iterationStep(j, ptr_model, False)

            #p-value
            w = self.computeSimilarity(np.array(ptr_model.responses))
            tmp[v] = w
            ptr_model.setParameter(param, true_value)
        return tmp
        
    def iterationStep(self, iteration, model, display = True):
        state = self.cats.getStimulus(iteration)
        action = model.chooseAction(state)
        reward = self.cats.getOutcome(state, action)
        if model.__class__.__name__ == 'TreeConstruction':
            model.updateTrees(state, reward)
        else:
            model.updateValue(reward)

    def computeSimilarity(self, model):
        assert self.human.shape == model.shape
        m,n = self.human.shape
        tmp = []
        for i in xrange(n):
            a = np.sum(self.human[:,i] == 1)
            b = np.sum(model[:,i] == 1)
            obs = np.array([[a, m-a], [b,m-b]])
            #if a == 0 or b == 0:
            if a == b:
                tmp.append(1)
            else:
                chi, p, ddl, the = chi2_contingency(obs, correction=False)
                tmp.append(p)
        return np.log2(np.array(tmp))
