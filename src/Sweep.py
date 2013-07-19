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
from scipy.stats.contingency import expected_freq
from scipy.stats import norm


class Sweep_performances():
    """
    class to explore parameters by comparing between human peformances 
    and models performances for different range of data
    Initialize class with human responses
    
    """
    def __init__(self):
        return None
    
    def __init__(self, human, ptr_cats, nb_trials, nb_blocs):
        self.ptr_h = human
        self.human = human.responses['meg']
        self.data_human = extractStimulusPresentation2(self.ptr_h.responses['meg'], self.ptr_h.stimulus['meg'], self.ptr_h.action['meg'], self.ptr_h.responses['meg'])
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
            self.testModel(ptr_model)
            #COmparing with human
            ptr_model.state = convertStimulus(np.array(ptr_model.state))
            ptr_model.action = convertAction(np.array(ptr_model.action))
            ptr_model.responses = np.array(ptr_model.responses)
            data = extractStimulusPresentation2(ptr_model.responses, ptr_model.state, ptr_model.action, ptr_model.responses)
            tmp[v] = self.computeCorrelation(data)
            ptr_model.setParameter(param, true_value)
        return tmp

    def testModel(self, ptr_model):
        ptr_model.initializeList()
        for i in xrange(self.nb_blocs):
            self.cats.reinitialize()
            ptr_model.initialize()
            for j in xrange(self.nb_trials):
                self.iterationStep(j, ptr_model, False)
        
    def iterationStep(self, iteration, model, display = True):
        state = self.cats.getStimulus(iteration)
        action = model.chooseAction(state)
        reward = self.cats.getOutcome(state, action)
        if model.__class__.__name__ == 'TreeConstruction':
            model.updateTrees(state, reward)
        else:
            model.updateValue(reward)

    def computeCorrelation(self, model):
        """ Input should be dict(1:[], 2:[], 3:[])
        Return 1 - (divergence of Jensen-Shannon)
        """
        tmp = dict()
        for i in [1,2,3]:
            assert self.data_human[i].shape[1] == model[i].shape[1]
            m,n = self.data_human[i].shape
            tmp[i] = np.array([self.computeSingleCorrelation(self.data_human[i][:,j], model[i][:,j]) for j in xrange(n)])
        return tmp
    
    def computeCorrelation2(self, model):
        """ Return p value using chi-2 test
        """
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
                #tmp.append(np.sqrt(chi/(2*m)))
        return np.log2(np.array(tmp))
    

    def computeSingleCorrelation4(self, human, model):
        """Entry should be single-trial vector 
        of performance for each model
        uSING Jensen Shannon divergence
        """
        h = len(human)
        m = len(model)
        h1 = np.sum(human == 1)/float(h)
        m1 = np.sum(model == 1)/float(m)
        h0 = 1-h1
        m0 = 1-m1
        M1 = np.mean([h1, m1])
        M0 = np.mean([h0, m0])
        dm = self.computeSpecialKullbackLeibler(np.array([m0, m1]), np.array([M0, M1]))
        dh = self.computeSpecialKullbackLeibler(np.array([h0, h1]), np.array([M0, M1]))
        return 1-np.mean([dm, dh])


    def computeSingleCorrelation3(self, human, model):
        """Entry should be single-trial vector 
        of performance for each model
        Use contingency coefficient of Pearson C"""
        m = len(human)
        n = len(model)
        a = np.sum(human == 1)
        b = np.sum(model == 1)
        obs = np.array([[a, m-a], [b,n-b]])
        if a == b or (m-a) == (n-b):
            return 1
        else:
            chi, p, ddl, the = chi2_contingency(obs, correction=False)
            return (chi/(chi+(n+m)))**(0.5)
            #return np.sqrt(chi)
                                                

    def computeSingleCorrelation2(self, human, model):
        """Entry should be single-trial vector 
        of performance for each model
        Use phi coefficient """
        m = len(human)
        n = len(model)
        a = np.sum(human == 1)
        b = np.sum(model == 1)
        obs = np.array([[a, m-a], [b,n-b]])
        if a == b:
            return 1
        else:
            chi, p, ddl, the = chi2_contingency(obs, correction=False)
            return np.sqrt(chi)
            
    def computeSingleCorrelation(self, human, model):
        """Entry should be single-trial vector 
        of performance for each model
        Compute the difference between the two frequency 
        using normal distribution 
        """
        h = len(human)
        m = len(model)
        a = float(np.sum(human == 1))
        b = float(np.sum(model == 1))
        ph1 = float(a)/h
        pm1 = float(b)/m
        p = np.mean([ph1, pm1])
        if ph1 == pm1:
            return 0.5
        else:
            e = (np.abs(ph1-pm1))/(np.sqrt(p*(1-p)*((1/a)+(1/b))))
            return 1-norm.cdf(e, 0, 1)
    
    def computeSpecialKullbackLeibler(self, p, q):
        # Don;t use for compute Divergence Kullback-Leibler
        assert len(p) == len(q)
        tmp = 0
        for i in xrange(len(p)):
            if q[i] <> 0.0 and p[i] <> 0.0:
                tmp+=p[i]*np.log2(p[i]/q[i])
        return tmp

