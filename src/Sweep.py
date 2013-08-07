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

    def exploreParameters(self, ptr_model, param, values, correlation = 'Z'):
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
            tmp[v] = self.computeCorrelation(data, correlation)
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

    def computeCorrelation(self, model, correlation = 'Z'):
        """ Input should be dict(1:[], 2:[], 3:[])
        Return similarity estimate.
        """
        tmp = dict()
        for i in [1,2,3]:
            assert self.data_human[i].shape[1] == model[i].shape[1]
            m,n = self.data_human[i].shape
            tmp[i] = np.array([self.computeSingleCorrelation(self.data_human[i][:,j], model[i][:,j], correlation) for j in xrange(n)])
        return tmp
    

    def computeSingleCorrelation(self, human, model, case = 'JSD'):
        """Entry should be single-trial vector 
        of performance for each model
        case can be :
        - "JSD" : Jensen-Shannon Divergence
        - "C" : contingency coefficient of Pearson
        - "phi"  phi coefficient
        - "Z" : test Z 
        """
        h = len(human)
        m = len(model)
        a = float(np.sum(human == 1))
        b = float(np.sum(model == 1))
        obs = np.array([[a, h-a], [b,m-b]])
        h1 = float(a)/float(h)
        m1 = float(b)/float(m)
        h0 = 1-h1
        m0 = 1-m1
        if case == "JSD":
            if h1 == m1:
                return 1
            else:
                M1 = np.mean([h1, m1])
                M0 = np.mean([h0, m0])
                dm = self.computeSpecialKullbackLeibler(np.array([m0, m1]), np.array([M0, M1]))
                dh = self.computeSpecialKullbackLeibler(np.array([h0, h1]), np.array([M0, M1]))
                return 1-np.mean([dm, dh])

        elif case == "C":
            if h1 == m1:
                return 1
            else:
                chi, p, ddl, the = chi2_contingency(obs, correction=False)
                return (chi/(chi+(h+m)))**(0.5)
                #return np.sqrt(chi)
        
        elif case == "phi":
            if h1 == m1:
                return 1
            else:
                chi, p, ddl, the = chi2_contingency(obs, correction=False)
                return np.sqrt(chi)

        elif case == "Z":
            ph1 = float(a)/h
            pm1 = float(b)/m
            p = np.mean([ph1, pm1])
            if ph1 == pm1:
                return 1.0
            else:
                z = (np.abs(h1-m1))/(np.sqrt(p*(1-p)*((1/a)+(1/b))))
                return 1-(norm.cdf(z, 0, 1)-norm.cdf(-z, 0, 1))
    
    def computeSpecialKullbackLeibler(self, p, q):
        # Don;t use for compute Divergence Kullback-Leibler
        assert len(p) == len(q)
        tmp = 0
        for i in xrange(len(p)):
            if q[i] <> 0.0 and p[i] <> 0.0:
                tmp+=p[i]*np.log2(p[i]/q[i])
        return tmp


class Optimization():
    """
    class to optimize parameters for differents models
    try to minimize the difference between human and models data
    """

    def __init__(self, human, ptr_cats, nb_trials, nb_blocs):
        self.ptr_h = human
        self.human = human.responses['meg']
        self.data_human = extractStimulusPresentation2(self.ptr_h.responses['meg'], self.ptr_h.stimulus['meg'], self.ptr_h.action['meg'], self.ptr_h.responses['meg'])
        self.cats = ptr_cats
        self.nb_trials = nb_trials
        self.nb_blocs = nb_blocs
        self.correlation = "Z"
        
    def simulatedAnnealing(self, model, measure="Z"):
        self.correlation = measure
        p = model.getAllParameters()
        p_opt = p
        f_opt = self.evaluate(model, p_opt)
        T_finale = 1
        T = 100
        R = 100
        alpha = 0.95
        while T > T_finale:
            for i in xrange(R):
                new_p = self.generateSolution(p)
                print T, i, f_opt
                f_new = self.evaluate(model, new_p)
                f_old = self.evaluate(model, p)
                delta = f_new - f_old
                if delta < 0:
                    p = new_p
                    if f_new < f_opt:
                        p_opt = p
                        f_opt = f_new
                elif np.random.rand() <= np.exp(-(delta/T)):
                    p = new_p                    
            T *= alpha
            
        return p_opt

    def stochasticOptimization(self, model, measure ="Z", nb_iter=100):
        self.correlation = measure
        p = model.getAllParameters()
        f = self.evaluate(model, p)
        for i in xrange(nb_iter):
            new_p = self.generateRandomSolution(p)
            new_f = self.evaluate(model, new_p)
            print i, new_f, f
            if new_f < f:
                p = new_p
                f = new_f
        return p

    def generateRandomSolution(self, p):
        tmp = dict()
        for i in p.iterkeys():
            tmp[i] = p[i]
            tmp[i][1] = np.random.uniform(p[i][0],p[i][2])
        return tmp

    def generateSolution(self, p, width = 1):
        tmp = dict()
        for i in p.iterkeys():
            tmp[i] = p[i]
            tmp[i][1] = np.random.normal(p[i][1], width)
            tmp[i][1] = tmp[i][2] if tmp[i][1] > tmp[i][2] else tmp[i][1]
            tmp[i][1] = tmp[i][0] if tmp[i][1] < tmp[i][0] else tmp[i][1]                      
        return tmp

    def evaluate(self, model, p):
        model.setAllParameters(p)
        self.testModel(model)
        model.state = convertStimulus(np.array(model.state))
        model.action = convertAction(np.array(model.action))
        model.responses = np.array(model.responses)
        data = extractStimulusPresentation2(model.responses, model.state, model.action, model.responses)
        #return self.computeCorrelation(data)
        return self.computeAbsoluteDifference(data)
        
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
        model.updateValue(reward)

    def computeCorrelation(self, model):
        """ Input should be dict(1:[], 2:[], 3:[])
        Return similarity estimate.
        """
        tmp = 0.0
        for i in [1,2,3]:
            m,n = self.data_human[i].shape
            for j in xrange(n):
                tmp += 1-computeSingleCorrelation(self.data_human[i][:,j], model[i][:,j], self.correlation)
        return tmp

    def computeAbsoluteDifference(self, model):
        tmp = 0.0
        for i in [1,2,3]:
            tmp += np.sum((np.mean(self.data_human[i], 0)-np.mean(model[i], 0))**2)
        return tmp
