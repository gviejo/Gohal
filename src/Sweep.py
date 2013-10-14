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
from scipy.stats import norm
import scipy.optimize

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
    

    def computeSingleCorrelation(self, human, model, case = 'Z'):
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

    def evaluate(self, model, p, correlation):
        model.setAllParameters(p)
        self.testModel(model)
        model.state = convertStimulus(np.array(model.state))
        model.action = convertAction(np.array(model.action))
        model.responses = np.array(model.responses)
        data = extractStimulusPresentation2(model.responses, model.state, model.action, model.responses)
        return self.computeCorrelation(data, correlation)
        #return self.computeAbsoluteDifference(data)
        
    def testModel(self, ptr_model):
        ptr_model.initializeList()
        for i in xrange(self.nb_blocs):
            #sys.stdout.write("\r Testing model | Blocs : %i" % i); sys.stdout.flush()                        
            self.cats.reinitialize()
            ptr_model.initialize()
            for j in xrange(self.nb_trials):
                self.iterationStep(j, ptr_model, False)
        
    def iterationStep(self, iteration, model, display = True):
        state = self.cats.getStimulus(iteration)
        action = model.chooseAction(state)
        reward = self.cats.getOutcome(state, action)
        model.updateValue(reward)

    def computeCorrelation(self, model, correlation):
        """ Input should be dict(1:[], 2:[], 3:[])
        Return similarity estimate.
        """
        tmp = 0.0
        for i in [1,2,3]:
            m,n = self.data_human[i].shape
            for j in xrange(n):
                tmp += computeSingleCorrelation(self.data_human[i][:,j], model[i][:,j], correlation)
        return tmp

    def computeAbsoluteDifference(self, model):
        tmp = 0.0
        for i in [1,2,3]:
            tmp += np.sum((np.mean(self.data_human[i], 0)-np.mean(model[i], 0))**2)
        return tmp


class Likelihood():
    """
    Optimization with scipy.optimize.fmin
    See : Trial-by-trial data analysis using computational models, Daw, 2009
    """
    def __init__(self, human, n_run):
        self.X = human.subject['meg']
        self.subject = self.X.keys()
        self.n_run = n_run
        self.best_parameters = None
        self.start_parameters = None
        self.p = None
        self.p_order = None        
        self.model = None
        self.current_subject = None
        self.cvt = dict({i:'s'+str(i) for i in [1,2,3]})
        self.lower = None
        self.upper = None
        self.ranges = None

    def searchStimOrder(self):
        for s in self.subject:            
            for b in self.X[s].iterkeys():                
                sar = self.X[s][b]['sar']
                tmp = np.ones((sar.shape[0], 1))
                # search for order
                for j in [1,2,3]:
                    if len(np.where((sar[:,2] == 1) & (sar[:,0] == j))[0]):
                        correct = np.where((sar[:,2] == 1) & (sar[:,0] == j))[0][0]
                        t = len(np.where((sar[0:correct,2] == 0) & (sar[0:correct,0] == j))[0])
                        if t == 1:
                            first = j
                            tmp[np.where(sar[:,0] == first)[0][0]] = 0
                        elif t == 3:
                            second = j
                            tmp[np.where(sar[:,0] == second)[0][0]] = 0
                        elif t >= 4:
                            third = j
                            tmp[np.where(sar[:,0] == third)[0][0]] = 0
                        else:
                            print "Class Sweep.Likelihood.searchStimOrder : unable to find nb errors"
                            sys.exit()
                self.X[s][b]['sar'] = np.hstack((self.X[s][b]['sar'], tmp))

    def computeLikelihood(self, p):
        """ Maximize log-likelihood for one subject
            based only on performances
        """
        llh = 0.0
        for i, v in zip(self.p_order, p):
            self.model.setParameter(i, v)
        self.model.initializeList()
        for bloc in self.X[self.current_subject].iterkeys():
            self.model.initialize()                        
            for trial in self.X[self.current_subject][bloc]['sar']:                
                state = self.cvt[trial[0]]                
                true_action = trial[1]-1
                values = self.model.computeValue(state)
                llh = llh + np.log(values[true_action])*trial[3]
                self.model.current_action = true_action
                self.model.updateValue(trial[2])                        
        return llh

    def computeLikelihoodAll(self, p):
        """ Maximize log-likelihood based on 
            reaction time and performances
        """
        llh = 0.0
        for i,v in zip(self.p_order, p):
            self.model.setParameter(i, v)
        self.model.initializeList()
        for bloc in self.X[self.current_subject].iterkeys():
            self.model.initialize()
            for trial in self.X[self.current_subject][bloc]['sar']:
                print 1

    def set(self, ptr_m, subject):
        self.model = ptr_m
        self.current_subject = subject
        self.p = self.model.getAllParameters()
        self.p_order = self.p.keys()
        self.lower = np.array([self.p[i][0] for i in self.p_order])
        self.upper = np.array([self.p[i][2] for i in self.p_order])
        self.ranges = tuple(map(tuple, np.array([self.lower, self.upper]).transpose()))

    def optimize(self, ptr_m):
        self.searchStimOrder()
        self.best_parameters = dict({s:list() for s in self.subject})
        self.start_parameters = dict({s:list() for s in self.subject})
        self.model = ptr_m                
        self.p = self.model.getAllParameters()
        self.p_order = self.p.keys()
        self.lower = np.array([self.p[i][0] for i in self.p_order])
        self.upper = np.array([self.p[i][2] for i in self.p_order])
        self.ranges = tuple(map(tuple, np.array([self.lower, self.upper]).transpose()))
        for s in self.subject:
            self.current_subject = s            
            for i in xrange(self.n_run):                
                p_start = self.generateStart()                
                # new_p = scipy.optimize.minimize(fun=self.computeLikelihood,
                #                                 x0=p_start,
                #                                 method='L-BFGS-B',
                #                                 jac=None,
                #                                 hess=None,
                #                                 hessp=None,
                #                                 bounds=self.ranges)
                new_p = scipy.optimize.fmin(func=self.computeLikelihood,
                                            x0=p_start,
                                            maxiter=10000,
                                            maxfun=10000,
                                            xtol=0.01,
                                            ftol=0.01,
                                            disp=True)
                                            #retall=True)                
                # new_p = scipy.optimize.anneal(func=self.computeLikelihood,
                #                               x0=p_start,
                #                               schedule='fast',
                #                               lower=self.lower,
                #                               upper=self.upper,
                #                               disp=True)
                # new_p = scipy.optimize.brute(func=self.computeLikelihood,
                #                              ranges=self.ranges,
                #                              disp=True,
                #                              Ns=20,
                #                              full_output=False)
                self.best_parameters[s].append(new_p)
                self.start_parameters[s].append(p_start)
            self.best_parameters[s] = np.array(self.best_parameters[s])
            self.start_parameters[s] = np.array(self.start_parameters[s])
        return self.best_parameters, self.start_parameters

    def generateStart(self):
        return [np.random.uniform(self.p[i][0], self.p[i][2]) for i in self.p_order]

