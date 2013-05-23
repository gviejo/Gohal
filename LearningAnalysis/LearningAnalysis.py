#!/usr/bin/python
# encoding: utf-8
"""
LearningAnalysis.py

class that implement the state space model from Smith et al, 2004
'Dynamical analysis of learning in behavioral experiments'

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import scipy.integrate as sp
import numpy as np

class SSLearning():
    """  class that implement the state space model from Smith et al, 2004
    'Dynamical analysis of learning in behavioral experiments'"""
    
    def __init__(self, K, pOutset):
        self.K = K
        self.pO = pOutset
        self.mu = np.log(pOutset/(1-pOutset))
        self.p = np.zeros((K+1))
        self.x = np.zeros((K+1))
        self.x_old = np.zeros((K+1))
        self.x_new = np.zeros((K+1))
        self.v = np.zeros((K+1))
        self.v_old = np.zeros((K+1))
        self.v_new = np.zeros((K+1))
        self.A = np.zeros((K))
        self.p_final = np.zeros((K-1))
        self.criterion = 1e-8
        self.var = []
        self.x_first = []
        self.r = None
        self.sigmaE = 0.0        
        self.var_guess = 0.0
        self.pmode = np.zeros((K+1))

    def runAnalysis(self, responses, nStep, init_sigma = 0.005):
        self.__init__(len(responses), 0.2)
        self.r = responses
        self.sigmaE = init_sigma
        self.var_guess = init_sigma**2
        self.v[0] = self.var_guess
        for i in xrange(nStep):
            print 'EM iteration => '+str(i)
            self.forwardFilter()
            self.backwardSmoothing()
            self.varianceEstimation()
            self.x_first.append(self.x_new[0])
            if self.convergenceEstimation(i):
                break
            self.sigmaE = np.sqrt(self.var[-1])
            self.x[0] = self.x_new[0]
            self.v[0] = self.v_new[0]
        self.integratePDF()
        self.pmode = self.pmode[1:]

    def forwardFilter(self):
        for k in xrange(1, self.K+1):
            self.x_old[k] = self.x[k-1]
            self.v_old[k] = self.v[k-1] + self.sigmaE**2
            self.newtonSolve(k)
            denom = -1/self.v_old[k] - np.exp(self.mu)*np.exp(self.x[k])/(1+np.exp(self.mu)*np.exp(self.x[k]))**2
            self.v[k] = -1/denom
        self.p = np.exp(self.mu)*np.exp(self.x)/(1+np.exp(self.mu)*np.exp(self.x))
            
    def backwardSmoothing(self):
        self.x_new[-1] = self.x[-1]
        self.v_new[-1] = self.v[-1]
        for k in xrange(self.K-1, 0, -1):
            self.A[k] = self.v[k]/self.v_old[k+1]
            self.x_new[k] = self.x[k] + self.A[k]*(self.x_new[k+1]-self.x_old[k+1])
            self.v_new[k] = self.v[k] + self.A[k]*self.A[k]*(self.v_new[k+1]-self.v_old[k+1])
        self.x_new[0] = self.x_new[1]
        self.v_new[0] = self.v_new[1]
        
    def varianceEstimation(self):
        term1 = np.sum(self.x_new[2:]**2) + np.sum(self.v_new[2:])
        term2 = np.sum(self.v_new[2:]*self.A[1:]) + np.sum(self.x_new[2:]*self.x_new[1:-1])
        term3 = self.x_new[1]*self.x_new[1] + 2*self.v_new[1]
        term4 = self.x_new[-1]**2 + self.v_new[-1]        
        newvar = (2*(term1-term2)+term3-term4)/(self.K)
        self.var.append(newvar)
        
    def newtonSolve(self, k):
        it = [self.x_old[k] + self.v_old[k]*(float(self.r[k-1]) - float(np.max(self.r))*np.exp(self.mu)*np.exp(self.x_old[k])/(1+np.exp(self.mu)*np.exp(self.x_old[k])))]
        g = []
        gprime = []              
        for i in xrange(40):
            g.append(self.x_old[k] + self.v_old[k]*(float(self.r[k-1]) - float(np.max(self.r))*np.exp(self.mu)*np.exp(it[-1])/(1+np.exp(self.mu)*np.exp(it[-1]))) - it[-1])
            gprime.append((-float(np.max(self.r))*self.v_old[k]*np.exp(self.mu)*np.exp(it[-1]))/(1+np.exp(self.mu)*np.exp(it[-1]))**2 - 1)
            it.append(it[-1]-g[-1]/gprime[-1])
            self.x[k] = it[-1]
            if np.abs(self.x[k]-it[-1])<1e-14:
                break

    def convergenceEstimation(self, i):
        if i > 0:
            a1 = np.abs(self.var[-1]-self.var[-2])
            a2 = np.abs(self.x_first[-1]-self.x_first[-2])
            if a1 < self.criterion and a2 < self.criterion:
                return True
        else:
            return False
        
    def integratePDF(self):
        dels = 1e-4
        pr = np.arange(dels, 1-dels, dels)
        for k in xrange(self.K+1):
            x = self.x_new[k]
            v = self.v_new[k]
            pr = np.arange(dels, 1-dels, dels)
            term1 = 1./(np.sqrt(2*np.pi*v)*(pr*(1-pr)))
            term2 = np.exp(-1/(2**v) * (np.log (pr/((1-pr)*np.exp(self.mu))) - x)**2)
            pdf = term1 * term2
            pdf = dels*pdf                        
            sumpdf = sp.cumtrapz(pdf)
            lowlimit = np.where(sumpdf > 0.05)[0]
            if len(lowlimit) <> 0:
                lowlimit = lowlimit[0]
            else:
                lowlimit = 1;
            highlimit = np.where(sumpdf > 0.95)
            if len(highlimit) <> 0:
                if len(highlimit) > 1:
                    highlimit = highlimit[0]-1
                else:
                    highlimit = highlimit[0]
            else:
                highlimit = len(pr)
            ######
            #to finish
            ######
            self.pmode[k] = pr[np.argmax(pdf)]
                
            
                
                
