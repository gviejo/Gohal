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
#from scipy.stats import chi2_contingency
from scipy.stats import norm
import scipy.optimize
from multiprocessing import Pool, Process

def unwrap_self_multiOptimize(arg, **kwarg):
    return Likelihood.multiOptimize(*arg, **kwarg)

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

        # elif case == "C":
        #     if h1 == m1:
        #         return 1
        #     else:
        #         chi, p, ddl, the = chi2_contingency(obs, correction=False)
        #         return (chi/(chi+(h+m)))**(0.5)
        #         #return np.sqrt(chi)
        
        # elif case == "phi":
        #     if h1 == m1:
        #         return 1
        #     else:
        #         chi, p, ddl, the = chi2_contingency(obs, correction=False)
        #         return np.sqrt(chi)

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
    Optimization with the function you have to choose carefully my friend
    See : Trial-by-trial data analysis using computational models, Daw, 2009
    """
    def __init__(self, human, ptr_model, fname, n_run, n_grid, maxiter, maxfun, xtol, ftol, disp):
        self.X = human.subject['fmri']
        self.model = ptr_model
        self.fname = fname
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.xtol = xtol
        self.ftol = ftol
        self.disp = disp
        self.subject = self.X.keys()
        self.n_run = n_run
        self.n_grid = n_grid
        self.best_parameters = None
        self.start_parameters = None        
        self.brute_grid = None
        self.p = self.model.getAllParameters()
        self.p_order = self.p.keys()        
        self.current_subject = None
        self.cvt = dict({i:'s'+str(i) for i in [1,2,3]})
        self.lower = np.array([self.p[i][0] for i in self.p_order])
        self.upper = np.array([self.p[i][2] for i in self.p_order])
        self.ranges = tuple(map(tuple, np.array([self.lower, self.upper]).transpose()))        
        self.searchStimOrder()
        self.data = None

    def searchStimOrder(self):
        """ Done for all bloc 
            for all subject
        """
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
                llh = llh + np.log(values[true_action])
                self.model.current_action = true_action
                self.model.updateValue(trial[2])                                                        
        return -llh

    def computeFullLikelihood(self, p):
        """ Performs a likelihood on performances
            and a linear regression on reaction time
        """
        llh = 0.0
        rt = []
        for i, v in zip(self.p_order, p):
            self.model.setParameter(i, v)
        self.model.initializeList()
        for bloc in self.X[self.current_subject].iterkeys():
            self.model.initialize()
            tmp = self.X[self.current_subject][bloc]
            for trial, hrt in zip(tmp['sar'],tmp['rt']):            
                state = self.cvt[trial[0]]                
                true_action = trial[1]-1
                values = self.model.computeValue(state)
                llh = llh + np.log(values[true_action])
                self.model.current_action = true_action
                self.model.updateValue(trial[2])                                                        
                rt.append([hrt[0],self.model.reaction[-1][-1]])
        rt = np.array(rt)
        rt = rt-np.min(rt,0)
        rt = rt/np.max(rt,0)
        lr = np.sum((rt[:,0]-rt[:,1])**2)
        return -llh, lr


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

    def optimize(self, subject): 
        self.best_parameters = list()
        self.start_parameters = list()
        max_likelihood = list()
        self.brute_grid = list()
        self.current_subject = subject
        if self.fname == 'minimize':
            for i in xrange(self.n_run):
                p_start = self.generateStart()                
                tmp = scipy.optimize.minimize(fun=self.computeLikelihood,
                                              x0=p_start,
                                              method='TNC',
                                              jac=None,
                                              hess=None,
                                              hessp=None,
                                              bounds=self.ranges,
                                              options={'maxiter':self.maxiter,
                                                       'disp':self.disp})
                self.best_parameters.append(tmp.x)
                max_likelihood.append(-tmp.fun)
                self.start_parameters.append(p_start)
            self.best_parameters = np.array(self.best_parameters)
            self.start_parameters = np.array(self.start_parameters)
            max_likelihood = np.array(max_likelihood)
            return dict({self.current_subject:dict({'start':self.start_parameters,
                                                    'best':self.best_parameters, 
                                                    'max':max_likelihood})})
        elif self.fname == 'fmin':
            warnflag = []
            for i in xrange(self.n_run):
                p_start = self.generateStart()
                tmp = scipy.optimize.fmin(func=self.computeLikelihood,
                                            x0=p_start,
                                            full_output = True,
                                            #maxiter=self.maxiter,
                                            #maxfun=self.maxfun,
                                            #xtol=self.xtol,
                                            #ftol=self.ftol,
                                            disp=self.disp)                
                self.best_parameters.append(tmp[0])
                max_likelihood.append(-tmp[1])
                self.start_parameters.append(p_start)
                warnflag.append(tmp[4])
            self.best_parameters = np.array(self.best_parameters)
            self.start_parameters = np.array(self.start_parameters)
            max_likelihood = np.array(max_likelihood)
            warnflag = np.array(warnflag)
            self.data =  [dict({self.current_subject:dict({'start':self.start_parameters,
                                                            'best':self.best_parameters,
                                                            'warnflag':warnflag,
                                                            'max':max_likelihood})})]
        elif self.fname == 'anneal':
            for i in xrange(self.n_run):
                p_start = self.generateStart()
                tmp = scipy.optimize.anneal(func=self.computeLikelihood,
                                            x0=p_start,
                                            schedule='fast',
                                            lower=self.lower,
                                            upper=self.upper,
                                            disp=self.disp)
                self.best_parameters.append(tmp.x)
                max_likelihood.append(-tmp.fun)
                self.start_parameters.append(p_start)
            self.best_parameters = np.array(self.best_parameters)
            self.start_parameters = np.array(self.start_parameters)
            max_likelihood = np.array(max_likelihood)
            return dict({self.current_subject:dict({'start':self.start_parameters,
                                                    'best':self.best_parameters, 
                                                    'max':max_likelihood})})
        elif self.fname == 'brute':            
            rranges = tuple([slice(i[0],i[1],(i[1]-i[0])/self.n_grid) for i in self.ranges])
            tmp = scipy.optimize.brute(func=self.computeLikelihood,
                                       ranges=rranges,
                                       disp=self.disp,
                                       Ns=self.n_grid,
                                       full_output=True,                                       
                                       finish=scipy.optimize.fmin)
            self.best_parameters = tmp[0]
            max_likelihood = tmp[1]
            self.brute_grid = tuple((tmp[2], tmp[3]))
            self.data = [dict({self.current_subject:dict({'best':tmp[0],
                                                         'grid':tmp[2],
                                                        'grid_f':tmp[3],
                                                        'max':tmp[1]})})]
        elif self.fname == 'fmin_tnc':
            for i in xrange(self.n_run):
                p_start = self.generateStart()                
                tmp = scipy.optimize.fmin_tnc(func=self.computeLikelihood,
                                              x0=p_start,
                                              fprime=None,
                                              approx_grad=0,
                                              bounds=list(self.ranges),
                                              disp=4)

                                              
                self.best_parameters.append(tmp.x)
                self.start_parameters.append(p_start)
                max_likelihood.append(tmp.rc)
            self.best_parameters = np.array(self.best_parameters)
            self.start_parameters = np.array(self.state)
            max_likelihood = np.array(max_likelihood)
            return dict({self.current_subject:dict({'start':self.start_parameters,
                                                    'best':self.best_parameters, 
                                                    'max':max_likelihood})})
        else:
            print "Function not found"
            sys.exit()

    def generateStart(self):
        return [np.random.uniform(self.p[i][0], self.p[i][2]) for i in self.p_order]

    def set(self, ptr_m, subject):
        self.model = ptr_m
        self.current_subject = subject
        self.p = self.model.getAllParameters()
        self.p_order = self.p.keys()
        self.lower = np.array([self.p[i][0] for i in self.p_order])
        self.upper = np.array([self.p[i][2] for i in self.p_order])
        self.ranges = tuple(map(tuple, np.array([self.lower, self.upper]).transpose()))

    def multiOptimize(self, subject):
        self.current_subject = subject
        self.best_parameters = list()
        self.start_parameters = list()              
        max_likelihood = list()        
        self.brute_grid = list()

        if self.fname == 'minimize':
            for i in xrange(self.n_run):
                p_start = self.generateStart()                
                tmp = scipy.optimize.minimize(fun=self.computeLikelihood,
                                              x0=p_start,
                                              method='TNC',
                                              jac=None,
                                              hess=None,
                                              hessp=None,
                                              bounds=self.ranges,
                                              options={'maxiter':self.maxiter,
                                                       'disp':self.disp})
                self.best_parameters.append(tmp.x)
                max_likelihood.append(-tmp.fun)
                self.start_parameters.append(p_start)
            self.best_parameters = np.array(self.best_parameters)
            self.start_parameters = np.array(self.start_parameters)
            max_likelihood = np.array(max_likelihood)
            return dict({self.current_subject:dict({'start':self.start_parameters,
                                                    'best':self.best_parameters, 
                                                    'max':max_likelihood})})
        elif self.fname == 'fmin':
            warnflag = []
            for i in xrange(self.n_run):
                p_start = self.generateStart()
                tmp = scipy.optimize.fmin(func=self.computeLikelihood,
                                            x0=p_start,
                                            #maxiter=self.maxiter,
                                            #maxfun=self.maxfun,
                                            #xtol=self.xtol,
                                            #ftol=self.ftol,
                                            full_output=True,
                                            disp=self.disp)
                self.best_parameters.append(tmp[0])
                max_likelihood.append(-tmp[1])
                self.start_parameters.append(p_start)
                warnflag.append(tmp[4])
            self.best_parameters = np.array(self.best_parameters)
            self.start_parameters = np.array(self.start_parameters)
            max_likelihood = np.array(max_likelihood)
            warnflag = np.array(warnflag)
            return dict({self.current_subject:dict({'start':self.start_parameters,
                                                    'best':self.best_parameters, 
                                                    'warnflag':warnflag,
                                                    'max':max_likelihood})})
        elif self.fname == 'anneal':                    
            for i in xrange(self.n_run):
                p_start = self.generateStart()
                tmp = scipy.optimize.anneal(func=self.computeLikelihood,
                                            x0=p_start,
                                            schedule='fast',
                                            lower=self.lower,
                                            upper=self.upper,
                                            disp=self.disp)
                self.best_parameters.append(tmp.x)
                max_likelihood.append(-tmp.fun)
                self.start_parameters.append(p_start)
            self.best_parameters = np.array(self.best_parameters)
            self.start_parameters = np.array(self.start_parameters)
            max_likelihood = np.array(max_likelihood)
            return dict({self.current_subject:dict({'start':self.start_parameters,
                                                    'best':self.best_parameters, 
                                                    'max':max_likelihood})})

        elif self.fname == 'brute':
            rranges = tuple([slice(i[0],i[1],(i[1]-i[0])/self.n_grid) for i in self.ranges])
            tmp = scipy.optimize.brute(func=self.computeLikelihood,
                                       ranges=rranges,
                                       disp=self.disp,
                                       Ns=self.n_grid,
                                       full_output=True, 
                                       finish=scipy.optimize.fmin)
            self.best_parameters = tmp[0]
            max_likelihood = tmp[1]
            self.brute_grid = tuple((tmp[2], tmp[3]))
            return dict({self.current_subject:dict({'best':tmp[0],
                                                    'grid':tmp[2],
                                                    'grid_f':tmp[3],
                                                    'max':tmp[1]})})

        else:
            print "Function not found"
            sys.exit()

    def run(self):        
        #subject = ['S1', 'S9', 'S8', 'S3']        
        subject = ['S2', 'S8', 'S9', 'S11']
        #subject = self.subject
        pool = Pool(len(subject))
        self.data = pool.map(unwrap_self_multiOptimize, zip([self]*len(subject), subject))                

    def save(self, output_file):
        opt = []
        start = []
        fun = []
        subject = []
        grid = []
        grid_fun = []
        warn = []
        for i in self.subject:
            for j in self.data:
                if j.keys()[0] == i and self.fname != 'brute':
                    opt.append(j[i]['best'])
                    start.append(j[i]['start'])
                    fun.append(j[i]['max'])
                    subject.append(i)
                    warn.append(j[i]['warnflag'])
                elif j.keys()[0] == i and self.fname == 'brute':
                    opt.append(j[i]['best'])
                    grid.append(j[i]['grid'])
                    grid_fun.append(j[i]['grid_f'])
                    fun.append(j[i]['max'])
                    subject.append(i)
        opt = np.array(opt)
        start = np.array(start)
        fun = np.array(fun)
        grid = np.array(grid)
        grid_fun = np.array(grid_fun)
        data = dict({'start':start,
                     'opt':opt,
                     'max':fun,
                     'grid':grid,
                     'grid_fun':grid_fun,
                     'p_order':self.p_order,
                     'subject':subject,
                     'parameters':self.p,
                     'search':self.n_run,
                     'fname':self.fname,
                     'warnflag':warn})
        output = open(output_file, 'wb')
        pickle.dump(data, output)
        output.close()

        
