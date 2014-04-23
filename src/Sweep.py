#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from itertools import izip
from ColorAssociationTasks import CATS

def unwrap_self_multiOptimize(arg, **kwarg):
    return Likelihood.multiOptimize(*arg, **kwarg)

def unwrap_self_multiSampling(arg, **kwarg):
    return SamplingPareto.multiSampling(*arg, **kwarg)

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

    def computeFullLikelihood(self):
        """ Performs a likelihood on performances
            and a linear regression on reaction time
        """
        llh = 0.0
        rt = []
        # for i, v in zip(self.p_order, p):
        #     self.model.setParameter(i, v)
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
        subject = ['S2', 'S8']
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


class SamplingPareto():
    """ Simple sampling of parameters to
    draw pareto front """

    def __init__(self, human, model, n = 10000):
        self.human = human
        self.model = model
        self.subject = self.human.keys()
        self.n = n
        self.nb_repeat = 8
        self.nb_blocs = 4
        self.nb_trials = 39
        self.nb_param = len(self.model.bounds.keys())
        self.p_order = self.model.bounds.keys()
        self.cats = CATS(self.nb_trials)
        self.rt = dict()
        self.state = dict()
        self.action = dict()
        self.responses = dict()
        self.indice = dict()
        self.hrt = dict()        
        for s in self.human.keys():
            self.rt[s] = np.array([self.human[s][i]['rt'][0:self.nb_trials,0] for i in range(1,self.nb_blocs+1)])
            self.rt[s] = np.tile(self.rt[s], (self.nb_repeat,1))
            self.state[s] = np.array([self.human[s][i]['sar'][0:self.nb_trials,0] for i in range(1,self.nb_blocs+1)])
            self.state[s] = np.tile(self.state[s], (self.nb_repeat,1))
            self.action[s] = np.array([self.human[s][i]['sar'][0:self.nb_trials,1] for i in range(1,self.nb_blocs+1)])
            self.action[s] = np.tile(self.action[s], (self.nb_repeat,1))
            self.responses[s] = np.array([self.human[s][i]['sar'][0:self.nb_trials,2] for i in range(1,self.nb_blocs+1)])
            self.responses[s] = np.tile(self.responses[s], (self.nb_repeat,1))
            step, indice = getRepresentativeSteps(self.rt[s], self.state[s], self.action[s], self.responses[s])
            self.hrt[s] = computeMeanRepresentativeSteps(step)[0]
            self.hrt[s] = self.center(self.hrt[s])

    def _convertStimulus(self, s):
            return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

    def center(self, x):
        x = x - np.median(x)
        x = x / float(np.percentile(x, 75)-np.percentile(x, 25))
        return x

    def evaluate(self, s):
        p_test = {k:np.random.uniform(self.model.bounds[k][0],self.model.bounds[k][1]) for k in self.model.bounds.keys()}
        self.model.setAllParameters(p_test)
        self.model.startExp()
        for i in xrange(self.nb_repeat):
            for j in xrange(self.nb_blocs):
                self.cats.reinitialize()
                self.cats.stimuli = np.array(map(self._convertStimulus, self.human[s][j+1]['sar'][:,0]))
                self.model.startBloc()
                for k in xrange(self.nb_trials):
                    state = self.cats.getStimulus(k)
                    action = self.model.chooseAction(state)
                    reward = self.cats.getOutcome(state, action)
                    self.model.updateValue(reward)
        self.model.reaction = np.array(self.model.reaction)
        self.model.action = np.array(self.model.action)
        self.model.responses = np.array(self.model.responses)
        self.model.value = np.array(self.model.value)
        step, indice = getRepresentativeSteps(self.model.reaction, self.state[s], self.model.action, self.model.responses)
        hrtm = computeMeanRepresentativeSteps(step)[0]
        hrtm = self.center(hrtm)

        rt = -np.sum(np.power(hrtm-self.hrt[s], 2))

        choice = np.sum(np.log(self.model.value))
        return np.array([choice, rt])

    def multiSampling(self, s):
        n = 100
        data = np.zeros((n, 3))
        pareto = np.array([[-1, -1000., -1000.0]])
        p = np.zeros((n,self.nb_param+1))        
        good = np.zeros((1, self.nb_param+1))
        good[0,0] = -1.0
        for i in xrange(1,self.n+1):
            ind = (i-1)%n            
            data[ind,0] = i
            p[ind,0] = i
            data[ind,1:] = self.evaluate(s)
            p[ind,1:] = np.array([self.model.parameters[k] for k in self.p_order])
            if i%n == 0:
                pareto, good = self.constructParetoFrontier(np.vstack((data, pareto)), np.vstack((p, good)))
                
        return dict({s:np.hstack((pareto, good[:,1:]))})

    def constructParetoFrontier(self, front, param):
        front = front[front[:,1].argsort()][::-1]
        pareto_frontier = [front[0]]
        for pair in front[1:]:
            if pair[2] >= pareto_frontier[-1][2]:
                pareto_frontier.append(pair)
        pareto_frontier = np.array(pareto_frontier)        
        good = np.array([param[param[:,0] == i][0] for i in pareto_frontier[:,0]])
        return pareto_frontier, good

    def run(self):
        subject = self.subject
        pool = Pool(len(subject))
        self.data = pool.map(unwrap_self_multiSampling, zip([self]*len(subject), subject))
        tmp = dict()
        for i in self.data:
            s = i.keys()[0]
            tmp[s] = i[s]            
        self.data = tmp

    def save(self, output_file):
        output = open(output_file, 'wb')
        pickle.dump(self.data, output)
        output.close()