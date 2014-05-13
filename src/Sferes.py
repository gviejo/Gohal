#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sferes.py

    class for multi-objective optimization
    to interface with sferes2 : see
    http://sferes2.isir.upmc.fr/
    fitness function is made of Bayesian Information Criterion
    and either Linear Regression
    or possible Reaction Time Likelihood

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import mmap
import numpy as np
if os.uname()[1] in ['atlantis', 'paradise']:
    from multiprocessing import Pool, Process
    from pylab import *

from fonctions import *
from Selection import *
from Models import *
from HumanLearning import HLearning
from ColorAssociationTasks import CATS
from scipy.stats import sem
from scipy.stats import norm
from scipy.optimize import leastsq

def unwrap_self_load_data(arg, **kwarg):
    return pareto.loadPooled(*arg, **kwarg)

class EA():
    """
    Optimization is made for one subject
    """
    def __init__(self, data, subject, ptr_model):
        self.model = ptr_model
        self.subject = subject
        self.data = data
        self.n_trials = 39
        self.n_blocs = 4
        self.n_repets = 5
        self.fit = np.zeros(2)
        self.rt = np.array([self.data[i]['rt'][0:self.n_trials,0] for i in [1,2,3,4]]).flatten()        
        self.state = np.array([self.data[i]['sar'][0:self.n_trials,0] for i in [1,2,3,4]])
        self.action = np.array([self.data[i]['sar'][0:self.n_trials,1] for i in [1,2,3,4]]).astype(int)
        self.responses = np.array([self.data[i]['sar'][0:self.n_trials,2] for i in [1,2,3,4]])
        # self.fitfunc = lambda p, x: p[0] + p[1] * x
        # self.errfunc = lambda p, x, y : (y - self.fitfunc(p, x))
        # self.bin_size = 2*(np.percentile(self.rt, 75)-np.percentile(self.rt, 25))*np.power(len(self.rt), -(1/3.))        
        # self.mass, self.edges = np.histogram(self.rt, bins=np.arange(self.rt.min(), self.rt.max()+self.bin_size, self.bin_size))        
        # self.mass = self.mass/float(self.mass.sum())
        # self.position = np.digitize(self.rt, self.edges)-1
        # self.f = lambda i, x1, x2, y1, y2: (i*(y2-y1)-y2*x1+y1*x2)/(x2-x1)
        # self.p = None
        # self.p_rtm = None

    def getFitness(self):
        np.seterr(all = 'ignore')
        self.model.startExp()
        for e in xrange(self.n_repets):
            for i in xrange(self.n_blocs):
                self.model.startBloc()
                for j in xrange(self.n_trials):                
                    self.model.computeValue(int(self.state[i,j])-1, int(self.action[i,j])-1)                
                    self.model.updateValue(self.responses[i,j])
        
        self.model.value = np.array(self.model.value)        
        self.rtm = np.array(self.model.reaction).flatten()

        self.fit[0] = float(np.sum(np.log(self.model.value)))
        #tmp = self.computeMutualInformation()        
        self.alignToMedian()
        self.fit[1] = float(-self.leastSquares())        
        self.fit = np.round(self.fit, 4)
        self.fit[np.isnan(self.fit)] = -10000.0
        self.fit[np.isinf(self.fit)] = -10000.0                
        choice = str(self.fit[0]+2000.0)
        rt = str(self.fit[1]+500.0)
        # FUCKING UGLY ########
        if choice == '0.0' or choice == '0': choice = '-10000.0'
        if rt == '0.0' or rt == '0': rt = '-10000.0'
        #######################
        
        return choice, rt

    def leastSquares(self):
        self.rt = np.tile(self.rt, self.n_repets)        
        self.state = np.tile(self.state, (self.n_repets, 1))
        self.action = np.tile(self.action, (self.n_repets, 1))
        self.responses = np.tile(self.responses, (self.n_repets, 1))
        pinit = [1.0, -1.0]
        self.mean = []
        for i in [self.rt, self.rtm]:
            tmp = i.reshape(self.n_blocs*self.n_repets, self.n_trials)            
            step, indice = getRepresentativeSteps(tmp, self.state, self.action, self.responses)
            self.mean.append(computeMeanRepresentativeSteps(step)[0])
        self.mean = np.array(self.mean)
        #p = leastsq(self.errfunc, pinit, args = (mean[1][0], mean[0][0]), full_output = False)                
        return np.sum(np.power(self.mean[0]-self.mean[1], 2))            
        #return np.sum(np.power(self.errfunc(p[0], mean[1][0], mean[0][0]), 2))

    def computeMutualInformation(self):
        self.p_rtm, edges = np.histogram(self.rtm, bins = np.linspace(self.rtm.min(), self.rtm.max()+0.00001, 25))        
        self.p_rtm = self.p_rtm/float(self.p_rtm.sum())
        self.p = np.zeros((len(self.mass), len(self.p_rtm)))
        positionm = np.digitize(self.rtm, edges)-1        
        self.position = np.tile(self.position, self.n_repets)
 
        for i in xrange(len(self.position)): self.p[self.position[i], positionm[i]] += 1        
        
        self.p = self.p/float(self.p.sum())
        
        tmp = np.log2(self.p/np.outer(self.mass, self.p_rtm))        
        tmp[np.isinf(tmp)] = 0.0
        tmp[np.isnan(tmp)] = 0.0
        return np.sum(self.p*tmp)        

    def computeDistance(self):
        sup = self.edges[self.position]
        #self.sup = sup
        #size_bin = self.edges[1]-self.edges[0]        
        sup = np.tile(np.vstack(sup), int(self.model.parameters['length'])+1)
        self.d = norm.cdf(sup, self.rt_model, self.model.sigma)-norm.cdf(sup-self.bin_size, self.rt_model, self.model.sigma)
        self.d[np.isnan(self.d)] = 0.0

    def alignToMedian(self):
        self.rt = self.rt - np.median(self.rt)
        self.rtm = self.rtm - np.median(self.rtm)
        self.rtm = self.rtm / (np.percentile(self.rtm, 75)-np.percentile(self.rtm, 25))
        self.rt = self.rt / (np.percentile(self.rt, 75)-np.percentile(self.rt, 25))


class RBM():
    """
    Restricted Boltzman machine
    x : Human reaction time
    y : Model Inference
    """
    def __init__(self, x, y, nh = 10, nbiter = 1000):
        # Parameters
        self.nh = nh
        self.nbiter = nbiter
        self.nx = x.shape[1]
        self.ny = y.shape[1]
        self.nd = x.shape[0]
        self.nv = self.nx+self.ny
        self.sig = 0.2        
        self.epsW = 0.5
        self.epsA = 0.5
        self.cost = 0.00001
        self.momentum = 0.95
        # data        
        self.x = np.hstack((x, y))
        self.xx = np.zeros(self.x.shape)  # TEST DATASET
        # Weights
        self.W = np.random.normal(0, 0.1,size=(self.nh+1,self.nv+1))        
        self.dW = np.random.normal(0, 0.001, size = (self.nh+1,self.nv+1))
        # Units
        self.Svis = np.zeros((self.nv+1))                
        self.Svis[-1] = 1.0
        self.Shid = np.zeros((self.nh+1))        
        # Gradient
        self.Wpos = np.zeros((self.nh+1,self.nv+1))
        self.Wneg = np.zeros((self.nh+1,self.nv+1))
        self.apos = np.zeros((self.nd, self.nh+1))
        self.aneg = np.zeros((self.nd, self.nh+1))        
        # Biais
        self.Ahid = np.ones(self.nh+1)
        self.Avis = 0.1*np.ones(self.nv+1)
        self.dA = np.zeros(self.nv+1)
    
        self.Error = np.zeros(self.nbiter)

    def sigmoid(self, x, a):
        return 1.0/(1.0+np.exp(-a*x))

    # visible=0, hidden=1
    def activ(self, who):
        if(who==0):
            self.Svis = np.dot(self.Shid, self.W) + self.sig*np.random.standard_normal(self.nv+1)         
            self.Svis = self.sigmoid(self.Svis, self.Avis)
            self.Svis[-1] = 1.0 # bias
        if(who==1):
            self.Shid = np.dot(self.Svis, self.W.T) + self.sig*np.random.standard_normal(self.nh+1)
            self.Shid = self.sigmoid(self.Shid, self.Ahid)
            #self.Shid = (self.Shid>np.random.rand(self.Shid.shape[0]))*1.0
            self.Shid[-1] = 1.0 # bias        

    def train(self):        
        for i in xrange(self.nbiter):            
            self.Wpos = np.zeros((self.nh+1,self.nv+1))
            self.Wneg = np.zeros((self.nh+1,self.nv+1))
            self.apos = np.zeros((self.nh+1))
            self.aneg = np.zeros((self.nh+1))
            error = 0.0
            for point in xrange(self.nd):
                # Positive Phase
                self.Svis[0:self.nv] = self.x[point]
            
                self.activ(1)            
                self.Wpos = self.Wpos + np.outer(self.Shid, self.Svis)            
                self.apos = self.apos + self.Shid*self.Shid
                # Negative Phase                
                self.activ(0)
                self.activ(1)

                error += np.sum(np.power(self.Svis[0:self.nv]-self.x[point], 2))
                
                # Update phase                
                self.Wneg = self.Wneg + np.outer(self.Shid, self.Svis)
                self.aneg = self.aneg + self.Shid*self.Shid        

            self.Error[i] = error
            self.dW = self.dW*self.momentum + self.epsW * ((self.Wpos - self.Wneg)/float(self.nd) - self.cost*self.W)
            self.W = self.W + self.dW
            self.Ahid = self.Ahid + self.epsA*(self.apos - self.aneg)/(float(self.nd)*self.Ahid*self.Ahid)

            print "Epoch "+str(i)+" Error = "+str(error)

    def getInputfromOutput(self,xx, n = 15):
        self.xx = xx
        self.out = np.zeros((self.xx.shape[0], self.nx))
        for point in xrange(self.xx.shape[0]):
            self.Svis[0:self.nv] = 0.0
            self.Svis[self.nx:self.nx+self.ny] = self.xx[point]
            for i in xrange(n):
                self.activ(1)
                self.activ(0)
                self.Svis[self.nx:self.nx+self.ny] = self.xx[point]
            self.out[point] = self.Svis[0:self.nx]
        return self.out

    def reconstruct(self, xx, n = 10):
        self.xx = xx
        self.out = np.zeros(self.xx.shape)
        for point in xrange(self.xx.shape[0]):
            self.Svis[0:self.nv] = self.xx[point]
            for i in xrange(n):
                self.activ(1)
                self.activ(0)
            self.out[point] = self.Svis[0:self.nv]
        return self.out


class pareto():
    """
    Explore Pareto Front from Sferes Optimization
    """
    def __init__(self, directory, threshold, N = 156.):
        self.directory = directory
        self.threshold = threshold
        self.N = N
        self.human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
        self.data = dict()
        self.states = ['s1', 's2', 's3']
        self.actions = ['thumb', 'fore', 'midd', 'ring', 'little']
        self.models = dict({"fusion":FSelection(self.states, self.actions),
                            "qlearning":QLearning(self.states, self.actions),
                            "bayesian":BayesianWorkingMemory(self.states, self.actions),
                            "selection":KSelection(self.states, self.actions),
                            "mixture":CSelection(self.states, self.actions)})
        
        self.p_order = dict({'fusion':['alpha','beta', 'gamma', 'noise','length','gain','threshold', 'sigma'],
                            'qlearning':['alpha','beta','gamma'],
                            'bayesian':['length','noise','threshold', 'sigma'],
                            'selection':['gamma','beta','eta','length','threshold','noise','sigma', 'sigma_rt'],
                            'mixture':['alpha', 'beta', 'gamma', 'noise', 'length', 'weight', 'threshold', 'sigma']})

        self.m_order = ['qlearning', 'bayesian', 'selection', 'fusion', 'mixture']
        self.colors_m = dict({'fusion':'r', 'bayesian':'g', 'qlearning':'grey', 'selection':'b', 'mixture':'y'})
        self.opt = dict()
        self.pareto = dict()
        self.rank = dict()
        self.p_test = dict()
        self.mixed = dict()
        self.beh = dict({'state':[],'action':[],'responses':[],'reaction':[]})
        self.indd = dict()
        #self.loadData()
        self.simpleLoadData()
        self.constructParetoFrontier()        
        self.constructMixedParetoFrontier()


    def loadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)

        pool = Pool(len(model_in_folders))
        tmp = pool.map(unwrap_self_load_data, zip([self]*len(model_in_folders), model_in_folders))
        
        for d in tmp:
            self.data[d.keys()[0]] = d[d.keys()[0]]

    def simpleLoadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)
        for m in model_in_folders:
            self.data[m] = dict()
            lrun = os.listdir(self.directory+"/"+m)
            order = self.p_order[m.split("_")[0]]
            scale = self.models[m.split("_")[0]].bounds
            for r in lrun:
                s = r.split("_")[3]                
                n = int(r.split("_")[4].split(".")[0])
                if s in self.data[m].keys():
                    self.data[m][s][n] = np.genfromtxt(self.directory+"/"+m+"/"+r)
                else :
                    self.data[m][s] = dict()
                    self.data[m][s][n] = np.genfromtxt(self.directory+"/"+m+"/"+r)                                
                for p in order:
                    self.data[m][s][n][:,order.index(p)+4] = scale[p][0]+self.data[m][s][n][:,order.index(p)+4]*(scale[p][1]-scale[p][0])

    def loadPooled(self, m):         
        data = {m:{}}
        list_file = os.listdir(self.directory+"/"+m)
        order = self.p_order[m]
        scale = self.models[m].bounds
        for r in list_file:
            s = r.split("_")[3]
            n = int(r.split("_")[4].split(".")[0])
            data[m][s] = dict()
            filename = self.directory+"/"+m+"/"+r            
            nb_ind = int(self.tail(filename, 1)[0].split(" ")[1])
            last_gen = np.array(map(lambda x: x[0:-1].split(" "), self.tail(filename, nb_ind+1))).astype('float')
            if s in data[m].keys():
                data[m][s][n] = last_gen
            else:
                data[m][s] = {n:last_gen}
            for p in order:
                data[m][s][n][:,order.index(p)+4] = scale[p][0]+data[m][s][n][:,order.index(p)+4]*(scale[p][1]-scale[p][0])                    
        return data

    def tail(self, filename, n):
        size = os.path.getsize(filename)
        with open(filename, "rb") as f:
            fm = mmap.mmap(f.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
            for i in xrange(size-1, -1, -1):
                if fm[i] == '\n':
                    n -= 1
                    if n == -1:
                        break
            return fm[i+1 if i else 0:].splitlines()

    def constructParetoFrontier(self):
        for m in self.data.iterkeys():
            self.pareto[m] = dict()
            for s in self.data[m].iterkeys():
                print m, s        
                self.pareto[m][s] = dict()   
                tmp={n:self.data[m][s][n][self.data[m][s][n][:,0]==np.max(self.data[m][s][n][:,0])] for n in self.data[m][s].iterkeys()}
                tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])
                ind = tmp[:,3] != 0
                tmp = tmp[ind]
                tmp = tmp[tmp[:,3].argsort()][::-1]
                pareto_frontier = [tmp[0]]
                for pair in tmp[1:]:
                    if pair[4] >= pareto_frontier[-1][4]:
                        pareto_frontier.append(pair)
                self.pareto[m][s] = np.array(pareto_frontier)
                
                self.pareto[m][s][:,3] = self.pareto[m][s][:,3] - 2000.0
                self.pareto[m][s][:,4] = self.pareto[m][s][:,4] - 500.0
                #self.pareto[m][s][:,3] = self.pareto[m][s][:,3] - np.log(self.N)*float(len(self.models[m].bounds.keys()))
                #self.pareto[m][s][:,4] = self.pareto[m][s][:,4] - np.log(self.N)*float(len(self.models[m].bounds.keys()))
                for t in xrange(len(self.threshold)):
                    self.pareto[m][s] = self.pareto[m][s][self.pareto[m][s][:,3+t] >= self.threshold[t]]                            

    def constructMixedParetoFrontier(self):
        subjects = set.intersection(*map(set, [self.pareto[m].keys() for m in self.pareto.keys()]))
        for s in subjects:
            self.mixed[s] = []
            tmp = []
            for m in self.pareto.iterkeys():
                if s in self.pareto[m].keys():
                    tmp.append(np.hstack((np.ones((len(self.pareto[m][s]),1))*self.m_order.index(m), self.pareto[m][s][:,0:5])))
            tmp = np.vstack(tmp)            
            tmp = tmp[tmp[:,4].argsort()][::-1]
            self.mixed[s] = [tmp[0]]
            for pair in tmp[1:]:
                if pair[5] >= self.mixed[s][-1][5]:
                    self.mixed[s].append(pair)
            self.mixed[s] = np.array(self.mixed[s])

    def rankMixedFront(self, w):    
        for s in self.mixed.iterkeys():
            #self.rank[s] = self.OWA((self.mixed[s][:,4:]), w)            
            self.rank[s] = self.Tchebychev(self.mixed[s][:,4:], w, 0.01)
            #i = np.argmax(self.rank[s])
            i = np.argmin(self.rank[s])
            m = self.m_order[int(self.mixed[s][i,0])]
            ind = self.pareto[m][s][(self.pareto[m][s][:,0] == self.mixed[s][i][1])*(self.pareto[m][s][:,1] == self.mixed[s][i][2])*(self.pareto[m][s][:,2] == self.mixed[s][i][3])][0]
            self.indd[s] = ind            
            self.p_test[s] = {m:{}}
            for p in self.p_order[m.split("_")[0]]:
                self.p_test[s][m][p] = ind[self.p_order[m].index(p)+5]

    def OWA(self, value, w):
        m,n=value.shape
        #assert m>=n
        assert len(w) == n
        assert np.sum(w) == 1
        return np.sum(np.sort(value)*w,1)

    def Tchebychev(self, value, lambdaa, epsilon):
        m,n = value.shape
        #assert m>=n
        assert len(lambdaa) == n
        assert np.sum(lambdaa) == 1
        assert epsilon < 1.0
        ideal = np.max(value, 0)
        nadir = np.min(value, 0)
        tmp = lambdaa*((ideal-value)/(ideal-nadir))
        return np.max(tmp, 1)+epsilon*np.sum(tmp,1) 

    def preview(self):
        fig_pareto = figure(figsize = (12, 9))
        #fig_par = figure(figsize = (12, 9))
        #fig_p = figure(figsize = (12,9))
        rcParams['ytick.labelsize'] = 8
        rcParams['xtick.labelsize'] = 8
        # n_params_max = np.max([len(t) for t in [self.p_order[m.split("_")[0]] for m in self.pareto.keys()]])
        # n_model = len(self.pareto.keys())
        # for i in xrange(n_model):
        #     m = self.pareto.keys()[i]
        #     for j in xrange(len(self.p_order[m.split("_")[0]])):
        #         p = self.p_order[m.split("_")[0]][j]
        #         ax2 = fig_par.add_subplot(n_params_max, n_model, i+1+n_model*j)                
        #         ax3 = fig_p.add_subplot(n_params_max, n_model, i+1+n_model*j)
        #         for s in self.pareto[m].iterkeys():
        #             y, x = np.histogram(self.pareto[m][s][:,5+j])
        #             y = y/np.sum(y.astype("float"))
        #             x = (x-(x[1]-x[0])/2)[1:]
        #             ax2.plot(x, y, 'o-', linewidth = 2)
        #             ax2.set_ylim(0, 1)
        #             ax2.set_xlim(self.models[m.split("_")[0]].bounds[p][0],self.models[m.split("_")[0]].bounds[p][1])
        #             ax2.set_xlabel(p)
        #             ax3.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,5+j], 'o-')
        #             ax3.set_ylim(self.models[m.split("_")[0]].bounds[p][0],self.models[m.split("_")[0]].bounds[p][1])
        #             ax3.set_ylabel(p)
        #         if j == 0:
        #             ax2.set_title(m)
                
        for s, i in zip(self.mixed.keys(), xrange(len(self.mixed.keys()))):
            ax1 = fig_pareto.add_subplot(4,4,i+1)
            ax1.scatter(self.indd[s][3], self.indd[s][4], s = 100, color = 'black')
            for m in self.pareto.iterkeys():                
                ax1.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,4], "-o", color = self.colors_m[m], alpha = 0.6)
                ax1.set_title(s)                            

        #fig_par.subplots_adjust(hspace = 0.8, top = 0.98, bottom = 0.1)
        fig_pareto.subplots_adjust(left = 0.08, wspace = 0.26, hspace = 0.26, right = 0.92, top = 0.96)
        #fig_p.subplots_adjust(left = 0.08, wspace = 0.26, hspace = 0.26, right = 0.92, top = 0.96)
        show()        
    
    def _convertStimulus(self, s):
        return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

    # def alignToMedian(self, m, s, n_blocs, n_trials):
    #     x = np.array(self.models[m].reaction).flatten()
    #     y = np.array([self.human.subject['fmri'][s][i]['rt'][:,0][0:n_trials] for i in xrange(1, n_blocs+1)])                
    #     Ex = np.percentile(x, 75) - np.percentile(x, 25)        
    #     Ey = np.percentile(y, 75) - np.percentile(y, 25)
    #     if Ex == 0.0: Ex = 1.0
    #     x = x*(Ey/Ex)
    #     x = x-(np.median(x)-np.median(y))
    #     self.models[m].reaction = x.reshape(n_blocs, n_trials)

    def alignToMedian(self, m, s, n_blocs, n_trials):
        p = np.sum(np.array(self.models[m].pdf), 0)
        p = p/p.sum()        
        wp = []
        tmp = np.cumsum(p)
        f = lambda x: (x-np.sum(tmp<x)*tmp[np.sum(tmp<x)-1]+(np.sum(tmp<x)-1.0)*tmp[np.sum(tmp<x)])/(tmp[np.sum(tmp<x)]-tmp[np.sum(tmp<x)-1])
        for i in [0.25, 0.75]:
            if np.min(tmp)>i:
                wp.append(0.0)
            else:
                wp.append(f(i))              
        yy = np.array([self.human.subject['fmri'][s][i]['rt'][:,0][0:n_trials] for i in xrange(1, n_blocs+1)])                                
        xx = np.array(self.models[m].reaction).flatten()
        b = np.arange(int(self.models[m].parameters['length'])+1)
        wh = [np.percentile(yy, i) for i in [25, 75]]
        if (wp[1]-wp[0]):
            xx = xx*((wh[1]-wh[0])/(wp[1]-wp[0]))
            b = b*((wh[1]-wh[0])/(wp[1]-wp[0]))
        f = lambda x: (x*(b[np.sum(tmp<x)]-b[np.sum(tmp<x)-1])-b[np.sum(tmp<x)]*tmp[np.sum(tmp<x)-1]+b[np.sum(tmp<x)-1]*tmp[np.sum(tmp<x)])/(tmp[np.sum(tmp<x)]-tmp[np.sum(tmp<x)-1])
        half = f(0.5) if np.min(tmp)<0.5 else 0.0        
        
        xx = xx-(half-np.median(yy))
        self.models[m].reaction = xx.reshape(n_blocs, n_trials)

    def learnRBM(self, m, s, n_blocs, n_trials):
        x = np.array([self.human.subject['fmri'][s][i]['rt'][0:n_trials,0] for i in range(1,n_blocs+1)]).flatten()
        #x = 1./x
        x_bin_size = 2*(np.percentile(x, 75)-np.percentile(x, 25))*np.power(len(x), -(1/3.))
        px, xedges = np.histogram(x, bins=np.arange(x.min(), x.max()+ x_bin_size, x_bin_size))
        px = px/float(px.sum())
        xposition = np.digitize(x, xedges)-1

        y = self.models[m].reaction.flatten()
        y_bin_size = 2*(np.percentile(y, 75)-np.percentile(y, 25))*np.power(len(y), -(1/3.))
        py, yedges = np.histogram(y, bins=np.arange(y.min(), y.max()+y_bin_size, y_bin_size))
        py = py/float(py.sum())
        yposition = np.digitize(y, yedges)-1

        f = lambda i, x1, x2, y1, y2: (i*(y2-y1)-y2*x1+y1*x2)/(x2-x1)
        xdata = np.zeros((x.shape[0], xposition.max()+1))
        for i in xrange(xposition.shape[0]): xdata[i,xposition[i]] = 1.0
        ydata = np.zeros((y.shape[0], yposition.max()+1))
        for i in xrange(yposition.shape[0]): ydata[i,yposition[i]] = 1.0

        rbm = RBM(xdata, ydata, nh = 10, nbiter = 1000)
        rbm.train()
                
        Y = rbm.getInputfromOutput(ydata)

        tirage = np.argmax(Y, 1)
        
        center = xedges[1:]-(x_bin_size/2.)
                
        self.models[m].reaction = np.reshape(center[tirage], (n_blocs, n_trials))        

    def leastSquares(self, m, s, n_blocs, n_trials):
        rt = np.array([self.human.subject['fmri'][s][i]['rt'][0:n_trials,0] for i in range(1,n_blocs+1)]).flatten()
        rtm = self.models[m].reaction.flatten()
        state = np.array([self.human.subject['fmri'][s][i]['sar'][0:n_trials,0] for i in range(1,n_blocs+1)])
        action = np.array([self.human.subject['fmri'][s][i]['sar'][0:n_trials,1] for i in range(1,n_blocs+1)])
        responses = np.array([self.human.subject['fmri'][s][i]['sar'][0:n_trials,2] for i in range(1,n_blocs+1)])
        pinit = [1.0, -1.0]
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y : (y - fitfunc(p, x))
        mean = []
        for i in [rt,rtm]:
            tmp = i.reshape(n_blocs, n_trials)
            step, indice = getRepresentativeSteps(tmp, state, action, responses)
            mean.append(computeMeanRepresentativeSteps(step))
        mean = np.array(mean)
        p = leastsq(errfunc, pinit, args = (mean[1][0], mean[0][0]), full_output = False)        
        #self.models[m].reaction = fitfunc(p[0], mean[1][0])
        self.models[m].reaction = mean[1][0]
        ###
        self.hrt.append(mean[0][0])
        ###

    def run(self, plot=True):
        nb_blocs = 4
        nb_trials = self.human.responses['fmri'].shape[1]
        cats = CATS(nb_trials)
        ###
        self.hrt = []
        ###
        for s in self.p_test.iterkeys():             
            m = self.p_test[s].keys()[0]
            print "Testing "+s+" with "+m            
            self.models[m].setAllParameters(self.p_test[s][m])
            self.models[m].startExp()
            for i in xrange(nb_blocs):
                cats.reinitialize()
                cats.stimuli = np.array(map(self._convertStimulus, self.human.subject['fmri'][s][i+1]['sar'][:,0]))
                self.models[m].startBloc()                
                for j in xrange(nb_trials):
                    state = cats.getStimulus(j)
                    action = self.models[m].chooseAction(state)
                    reward = cats.getOutcome(state, action)
                    self.models[m].updateValue(reward)            
            self.models[m].reaction = np.array(self.models[m].reaction)
            
            #self.learnRBM(m, s, nb_blocs, nb_trials)
            #sys.exit()
            #self.alignToMedian(m, s, nb_blocs, nb_trials)
            self.leastSquares(m, s, nb_blocs, nb_trials)

            self.beh['reaction'].append(self.models[m].reaction)
            for i in xrange(nb_blocs):
                self.beh['state'].append(self.models[m].state[i])
                self.beh['action'].append(self.models[m].action[i])
                self.beh['responses'].append(self.models[m].responses[i])

        self.hrt = np.array(self.hrt)
        for k in self.beh.iterkeys():
            self.beh[k] = np.array(self.beh[k])
        self.beh['state'] = convertStimulus(self.beh['state'])
        self.beh['action'] = convertAction(self.beh['action'])
         
    
        if plot:                                                            
            pcr = extractStimulusPresentation(self.beh['responses'], self.beh['state'], self.beh['action'], self.beh['responses'])
            pcr_human = extractStimulusPresentation(self.human.responses['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])            
                                                            
            #step, indice = getRepresentativeSteps(self.beh['reaction'], self.beh['state'], self.beh['action'], self.beh['responses'])
            #rt = computeMeanRepresentativeSteps(step)
            step, indice = getRepresentativeSteps(self.human.reaction['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])
            rt_human = computeMeanRepresentativeSteps(step) 
            rt = (np.mean(self.beh['reaction'], 0), np.var(self.beh['reaction'], 0))

            colors = ['blue', 'red', 'green']
            self.fig_quick = figure(figsize=(10,5))
            ax1 = self.fig_quick.add_subplot(1,2,1)
            [ax1.errorbar(range(1, len(pcr['mean'][t])+1), pcr['mean'][t], pcr['sem'][t], linewidth = 1.5, elinewidth = 1.5, capsize = 0.8, linestyle = '-', alpha = 1, color = colors[t]) for t in xrange(3)]
            [ax1.errorbar(range(1, len(pcr_human['mean'][t])+1), pcr_human['mean'][t], pcr_human['sem'][t], linewidth = 2.5, elinewidth = 1.5, capsize = 0.8, linestyle = '--', alpha = 0.7,color = colors[t]) for t in xrange(3)]    
            ax2 = self.fig_quick.add_subplot(1,2,2)
            ax2.errorbar(range(1, len(rt[0])+1), rt[0], rt[1], linewidth = 2.0, elinewidth = 1.5, capsize = 1.0, linestyle = '-', color = 'black', alpha = 1.0)        
            #ax3 = ax2.twinx()
            ax2.errorbar(range(1, len(rt_human[0])+1), rt_human[0], rt_human[1], linewidth = 2.5, elinewidth = 2.5, capsize = 1.0, linestyle = '--', color = 'grey', alpha = 0.7)
            show()
