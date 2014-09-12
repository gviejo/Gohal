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

#from fonctions import *
from Selection import *
from Models import *
from HumanLearning import HLearning
#from ColorAssociationTasks import CATS
#from scipy.stats import sem
#from scipy.stats import norm
#from scipy.optimize import leastsq

def unwrap_self_load_data(arg, **kwarg):
    return pareto.loadPooled(*arg, **kwarg)

def unwrap_self_re_test(arg, **kwarg):
    return pareto.poolTest(*arg, **kwarg)

class EA():
    """
    Optimization is made for one subject
    """
    def __init__(self, data, subject, ptr_model):
        self.model = ptr_model
        self.subject = subject
        self.data = data
        self.n_repets = 5 # if changing here don't forget to change inside models
        self.n_rs = 15
        self.mean = np.zeros((2,self.n_rs))
        self.fit = np.zeros(2)
        self.rt = self.data['rt'] # array (4*39)
        self.state = self.data['state'] # array int (4*39) 
        self.action = self.data['action'] # array int (4*39)
        self.responses = self.data['reward'] # array int (4*39)
        self.indice = self.data['indice'] # array int (4*39)
        self.mean[0] = self.data['mean'][0] # array (15) centered on median for human
        self.n_trials = self.state.shape[1]
        self.n_blocs = self.state.shape[0]

    def getFitness(self):
        np.seterr(all = 'ignore')
        #self.model.startExp()        
        for i in xrange(self.n_blocs*self.n_repets):
                self.model.startBloc()
                for j in xrange(self.n_trials):
                    self.model.computeValue(self.state[i%self.n_blocs,j]-1, self.action[i%self.n_blocs,j]-1, (i,j))
                    self.model.updateValue(self.responses[i%self.n_blocs,j])
        
        self.fit[0] = float(np.sum(np.log(self.model.value)))        
        self.alignToMedian()        
        self.fit[1] = float(-self.leastSquares())        
        self.fit = np.round(self.fit, 4)
        self.fit[np.isnan(self.fit)] = -100000.0
        self.fit[np.isinf(self.fit)] = -100000.0                
        choice = str(self.fit[0]+2000.0)
        rt = str(self.fit[1]+500.0)
        # FUCKING UGLY ########
        if choice == '0.0' or choice == '0': choice = '-10000.0'
        if rt == '0.0' or rt == '0': rt = '-10000.0'
        #######################
        
        return choice, rt

    def leastSquares(self):        
        self.indice = np.tile(self.indice, (self.n_repets, 1))
        for i in xrange(self.n_rs):            
            self.mean[1,i] = np.mean(self.model.reaction[self.indice == i+1])
        return np.sum(np.power(self.mean[0]-self.mean[1], 2))            
        #return np.sum(np.power(self.errfunc(p[0], mean[1][0], mean[0][0]), 2))

    def alignToMedian(self):        
        self.model.reaction = self.model.reaction - np.median(self.model.reaction)
        self.model.reaction = self.model.reaction / (np.percentile(self.model.reaction, 75)-np.percentile(self.model.reaction, 25))        


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
    def __init__(self, directory, best, N = 156.):
        self.directory = directory        
        self.N = N
        self.best = best
        # loading pre-treated data for fmri
        self.human = dict({s_dir.split(".")[0]:self.pickling("fmri/"+s_dir) for s_dir in os.listdir("fmri/")})
        # making bound for rt et likelihood
        self.front_bounds = dict(zip(self.human.keys(), np.zeros((len(self.human.keys()), 2))))
        self.bounding_front()
        self.data = dict()
        self.states = ['s1', 's2', 's3']
        self.actions = ['thumb', 'fore', 'midd', 'ring', 'little']
        self.models = dict({"fusion":FSelection(self.states, self.actions),
                            "qlearning":QLearning(self.states, self.actions),
                            "bayesian":BayesianWorkingMemory(self.states, self.actions),
                            "selection":KSelection(self.states, self.actions),
                            "mixture":CSelection(self.states, self.actions)})

        self.p_order = dict({'fusion':['alpha','beta', 'gamma', 'noise','length','gain','threshold', 'sigma'],
                            'qlearning':['alpha','beta','gamma','sigma'],
                            'bayesian':['length','noise','threshold', 'sigma'],
                            'selection':['gamma','beta','eta','length','threshold','noise','sigma', 'sigma_rt'],
                            'mixture':['alpha', 'beta', 'gamma', 'noise', 'length', 'weight', 'threshold', 'sigma', 'gain']})

        self.m_order = ['qlearning', 'bayesian', 'selection', 'fusion', 'mixture']
        self.colors_m = dict({'fusion':'r', 'bayesian':'g', 'qlearning':'grey', 'selection':'b', 'mixture':'y'})
        self.opt = dict()
        self.pareto = dict()
        self.distance = dict()
        self.owa = dict()
        self.tche = dict()
        self.p_test = dict()
        self.mixed = dict()
        self.beh = dict({'state':[],'action':[],'responses':[],'reaction':[]})
        self.indd = dict()
        self.zoom = dict()
        self.loadData()
        #self.simpleLoadData()

    def bounding_front(self):        
        for s in self.human.keys():
            self.front_bounds[s][0] = self.N*np.log(0.2)*5
            self.front_bounds[s][1] = -np.sum(2*np.abs(self.human[s]['mean'][0]))

    def pickling(self, direc):
        with open(direc, "rb") as f:
            return pickle.load(f)

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
                # print m, s        
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
                self.pareto[m][s][:,3] = self.pareto[m][s][:,3] - np.log(self.N)*float(len(self.models[m].bounds.keys()))
                #self.pareto[m][s][:,4] = self.pareto[m][s][:,4] - np.log(self.N)*float(len(self.models[m].bounds.keys()))                
                for i in xrange(2): 
                    self.pareto[m][s] = self.pareto[m][s][self.pareto[m][s][:,3+i] >= self.front_bounds[s][i]]
                if len(self.pareto[m][s]):
                    self.pareto[m][s][:,3:5] = (self.pareto[m][s][:,3:5]-self.front_bounds[s])/(self.best-self.front_bounds[s])
                else:
                    print "No point for ", s, m


    def constructMixedParetoFrontier(self):
        # One transformation is applied to rescale each fitness function
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
            #self.mixed[s][:,4:6] = (self.mixed[s][:,4:6]-self.front_bounds[s])/(self.best-self.front_bounds[s])

    def removeIndivDoublons(self):
        for m in self.pareto.iterkeys():
            for s in self.pareto[m].iterkeys():
                if len(self.pareto[m][s]):
                    # start at column 5; for each parameters columns, find the minimal number of value
                    # then mix all parameters
                    tmp = np.zeros((len(self.pareto[m][s]),len(self.p_order[m])))
                    for i in xrange(len(self.p_order[m])):
                        tmp[:,i][np.unique(self.pareto[m][s][:,i+5], return_index = True)[1]] = 1.0
                    self.pareto[m][s] = self.pareto[m][s][tmp.sum(1)>0]

    def reTest(self, n):        
        pool = Pool(len(self.pareto.keys()))
        tmp = pool.map(unwrap_self_re_test, zip([self]*len(self.pareto.keys()), self.pareto.iterkeys(), [n]*len(self.pareto.keys())))
        return tmp
                                

    def poolTest(self, m, n):       
        models = dict({"fusion":FSelection(self.states, self.actions, sferes = True),
                "qlearning":QLearning(self.states, self.actions, sferes = True),
                "bayesian":BayesianWorkingMemory(self.states, self.actions, sferes = True),
                "selection":KSelection(self.states, self.actions, sferes = True),
                "mixture":CSelection(self.states, self.actions, sferes = True)})        
        model = models[m]
        for s in self.pareto[m].iterkeys():                
                for i in xrange(len(self.pareto[m][s])):                    
                    parameters = dict(zip(self.p_order[m], self.pareto[m][s][i,5:]))
                    model.setAllParameters(parameters)
                    model.value = np.zeros((4*n,39))
                    model.reaction = np.zeros((4*n,39))
                    with open("fmri/"+s+".pickle", "rb") as f:
                        data = pickle.load(f)
                    opt = EA(data, s, model)
                    opt.n_repets = n
                    fit1, fit2 = opt.getFitness()
                    self.pareto[m][s][i,3] = float(fit1)
                    self.pareto[m][s][i,4] = float(fit2)

    def rankDistance(self):
        self.p_test['distance'] = dict()
        for s in self.mixed.iterkeys():
            self.distance[s] = np.zeros((len(self.mixed[s]), 3))
            self.distance[s][:,1] = np.sqrt(np.sum(np.power(self.mixed[s][:,4:6]-np.ones(2), 2),1))
            ind_best_point = np.argmin(self.distance[s][:,1])
            best_point = self.mixed[s][ind_best_point,4:6]
            self.distance[s][:,0] = np.sqrt(np.sum(np.power(self.mixed[s][:,4:6]-best_point,2),1))
            self.distance[s][0:ind_best_point,0] = -1.0*self.distance[s][0:ind_best_point,0]
            self.distance[s][0:ind_best_point,2] = np.arange(-ind_best_point,0)
            self.distance[s][ind_best_point:,2] = np.arange(0, len(self.distance[s])-ind_best_point)
            # Saving best individual                        
            best_ind = self.mixed[s][ind_best_point]
            m = self.m_order[int(best_ind[0])]            
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['distance'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})

    def rankOWA(self):
        self.p_test['owa'] = dict()
        for s in self.mixed.iterkeys():
            tmp = self.mixed[s][:,4:6]
            value = np.sum(np.sort(tmp)*[0.5, 0.5], 1)
            self.owa[s] = value
            ind_best_point = np.argmax(value)
            # Saving best indivudual
            best_ind = self.mixed[s][ind_best_point]
            m = self.m_order[int(best_ind[0])]
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['owa'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})            

    def rankTchebytchev(self, lambdaa = 0.5, epsilon = 0.001):
        self.p_test['tche'] = dict()
        for s in self.mixed.iterkeys():
            tmp = self.mixed[s][:,4:6]
            ideal = np.max(tmp, 0)
            nadir = np.min(tmp, 0)
            value = lambdaa*((ideal-tmp)/(ideal-nadir))
            value = np.max(value, 1)+epsilon*np.sum(value,1)
            self.tche[s] = value
            ind_best_point = np.argmin(value)
            # Saving best individual
            best_ind = self.mixed[s][ind_best_point]
            m = self.m_order[int(best_ind[0])]
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['tche'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})                        

    def preview(self):
        rcParams['ytick.labelsize'] = 8
        rcParams['xtick.labelsize'] = 8        
        fig_model = figure(figsize = (10,10)) # for each model all subject            
        fig_rank = figure(figsize = (6,6))         
        for m,i in zip(self.pareto.iterkeys(), xrange(len(self.pareto.keys()))):
            ax2 = fig_model.add_subplot(3,2,i+1)
            for s in self.pareto[m].iterkeys():
                ax2.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,4], "-o", color = self.colors_m[m], alpha = 1.0)        
            ax2.set_title(m)
            ax2.set_xlim(0,1)
            ax2.set_ylim(0,1)
        ax4 = fig_model.add_subplot(3,2,6)                                    
        ax5 = fig_rank.add_subplot(1,1,1)
        for s in self.mixed.keys():
            for m in np.unique(self.mixed[s][:,0]):
                ind = self.mixed[s][:,0] == m
                ax4.plot(self.mixed[s][ind,4], self.mixed[s][ind,5], 'o-', color = self.colors_m[self.m_order[int(m)]])
                ax5.plot(self.distance[s][ind,0], self.distance[s][ind,1], 'o-', color = self.colors_m[self.m_order[int(m)]])
                #ax5.plot(self.distance[s][ind,2], self.distance[s][ind,1], 'o-', color = self.colors_m[self.m_order[int(m)]])
        ax5.axvline(0.0)
        
        fig_zoom = figure(figsize = (5,5))
        ax6 = fig_zoom.add_subplot(1,1,1)
        for s in self.zoom.keys():
            print s
            ax6.plot(self.zoom[s][:,0], self.zoom[s][:,1], '.-', color = 'grey')
            ax6.plot(self.zoom[s][np.argmin(self.zoom[s][:,2]),0], self.zoom[s][np.argmin(self.zoom[s][:,2]),1], '*', markersize = 15, color = 'blue', alpha = 0.5)
            ax6.plot(self.zoom[s][np.argmax(self.zoom[s][:,3]),0], self.zoom[s][np.argmax(self.zoom[s][:,3]),1], '^', markersize = 15, color = 'red', alpha = 0.5)
            ax6.plot(self.zoom[s][np.argmin(self.zoom[s][:,4]),0], self.zoom[s][np.argmin(self.zoom[s][:,4]),1], 'o', markersize = 15, color = 'green', alpha = 0.5)
        ax6.set_xlim(0,1)
        ax6.set_ylim(0,1)
        
    def zoomBox(self, xmin, ymin):
        for s in self.mixed.iterkeys():
            self.zoom[s] = np.hstack((self.mixed[s][:,4:6], self.distance[s][:,1:2], np.vstack(self.owa[s]), np.vstack(self.tche[s])))
            if np.sum((self.zoom[s][:,0] > xmin)*(self.zoom[s][:,1] > ymin)):
                self.zoom[s] = self.zoom[s][self.zoom[s][:,0] > xmin]
                self.zoom[s] = self.zoom[s][self.zoom[s][:,1] > ymin]
            else:
                self.zoom.pop(s)       
        








    # def rankMixedFront(self, w):    
    #     for s in self.mixed.iterkeys():
    #         # self.distance[s] = self.OWA((self.mixed[s][:,4:]), w)            
    #         if s in w.keys():
    #             self.distance[s] = self.Tchebychev(self.mixed[s][:,4:], w[s], 0.01)
    #         else:
    #             self.distance[s] = self.Tchebychev(self.mixed[s][:,4:], [0.5,0.5], 0.01)
    #         # i = np.argmax(self.distance[s])
    #         i = np.argmin(self.distance[s])
    #         m = self.m_order[int(self.mixed[s][i,0])]
    #         ind = self.pareto[m][s][(self.pareto[m][s][:,0] == self.mixed[s][i][1])*(self.pareto[m][s][:,1] == self.mixed[s][i][2])*(self.pareto[m][s][:,2] == self.mixed[s][i][3])][0]
    #         self.indd[s] = ind            
    #         self.p_test[s] = {m:{}}
    #         for p in self.p_order[m.split("_")[0]]:
    #             self.p_test[s][m][p] = ind[self.p_order[m].index(p)+5]

    # def OWA(self, value, w):
    #     m,n=value.shape
    #     #assert m>=n
    #     assert len(w) == n
    #     assert np.sum(w) == 1
    #     return np.sum(np.sort(value)*w,1)

    # def Tchebychev(self, value, lambdaa, epsilon):
    #     m,n = value.shape
    #     #assert m>=n
    #     assert len(lambdaa) == n
    #     assert np.sum(lambdaa) == 1
    #     assert epsilon < 1.0
    

    # def _convertStimulus(self, s):
    #     return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

    # # def alignToMedian(self, m, s, n_blocs, n_trials):
    # #     x = np.array(self.models[m].reaction).flatten()
    # #     y = np.array([self.human.subject['fmri'][s][i]['rt'][:,0][0:n_trials] for i in xrange(1, n_blocs+1)])                
    # #     Ex = np.percentile(x, 75) - np.percentile(x, 25)        
    # #     Ey = np.percentile(y, 75) - np.percentile(y, 25)
    # #     if Ex == 0.0: Ex = 1.0
    # #     x = x*(Ey/Ex)
    # #     x = x-(np.median(x)-np.median(y))
    # #     self.models[m].reaction = x.reshape(n_blocs, n_trials)

    # def alignToMedian(self, m, s, n_blocs, n_trials):
    #     p = np.sum(np.array(self.models[m].pdf), 0)
    #     p = p/p.sum()        
    #     wp = []
    #     tmp = np.cumsum(p)
    #     f = lambda x: (x-np.sum(tmp<x)*tmp[np.sum(tmp<x)-1]+(np.sum(tmp<x)-1.0)*tmp[np.sum(tmp<x)])/(tmp[np.sum(tmp<x)]-tmp[np.sum(tmp<x)-1])
    #     for i in [0.25, 0.75]:
    #         if np.min(tmp)>i:
    #             wp.append(0.0)
    #         else:
    #             wp.append(f(i))              
    #     yy = np.array([self.human.subject['fmri'][s][i]['rt'][:,0][0:n_trials] for i in xrange(1, n_blocs+1)])                                
    #     xx = np.array(self.models[m].reaction).flatten()
    #     b = np.arange(int(self.models[m].parameters['length'])+1)
    #     wh = [np.percentile(yy, i) for i in [25, 75]]
    #     if (wp[1]-wp[0]):
    #         xx = xx*((wh[1]-wh[0])/(wp[1]-wp[0]))
    #         b = b*((wh[1]-wh[0])/(wp[1]-wp[0]))
    #     f = lambda x: (x*(b[np.sum(tmp<x)]-b[np.sum(tmp<x)-1])-b[np.sum(tmp<x)]*tmp[np.sum(tmp<x)-1]+b[np.sum(tmp<x)-1]*tmp[np.sum(tmp<x)])/(tmp[np.sum(tmp<x)]-tmp[np.sum(tmp<x)-1])
    #     half = f(0.5) if np.min(tmp)<0.5 else 0.0        
        
    #     xx = xx-(half-np.median(yy))
    #     self.models[m].reaction = xx.reshape(n_blocs, n_trials)

    # def learnRBM(self, m, s, n_blocs, n_trials):
    #     x = np.array([self.human.subject['fmri'][s][i]['rt'][0:n_trials,0] for i in range(1,n_blocs+1)]).flatten()
    #     #x = 1./x
    #     x_bin_size = 2*(np.percentile(x, 75)-np.percentile(x, 25))*np.power(len(x), -(1/3.))
    #     px, xedges = np.histogram(x, bins=np.arange(x.min(), x.max()+ x_bin_size, x_bin_size))
    #     px = px/float(px.sum())
    #     xposition = np.digitize(x, xedges)-1

    #     y = self.models[m].reaction.flatten()
    #     y_bin_size = 2*(np.percentile(y, 75)-np.percentile(y, 25))*np.power(len(y), -(1/3.))
    #     py, yedges = np.histogram(y, bins=np.arange(y.min(), y.max()+y_bin_size, y_bin_size))
    #     py = py/float(py.sum())
    #     yposition = np.digitize(y, yedges)-1

    #     f = lambda i, x1, x2, y1, y2: (i*(y2-y1)-y2*x1+y1*x2)/(x2-x1)
    #     xdata = np.zeros((x.shape[0], xposition.max()+1))
    #     for i in xrange(xposition.shape[0]): xdata[i,xposition[i]] = 1.0
    #     ydata = np.zeros((y.shape[0], yposition.max()+1))
    #     for i in xrange(yposition.shape[0]): ydata[i,yposition[i]] = 1.0

    #     rbm = RBM(xdata, ydata, nh = 10, nbiter = 1000)
    #     rbm.train()
                
    #     Y = rbm.getInputfromOutput(ydata)

    #     tirage = np.argmax(Y, 1)
        
    #     center = xedges[1:]-(x_bin_size/2.)
                
    #     self.models[m].reaction = np.reshape(center[tirage], (n_blocs, n_trials))        

    # def leastSquares(self, m, s, n_blocs, n_trials):
    #     rt = np.array([self.human.subject['fmri'][s][i]['rt'][0:n_trials,0] for i in range(1,n_blocs+1)]).flatten()
    #     rtm = self.models[m].reaction.flatten()
    #     state = np.array([self.human.subject['fmri'][s][i]['sar'][0:n_trials,0] for i in range(1,n_blocs+1)])
    #     action = np.array([self.human.subject['fmri'][s][i]['sar'][0:n_trials,1] for i in range(1,n_blocs+1)])
    #     responses = np.array([self.human.subject['fmri'][s][i]['sar'][0:n_trials,2] for i in range(1,n_blocs+1)])
    #     pinit = [1.0, -1.0]
    #     fitfunc = lambda p, x: p[0] + p[1] * x
    #     errfunc = lambda p, x, y : (y - fitfunc(p, x))
    #     mean = []
    #     for i in [rt,rtm]:
    #         tmp = i.reshape(n_blocs, n_trials)
    #         step, indice = getRepresentativeSteps(tmp, state, action, responses)
    #         mean.append(computeMeanRepresentativeSteps(step))
    #     mean = np.array(mean)
    #     p = leastsq(errfunc, pinit, args = (mean[1][0], mean[0][0]), full_output = False)        
    #     #self.models[m].reaction = fitfunc(p[0], mean[1][0])
    #     self.models[m].reaction = mean[1][0]
    #     ###
    #     self.hrt.append(mean[0][0])
    #     ###

    # def run(self, plot=True):
    #     nb_blocs = 4
    #     nb_trials = self.human.responses['fmri'].shape[1]
    #     cats = CATS(nb_trials)
    #     ###
    #     self.hrt = []
    #     ###
    #     for s in self.p_test.iterkeys():             
    #         m = self.p_test[s].keys()[0]
    #         print "Testing "+s+" with "+m            
    #         self.models[m].setAllParameters(self.p_test[s][m])
    #         self.models[m].startExp()
    #         for i in xrange(nb_blocs):
    #             cats.reinitialize()
    #             cats.stimuli = np.array(map(self._convertStimulus, self.human.subject['fmri'][s][i+1]['sar'][:,0]))
    #             self.models[m].startBloc()                
    #             for j in xrange(nb_trials):
    #                 state = cats.getStimulus(j)
    #                 action = self.models[m].chooseAction(state)
    #                 reward = cats.getOutcome(state, action)
    #                 self.models[m].updateValue(reward)            
    #         self.models[m].reaction = np.array(self.models[m].reaction)
            
    #         #self.learnRBM(m, s, nb_blocs, nb_trials)
    #         #sys.exit()
    #         #self.alignToMedian(m, s, nb_blocs, nb_trials)
    #         self.leastSquares(m, s, nb_blocs, nb_trials)

    #         self.beh['reaction'].append(self.models[m].reaction)
    #         for i in xrange(nb_blocs):
    #             self.beh['state'].append(self.models[m].state[i])
    #             self.beh['action'].append(self.models[m].action[i])
    #             self.beh['responses'].append(self.models[m].responses[i])

    #     self.hrt = np.array(self.hrt)
    #     for k in self.beh.iterkeys():
    #         self.beh[k] = np.array(self.beh[k])
    #     self.beh['state'] = convertStimulus(self.beh['state'])
    #     self.beh['action'] = convertAction(self.beh['action'])
         
    
    #     if plot:                                                            
    #         pcr = extractStimulusPresentation(self.beh['responses'], self.beh['state'], self.beh['action'], self.beh['responses'])
    #         pcr_human = extractStimulusPresentation(self.human.responses['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])            
                                                            
    #         #step, indice = getRepresentativeSteps(self.beh['reaction'], self.beh['state'], self.beh['action'], self.beh['responses'])
    #         #rt = computeMeanRepresentativeSteps(step)
    #         step, indice = getRepresentativeSteps(self.human.reaction['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])
    #         rt_human = computeMeanRepresentativeSteps(step) 
    #         rt = (np.mean(self.beh['reaction'], 0), np.var(self.beh['reaction'], 0))

    #         colors = ['blue', 'red', 'green']
    #         self.fig_quick = figure(figsize=(10,5))
    #         ax1 = self.fig_quick.add_subplot(1,2,1)
    #         [ax1.errorbar(range(1, len(pcr['mean'][t])+1), pcr['mean'][t], pcr['sem'][t], linewidth = 1.5, elinewidth = 1.5, capsize = 0.8, linestyle = '-', alpha = 1, color = colors[t]) for t in xrange(3)]
    #         [ax1.errorbar(range(1, len(pcr_human['mean'][t])+1), pcr_human['mean'][t], pcr_human['sem'][t], linewidth = 2.5, elinewidth = 1.5, capsize = 0.8, linestyle = '--', alpha = 0.7,color = colors[t]) for t in xrange(3)]    
    #         ax2 = self.fig_quick.add_subplot(1,2,2)
    #         ax2.errorbar(range(1, len(rt[0])+1), rt[0], rt[1], linewidth = 2.0, elinewidth = 1.5, capsize = 1.0, linestyle = '-', color = 'black', alpha = 1.0)        
    #         #ax3 = ax2.twinx()
    #         ax2.errorbar(range(1, len(rt_human[0])+1), rt_human[0], rt_human[1], linewidth = 2.5, elinewidth = 2.5, capsize = 1.0, linestyle = '--', color = 'grey', alpha = 0.7)
    #         show()
