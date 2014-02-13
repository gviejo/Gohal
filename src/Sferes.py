#!/usr/bin/python
# encoding: utf-8
"""
Sferes.py

    class for multi-objective optimization
    to interface with sferes2 : see
    http://sferes2.isir.upmc.fr/
    fitness function is made of Bayesian Information Criterion
    and either Linear Regression
    or possible Reaction Time Likelihood

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *
from Selection import *
from Models import *
from pylab import *
from HumanLearning import HLearning
from ColorAssociationTasks import CATS
#import scipy.optimize as opt
#from scipy.stats import pearsonr
from scipy.stats import sem
from scipy.stats import norm

        
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
        self.rt = np.array([self.data[i]['rt'][0:self.n_trials,0] for i in [1,2,3,4]]).flatten()
        self.rt_model = None        
        self.state = np.array([self.data[i]['sar'][0:self.n_trials,0] for i in [1,2,3,4]])
        self.action = np.array([self.data[i]['sar'][0:self.n_trials,1] for i in [1,2,3,4]])
        self.responses = np.array([self.data[i]['sar'][0:self.n_trials,2] for i in [1,2,3,4]])

    def getFitness(self):
        np.seterr(all = 'ignore')
        llh = 0.0
        lrs = 0.0
        for i in xrange(self.n_blocs):
            self.model.startBloc()
            for j in xrange(self.n_trials):
                values = self.model.computeValue(self.model.states[int(self.state[i,j])-1])
                llh = llh + np.log(values[int(self.action[i,j])-1])
                self.model.current_action = int(self.action[i,j])-1
                self.model.updateValue(self.responses[i,j])
        self.rt_model = np.array(self.model.reaction).flatten()        
        self.alignToMedian()

        self.density = np.array([norm.logpdf(self.rt[i], self.rt_model[i], self.model.parameters['sigma']) for i in xrange(self.n_trials*self.n_blocs)])  
        lrs = np.sum(self.density)

        return -np.abs(llh), -np.abs(lrs)

    def alignToMedian(self):
        if (np.percentile(self.rt_model, 75)-np.median(self.rt_model)) != 0:
            w = (np.percentile(self.rt, 75)-np.median(self.rt))/float((np.percentile(self.rt_model, 75)-np.median(self.rt_model)))
            #w = (np.percentile(self.rt, 75)-np.percentile(self.rt, 25))/float((np.percentile(self.rt_model, 75)-np.percentile(self.rt_model, 25)))
            self.rt_model = self.rt_model*w        
        self.rt_model = self.rt_model-(np.median(self.rt_model)-np.median(self.rt))

    def rtLikelihood(self):
        self.model.pdf = np.vstack(map(np.array, self.model.pdf))
        self.model.pdf = np.exp(self.model.pdf*10.0)
        self.model.pdf = self.model.pdf/np.sum(self.model.pdf, 1, keepdims=True)

    def leastSquares(self):
        self.rt_model = self.rt_model-np.mean(self.rt_model)
        self.rt = self.rt-np.mean(self.rt)
        if np.std(self.rt_model):
            self.rt_model = self.rt_model/np.std(self.rt_model)
        self.rt = self.rt/np.std(self.rt)                

class pareto():
    """
    Explore Pareto Front from Sferes Optimization
    """
    def __init__(self, directory):
        self.directory = directory
        self.human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
        self.data = dict()
        self.states = ['s1', 's2', 's3']
        self.actions = ['thumb', 'fore', 'midd', 'ring', 'little']
        self.models = dict({"fusion":FSelection(self.states, self.actions),
                            "qlearning":QLearning(self.states, self.actions),
                            "bayesian":BayesianWorkingMemory(self.states, self.actions),
                            "keramati":KSelection(self.states, self.actions)})
        self.p_order = dict({'fusion':['alpha','beta', 'gamma', 'noise','length','threshold','gain','sigma'],                            
                            'qlearning':['alpha','beta','gamma'],
                            'bayesian':['length','noise','threshold'],
                            'keramati':['gamma','beta','eta','length','threshold','noise','sigma']})
        self.good = dict({'fusion':{'alpha': 0.8,
                                     'beta': 3.0,
                                     'gain': 2.0,
                                     'gamma': 0.4,
                                     'length': 10,
                                     'noise': 0.0001,
                                     'threshold': 1.0}})
        self.opt = dict()
        self.pareto = dict()
        self.rank = dict()
        self.p_test = dict()
        self.loadData()
        self.constructParetoFrontier()

    def loadData(self):
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

    def constructParetoFrontier(self):
        for m in self.data.iterkeys():
            self.pareto[m] = dict()
            for s in self.data[m].iterkeys():
                self.pareto[m][s] = dict()   
                tmp={n:self.data[m][s][n][self.data[m][s][n][:,0]==np.max(self.data[m][s][n][:,0])] for n in self.data[m][s].iterkeys()}
                tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])
                ind = tmp[:,3:5] != 0
                tmp = tmp[ind[:,0]*ind[:,1]]
                tmp = tmp[tmp[:,3].argsort()][::-1]
                pareto_frontier = [tmp[0]]
                for pair in tmp[1:]:
                    if pair[4] >= pareto_frontier[-1][4]:
                        pareto_frontier.append(pair)
                self.pareto[m][s] = np.array(pareto_frontier)


    def rankFront(self, w):
        for m in self.pareto.iterkeys():
            self.opt[m] = dict()
            self.p_test[m] = dict()
            for s in self.pareto[m].iterkeys():
                print s
                self.opt[m][s] = dict()
                self.p_test[m][s] = dict()
                rank = self.OWA(self.pareto[m][s][:,3:5], w)                
                #rank = self.Tchebychev(self.pareto[m][s][:,3:5], w, 0.01)
                self.pareto[m][s] = np.hstack((np.vstack(rank),self.pareto[m][s]))
                self.opt[m][s] = self.pareto[m][s][self.pareto[m][s][:,0] == np.max(self.pareto[m][s][:,0])][0]
                #self.opt[m][s] = self.pareto[m][s][self.pareto[m][s][:,0] == np.min(self.pareto[m][s][:,0])][0]
                for p in self.p_order[m.split("_")[0]]:
                    self.p_test[m][s][p] = self.opt[m][s][self.p_order[m.split("_")[0]].index(p)+6]

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

    def plotParetoFront(self):
        self.fig_pareto = figure(figsize = (12,9))
        for m in self.pareto.iterkeys():
            for i in xrange(len(self.data[m].keys())):
                s = self.data[m].keys()[i]
                ax = self.fig_pareto.add_subplot(4,4,i+1)
                ax.plot(self.pareto[m][s][:,4], self.pareto[m][s][:,5], "-o")
                ax.scatter(self.pareto[m][s][:,4], self.pareto[m][s][:,5], c=self.pareto[m][s][:,0])
                ax.plot(self.opt[m][s][4], self.opt[m][s][5], 'o', markersize = 15, label = m, alpha = 0.8)
                ax.grid()
        rcParams['xtick.labelsize'] = 6
        rcParams['ytick.labelsize'] = 6                
        ax.legend(loc='lower left', bbox_to_anchor=(1.15, 0.2), fancybox=True, shadow=True)
        self.fig_pareto.subplots_adjust(left = 0.08, wspace = 0.26, hspace = 0.26, right = 0.92, top = 0.96)
        self.fig_pareto.show()

    def plotFrontEvolution(self):
        self.fig_evolution = figure(figsize = (12,9))
        for m in self.data.iterkeys():
            for i in xrange(len(self.data[m].keys())):
                s = self.data[m].keys()[i]                
                ax = self.fig_evolution.add_subplot(4,4,i+1)
                for j in self.data[m][s].iterkeys():                    
                    ax.scatter(self.data[m][s][j][:,2], self.data[m][s][j][:,3])
                ax.grid()
        rcParams['xtick.labelsize'] = 6
        rcParams['ytick.labelsize'] = 6                
        ax.legend(loc='lower left', bbox_to_anchor=(1.15, 0.2), fancybox=True, shadow=True)
        self.fig_evolution.subplots_adjust(left = 0.08, wspace = 0.26, hspace = 0.26, right = 0.92, top = 0.96)
        self.fig_evolution.show()
            
    def plotSolutions(self):
        self.fig_solution = figure(figsize= (12,9))
        n_params_max = np.max([len(t) for t in [self.p_order[m.split("_")[0]] for m in self.opt.keys()]])
        n_model = len(self.opt.keys())
        for i in xrange(n_model):
            m = self.opt.keys()[i]
            for j in xrange(len(self.p_order[m.split("_")[0]])):
                p = self.p_order[m.split("_")[0]][j]
                ax = self.fig_solution.add_subplot(n_params_max, n_model, i+1+n_model*j)
                for k in xrange(len(self.opt[m].keys())):
                    s = self.opt[m].keys()[k]
                    ax.scatter(self.opt[m][s][j+6],k+1)
                    #ax.axvline(self.good[m.split("_")[0]][p], 0, 1, linewidth = 2)
                ax.set_xlim(self.models[m.split("_")[0]].bounds[p][0],self.models[m.split("_")[0]].bounds[p][1])
                ax.set_xlabel(p)
        rcParams['xtick.labelsize'] = 6
        rcParams['ytick.labelsize'] = 6
        self.fig_solution.subplots_adjust(hspace = 0.8, top = 0.98, bottom = 0.1)
        self.fig_solution.show()

    def _convertStimulus(self, s):
        return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

    def alignToMedian(self, m, n_subject, n_blocs, n_trials):
        x = np.reshape(self.models[m.split("_")[0]].reaction, (n_subject, n_blocs*n_trials))        
        y = np.reshape(self.human.reaction['fmri'], (14, 4*39))             
        # Ex = np.percentile(x, 75, 1) - np.median(x, 1)
        # Ey = np.percentile(y, 75, 1) - np.median(y, 1)
        Ex = np.percentile(x, 75, 1) - np.percentile(x, 25, 1)
        Ey = np.percentile(y, 75, 1) - np.percentile(y, 25, 1)
        Ex[Ex == 0.0] = 1.0
        x = x*np.vstack(Ey/Ex)
        x = x-np.vstack((np.median(x, 1)-np.median(y,1)))
        self.models[m.split("_")[0]].reaction = np.reshape(x, (n_subject*n_blocs, n_trials))        
        self.human.reaction['fmri'] = np.reshape(y, (14*4, 39))
        
    def representativeSteps(self, m, s_order, n_blocs, n_trials):
        x = np.reshape(self.models[m].reaction, (len(s_order), n_blocs, n_trials))
        y = np.reshape(self.human.reaction['fmri'], (14, 4, 39))
        s = np.reshape(self.human.stimulus['fmri'], (14, 4, 39))
        a = np.reshape(self.human.action['fmri'], (14, 4, 39))
        r = np.reshape(self.human.responses['fmri'], (14, 4, 39))
        rt = []
        rt_model = []
        for i in xrange(len(s_order)):
            step, indice = getRepresentativeSteps(x[i], s[i], a[i], r[i])
            rt.append(np.hstack([np.mean(y[i][indice == j]) for j in step.iterkeys()]))
            rt_model.append(np.hstack([np.mean(x[i][indice == j]) for j in step.iterkeys()]))
        rt = np.array(rt)    
        rt_model = np.array(rt_model)

        Ex = np.percentile(rt_model, 75, 1) - np.median(rt_model, 1)
        Ey = np.percentile(rt, 75, 1) - np.median(rt, 1)
        Ex[Ex == 0.0] = 1.0
        rt_model = rt_model*np.vstack(Ey/Ex)
        rt_model = rt_model-np.vstack((np.median(rt_model, 1)-np.median(rt, 1)))

        rt_model = (np.mean(rt_model, 0), sem(rt_model, 0))
        rt_human = (np.mean(rt, 0), sem(rt, 0))

        return rt_model, rt_human

    def alignToMean(self, m, n_subject, n_blocs, n_trials):
        x = np.reshape(self.models[m.split("_")[0]].reaction, (n_subject, n_blocs*n_trials))        
        y = np.reshape(self.human.reaction['fmri'], (14, 4*39))     
        #w = np.reshape(self.human.weight['fmri'], (14, 4*39))

        x = x-np.vstack(np.mean(x,1))        
        y = y-np.vstack(np.mean(y,1))
        tmp = np.std(x, 1)
        tmp[tmp == 0.0] = 1.0
        x = x/np.vstack(tmp)        
        tmp = np.std(y, 1)
        tmp[tmp == 0.0] = 1.0
        y = y/np.vstack(tmp)

        #x = x - np.vstack(np.median(x, 1)-np.median(y, 1))

                
        # a = np.vstack((np.sum(y*x,1))/(np.sum(x**2,1)))
        # x = a*x

        self.models[m.split("_")[0]].reaction = np.reshape(x, (n_subject*n_blocs, n_trials))        
        self.human.reaction['fmri'] = np.reshape(y, (14*4, 39))        

    def alignToCste(self, m, n_subject, n_blocs, n_trials, s_order):
        x = np.reshape(self.models[m.split("_")[0]].reaction, (n_subject, n_blocs*n_trials))        
        y = np.reshape(self.human.reaction['fmri'], (14, 4*39))
        cste = np.vstack(np.array([self.p_test[m][s]['cste'] for s in s_order]))
        w = (np.max(y)-cste)/np.vstack(np.max(x+1,1))
        x = x*w
        x = x-(np.vstack(np.min(x,1))-cste)

        self.models[m.split("_")[0]].reaction = np.reshape(x, (n_subject*n_blocs, n_trials))        
        self.human.reaction['fmri'] = np.reshape(y, (14*4, 39))

        # self.w = w
        # self.s_order = s_order
        self.x = x
        self.y = y
        # sys.exit()

    def quickTest(self, m, plot=True):
        nb_blocs = 4
        nb_trials = self.human.responses['fmri'].shape[1]
        cats = CATS(nb_trials)
        model = self.models[m.split("_")[0]]
        model.startExp()
        s_order = []
        for s in self.p_test[m].iterkeys():             
            s_order.append(s)           
            model.setAllParameters(self.p_test[m][s])            
            for i in xrange(nb_blocs):
                cats.reinitialize()
                cats.stimuli = np.array(map(self._convertStimulus, self.human.subject['fmri'][s][i+1]['sar'][:,0]))
                model.startBloc()                
                #for j in xrange(len(cats.stimuli)):
                for j in xrange(nb_trials):
                    state = cats.getStimulus(j)
                    action = model.chooseAction(state)
                    reward = cats.getOutcome(state, action)
                    model.updateValue(reward)
        model.state = convertStimulus(np.array(model.state))
        model.action = convertAction(np.array(model.action))
        model.responses = np.array(model.responses)
        model.reaction = np.array(model.reaction)
        if plot:            
            #self.alignToMean(m, len(self.p_test[m].keys()), nb_blocs, nb_trials)
            self.alignToMedian(m, len(self.p_test[m].keys()), nb_blocs, nb_trials)
            #self.alignToCste(m, len(self.p_test[m].keys()), nb_blocs, nb_trials, s_order)
            pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
            pcr_human = extractStimulusPresentation(self.human.responses['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])            
            
            #rt, rt_human = self.representativeSteps(m, s_order, nb_blocs, nb_trials)
                        
            step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
            rt = computeMeanRepresentativeSteps(step)
            step, indice = getRepresentativeSteps(self.human.reaction['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])
            rt_human = computeMeanRepresentativeSteps(step) 


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

    # def aggregate(self, m, plot = False):
    #     self.comb_rank = list()
    #     self.final[m] = dict()
    #     self.combinaison = list()
    #     self.all_front = dict({m:dict()})
    #     for s in self.data[m].iterkeys():
    #         self.all_front[m][s] = dict()
    #         pareto = self.data[m][s][self.data[m][s][:,0] == np.max(self.data[m][s][:,0])]
    #         self.combinaison.append([])
    #         for line in pareto:
    #             self.all_front[m][s][str(int(line[1]))] = dict({self.p_order[m][i]:line[4+i] for i in xrange(len(self.p_order[m]))})
    #             self.combinaison[-1].append(s+"_"+str(int(line[1])))
    #     n_core = 5
    #     pool = Pool(n_core)
    #     self.combinaison = map(lambda x:x[0:2], self.combinaison)
                
    #     ite = chunked(product(*self.combinaison), np.prod(map(len, self.combinaison))/n_core)
    #     print np.prod(map(len, self.combinaison))/n_core
    #     self.m = m
    #     sys.exit()
    #     self.comb_rank = pool.map(unwrap_self_multi_agregate, zip([self]*n_core, ite))

    #     # for combi in product(*self.combinaison):
    #     #     print combi
    #     #     for ss in combi:
    #     #         s,solution=ss.split("_")
    #     #         self.final[m][s] = self.all_front[m][s][solution]
    #     #     self.quickTest(m, plot=False)
    #     #     self.comb_rank.append([combi, self.JSD(m), self.Pearson(m)])

    # def multi_agregate(self, iterator):
    #     front1 = [] 
    #     front2 = []
    #     comb_rank = []
    #     for combi in iterator:            
    #         for ss in combi:
    #             s, solution = ss.split("_")
    #             self.final[self.m][s] = self.all_front[self.m][s][solution]
    #         self.quickTest(self.m, plot=False)
    #         comb_rank.append((combi, self.JSD(self.m), self.Pearson(self.m)))            
    #     return comb_rank

    # def JSD(self, model):
    #         np.seterr(all='ignore')
    #         tmp = np.dstack((self.human.responses['fmri'], self.models[model].responses))
    #         p = np.mean(tmp, 0)
    #         m = np.mean(p,1)
    #         P = np.dstack((p,1-p))
    #         M = np.transpose(np.vstack((m,1-m)))
    #         kld = np.vstack((np.sum(P[:,0]*np.log2(P[:,0]/M),1), np.sum(P[:,1]*np.log2(P[:,1]/M),1)))
    #         kld[np.isnan(kld)] = 1.0
    #         return np.sum(1-np.mean(kld, 0))
        # def Pearson(self, model):
        # return np.sum(np.array(map(pearsonr, self.human.reaction['fmri'], self.models[model].reaction))[:,0])
