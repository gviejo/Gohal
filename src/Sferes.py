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
import scipy.optimize as optimization

def func(x, a, b):
    return a*x+b

class EA():
    """
    Optimization is made for one subject
    """
    def __init__(self, data, subject, ptr_model):
        self.model = ptr_model
        self.subject = subject
        self.data = data                
        self.rt = np.hstack(np.array([self.data[i]['rt'].flatten() for i in [1,2,3,4]]).flat)
        self.rt_model = None

    def getFitness(self):
        llh = 0.0
        lrs = 0.0
        self.model.startExp()
        for bloc in self.data.iterkeys():
            self.model.startBloc()            
            for trial in self.data[bloc]['sar']:
                values = self.model.computeValue(self.model.states[trial[0]-1])
                llh = llh + np.log(values[trial[1]-1])
                self.model.current_action = trial[1]-1
                self.model.updateValue(trial[2])                                                        
        self.rt_model = np.hstack(np.array(self.model.reaction).flat)

        self.centerReduct()
        self.leastSquares()

        lrs = np.sum(np.power((self.rt_model-self.rt),2))
        #max_llh = -float(len(self.rt_model))*np.log(0.2)
        #max_lrs = float(len(self.rt_model))*2
        return -np.abs(llh), -np.abs(lrs)

    def centerReduct(self):
        self.rt = self.rt-np.mean(self.rt)
        self.rt_model = self.rt_model-np.mean(self.rt_model)
        if np.std(self.rt_model):
            self.rt_model = self.rt_model/np.std(self.rt_model)
        if np.std(self.rt):
            self.rt = self.rt/np.std(self.rt)

    def leastSquares(self):
        ab, v = optimization.curve_fit(func, self.rt_model, self.rt)        
        self.rt_model = func(self.rt_model, ab[0], ab[1])

    def normalizeRT(self):
        for i in self.data.iterkeys():
            for  j in self.data[i]['rt']:
                self.rt.append(j)
        self.rt = np.array(self.rt)
        self.rt = self.rt - np.min(self.rt)
        self.rt = (self.rt/np.max(self.rt)).flatten()
    
    def normalizeRT2(self):
        for i in self.data.iterkeys():
            self.data[i]['rt'] = self.data[i]['rt']-np.min(self.data[i]['rt'])
            self.data[i]['rt'] = self.data[i]['rt']/np.max(self.data[i]['rt'])
            for j in self.data[i]['rt']:
                self.rt.append(j)
        self.rt = np.array(self.rt)



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
        self.models = dict({"fusion":FSelection(self.states, self.actions, {'alpha':0.0,'beta':0.0,'gamma':0.0,'length':0.0,'noise':0.0,'threshold':0.0,'gain':0.0}),
                            "qlearning":QLearning(self.states, self.actions, {'alpha':0.0, 'beta':0.0, 'gamma':0.0}),
                            "bayesian":BayesianWorkingMemory(self.states, self.actions, {'length':0.0, 'noise':0.0, 'threshold':0.0}),
                            "keramati":KSelection(self.states, self.actions,{"gamma":0.0,"beta":1.0,"eta":0.0001,"length":10.0,"threshold":0.0,"noise":0.0,"sigma":0.0})})
        self.p_order = dict({'fusion':['alpha','beta','gamma','noise','length','threshold','gain'],
                            'qlearning':['alpha','beta','gamma'],
                            'bayesian':['length','noise','threshold'],
                            'keramati':['gamma','beta','eta','length','threshold','noise','sigma']})
        self.good = dict({'fusion':{'alpha': 0.8,
                                     'beta': 3.0,
                                     'gain': 2.0,
                                     'gamma': 0.4,
                                     'length': 10,
                                     'noise': 0.0001,
                                     'threshold': 4.0}})
        self.opt = dict()
        self.pareto = dict()
        self.rank = dict()
        self.final = dict()
        self.loadData()

    def loadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)
        for m in model_in_folders:
            self.data[m] = dict()
            list_subject = os.listdir(self.directory+"/"+m)
            order = self.p_order[m]
            scale = self.models[m].bounds
            for s in list_subject:
                k = s.split("_")[-1].split(".")[0]
                self.data[m][k] = np.genfromtxt(self.directory+"/"+m+"/"+s)
                for p in order:
                    self.data[m][k][:,order.index(p)+4] = scale[p][0]+self.data[m][k][:,order.index(p)+4]*scale[p][1]

    def OWA(self, value, w):
        m,n=value.shape
        assert m>=n
        assert len(w) == n
        assert np.sum(w) == 1
        return np.sum(np.sort(value)*w,1)

    def Tchebychev(self, value, ideal, nadir, lambdaa, epsilon):
        m,n = value.shape
        assert m>=n
        assert len(lambdaa) == n
        assert np.sum(lambdaa) == 1
        assert epsilon < 1.0
        ideal = np.max(value, 0)
        nadir = np.min(value, 0)
        tmp = lambdaa*((ideal-value)/(ideal-nadir))
        return np.max(tmp, 1)+epsilon*np.sum(tmp,1)

    def plotParetoFront(self):
        self.fig_pareto = figure(figsize = (12,9))
        for m in self.data.iterkeys():
            for i in xrange(len(self.data[m].keys())):
                s = self.data[m].keys()[i]
                ax = self.fig_pareto.add_subplot(4,4,i+1)
                ax.scatter(self.pareto[m][s][:,0], self.pareto[m][s][:,1], c=self.rank[m][s])
                ax.plot(self.pareto[m][s][np.argmin(self.rank[m][s]), 0], self.pareto[m][s][np.argmin(self.rank[m][s]),1], 'o', markersize = 15, label = m, alpha = 0.8)
                ax.grid()
        rcParams['xtick.labelsize'] = 6
        rcParams['ytick.labelsize'] = 6                
        ax.legend(loc='lower left', bbox_to_anchor=(1.15, 0.2), fancybox=True, shadow=True)
        self.fig_pareto.show()

    def rankFront(self, w):
        for m in self.data.iterkeys():
            self.opt[m] = dict()
            self.pareto[m] = dict()
            self.rank[m] = dict()
            self.final[m] = dict()
            for s in self.data[m].iterkeys():
                gen = self.data[m][s][:,0]                
                values = self.data[m][s][:,2:4]
                pareto = self.data[m][s][:,2:4][gen == np.max(gen)]
                possible = self.data[m][s][:,4:][gen == np.max(gen)]
                ideal = np.max(pareto, 0)
                nadir = np.min(pareto, 0)
                uniq = np.array(list(set(tuple(r) for r in pareto)))
                #owa = self.OWA(uniq,[0.5,0.5])
                tche = self.Tchebychev(uniq, ideal, nadir, w, 0.01)
                self.opt[m][s] = possible[((pareto[:,0] == uniq[np.argmin(tche)][0])*(pareto[:,1] == uniq[np.argmin(tche)][1]))]
                self.pareto[m][s] = uniq
                self.rank[m][s] = tche            
                self.final[m][s] = dict()
                tmp = np.mean(self.opt[m][s],0)
                for p in self.p_order[m]:
                    self.final[m][s][p] = np.round(tmp[self.p_order[m].index(p)], 3)
            

    def plotSolutions(self):
        self.fig_solution = figure(figsize= (12,9))
        n_params_max = np.max([len(t) for t in [self.p_order[m] for m in self.opt.keys()]])
        n_model = len(self.opt.keys())
        for i in xrange(n_model):
            m = self.opt.keys()[i]
            for j in xrange(len(self.p_order[m])):
                p = self.p_order[m][j]
                ax = self.fig_solution.add_subplot(n_params_max, n_model, i+1+n_model*j)
                for k in xrange(len(self.opt[m].keys())):
                    s = self.opt[m].keys()[k]
                    ax.scatter(self.opt[m][s][:,j],np.ones(len(self.opt[m][s][:,j]))*(k+1))
                    ax.axvline(self.good[m][p], 0, 1, linewidth = 2)
                ax.set_xlim(self.models[m].bounds[p][0],self.models[m].bounds[p][1])
                ax.set_xlabel(p)
        rcParams['xtick.labelsize'] = 6
        rcParams['ytick.labelsize'] = 6
        self.fig_solution.subplots_adjust(hspace = 0.8, top = 0.98, bottom = 0.1)
        self.fig_solution.show()

    def writeOptimal(self, output=False):
        if output:
            target = open(output, 'w')
            target.write(str(self.final))
            target.close()

    def _convertStimulus(self, s):
        return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'


    def quickTest(self, model_to_test):
        nb_blocs = 4
        nb_trials = self.human.responses['fmri'].shape[1]
        cats = CATS(nb_trials)
        m = model_to_test
        pcr = dict()
        rt = dict()
        model = self.models[m]
        model.startExp()
        for s in self.final[m].iterkeys():
            model.setAllParameters(self.final[m][s])
            for i in xrange(nb_blocs):
                cats.reinitialize()
                cats.stimuli = np.array(map(self._convertStimulus, self.human.subject['fmri'][s][i+1]['sar'][:,0]))
                model.startBloc()
                for j in xrange(len(cats.stimuli)):
                #for j in xrange(nb_trials):
                    state = cats.getStimulus(j)
                    action = model.chooseAction(state)
                    reward = cats.getOutcome(state, action)
                    model.updateValue(reward)
            sys.exit()
            tmp = np.array(model.reaction[-nb_blocs:])
            tmp = tmp-np.mean(tmp)
            if np.std(tmp):
                tmp = tmp/np.std(tmp)
            for i,j in zip(xrange(-nb_blocs, 0), xrange(len(tmp))):
                model.reaction[i] = list(tmp[j])
        model.state = convertStimulus(np.array(model.state))
        model.action = convertAction(np.array(model.action))
        model.responses = np.array(model.responses)
        model.reaction = np.array(model.reaction)
        
        pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
        step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
        rt = computeMeanRepresentativeSteps(step)
        pcr_human = extractStimulusPresentation(self.human.responses['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])
        step, indice = getRepresentativeSteps(self.human.reaction['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])
        rt_human = computeMeanRepresentativeSteps(step) 
        colors = ['blue', 'red', 'green']
        self.fig_quick = figure(figsize=(10,5))
        ax1 = self.fig_quick.add_subplot(1,2,1)
        [ax1.errorbar(range(1, len(pcr['mean'][t])+1), pcr['mean'][t], pcr['sem'][t], linewidth = 1.5, elinewidth = 1.5, capsize = 0.8, linestyle = '-', alpha = 1, color = colors[t]) for t in xrange(3)]
        [ax1.errorbar(range(1, len(pcr_human['mean'][t])+1), pcr_human['mean'][t], pcr_human['sem'][t], linewidth = 2.5, elinewidth = 1.5, capsize = 0.8, linestyle = '--', alpha = 0.7,color = colors[t]) for t in xrange(3)]    
        ax2 = self.fig_quick.add_subplot(1,2,2)
        ax2.errorbar(range(1, len(rt[0])+1), rt[0], rt[1], linewidth = 2.0, elinewidth = 1.5, capsize = 1.0, linestyle = '-', color = 'black', alpha = 1.0)        
        ax2.errorbar(range(1, len(rt_human[0])+1), rt_human[0], rt_human[1], linewidth = 2.5, elinewidth = 2.5, capsize = 1.0, linestyle = '--', color = 'grey', alpha = 0.7)
        show()



