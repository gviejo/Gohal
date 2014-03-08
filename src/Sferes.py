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
        self.rt = np.array([self.data[i]['rt'][0:self.n_trials,0] for i in [1,2,3,4]]).flatten()
        self.rt_model = np.tile(np.arange(int(self.model.parameters['length'])+1), (self.n_trials*self.n_blocs, 1))        
        self.state = np.array([self.data[i]['sar'][0:self.n_trials,0] for i in [1,2,3,4]])
        self.action = np.array([self.data[i]['sar'][0:self.n_trials,1] for i in [1,2,3,4]]).astype(int)
        self.responses = np.array([self.data[i]['sar'][0:self.n_trials,2] for i in [1,2,3,4]])                        
        self.hist, self.edges = np.histogram(self.rt, 100)
        self.position = np.digitize(self.rt, self.edges, True)
        self.d = None
        
    def getFitness(self):
        np.seterr(all = 'ignore')
        self.model.startExp()
        for i in xrange(self.n_blocs):
            self.model.startBloc()
            for j in xrange(self.n_trials):                
                self.model.computeValue(int(self.state[i,j])-1, int(self.action[i,j])-1)                
                self.model.updateValue(self.responses[i,j])

        self.model.sigma = np.array(self.model.sigma)
        self.model.value = np.array(self.model.value)
        self.model.pdf = np.array(self.model.pdf)
        
        tmp = np.log(np.sum(self.model.pdf*self.model.value, 1))        
        tmp[np.isinf(tmp)] = -1000.0
        choice = np.sum(tmp)
                
        self.alignToMedian()        
        self.computeDistance()        
        tmp = np.log(np.sum(self.model.pdf*self.d, 1))
        
        tmp[np.isinf(tmp)] = -1000.0
        rt = np.sum(tmp)

        return choice, rt

    def computeDistance(self):
        sup = self.edges[self.position]
        size_bin = self.edges[1]-self.edges[0]
        sup = np.tile(np.vstack(sup), int(self.model.parameters['length'])+1)
        self.d = norm.cdf(sup, self.rt_model, self.model.sigma)-norm.cdf(sup-size_bin, self.rt_model, self.model.sigma)
        self.d[np.isnan(self.d)] = 0.0


    def alignToMedian(self):
        p = np.sum(self.model.pdf, 0)
        p = p/p.sum()
        wp = []
        tmp = np.cumsum(p)
        f = lambda x: (x-np.sum(tmp<x)*tmp[np.sum(tmp<x)-1]+(np.sum(tmp<x)-1.0)*tmp[np.sum(tmp<x)])/(tmp[np.sum(tmp<x)]-tmp[np.sum(tmp<x)-1])
        for i in [0.25, 0.5, 0.75]:
            if np.min(tmp)>i:
                wp.append(0.0)
            else:
                wp.append(f(i))              

        # h, b = np.histogram(self.rt, self.model.parameters['length']+1)
        # b = b[0:-1]+(b[1:]-b[0:-1])/2.
        # h = h.astype(float)
        # h = h/h.sum()
        # wh = []
        # tmp = np.cumsum(h)
        # f = lambda x: (x*(b[np.sum(tmp<x)]-b[np.sum(tmp<x)-1])-b[np.sum(tmp<x)]*tmp[np.sum(tmp<x)-1]+b[np.sum(tmp<x)-1]*tmp[np.sum(tmp<x)])/(tmp[np.sum(tmp<x)]-tmp[np.sum(tmp<x)-1])
        # for i in [0.25, 0.5, 0.75]:
        #     if np.min(tmp)>i:
        #         wh.append(0.0)
        #     else:
        #         wh.append(f(i))
        
        wh = [np.percentile(self.rt, i) for i in [25, 50, 75]]
        print wp[2]-wp[0]
        print (wh[2]-wh[0])/(wp[2]-wp[0])
        print wp[1]-wh[1]
        if (wp[2]-wp[0]):
             self.rt_model = self.rt_model*((wh[2]-wh[0])/(wp[2]-wp[0]))    
        self.rt_model = self.rt_model-(wp[1]-wh[1])
        

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
                            "selection":KSelection(self.states, self.actions)})
        self.p_order = dict({'fusion':['alpha','beta', 'gamma', 'noise','length','gain','sigma_bwm', 'sigma_ql'],
                            'qlearning':['alpha','beta','gamma', 'sigma'],
                            'bayesian':['length','noise','threshold', 'sigma'],
                            'selection':['gamma','beta','eta','length','threshold','noise','sigma', 'sigma_bwm', 'sigma_ql']})
        self.m_order = ['qlearning', 'bayesian', 'selection', 'fusion']
        self.colors_m = dict({'fusion':'r', 'bayesian':'g', 'qlearning':'grey', 'selection':'b'})
        self.opt = dict()
        self.pareto = dict()
        self.rank = dict()
        self.p_test = dict()
        self.mixed = dict()
        self.beh = dict({'state':[],'action':[],'responses':[],'reaction':[]})
        # self.loadData()        
        self.simpleLoadData()
        self.constructParetoFrontier()        
        self.constructMixedParetoFrontier()

    def loadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)
        self.simpleLoadData()

        # pool = Pool(len(model_in_folders))
        # #tmp = pool.map(unwrap_self_load_data, zip([self]*len(model_in_folders), model_in_folders))
        # tmp = [self.loadPooled(m) for m in model_in_folders]
        # for d in tmp:
        #     self.data[d.keys()[0]] = d[d.keys()[0]]

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
                for t in xrange(len(self.threshold)):
                    self.pareto[m][s] = self.pareto[m][s][self.pareto[m][s][:,3+t] >= self.threshold[t]]

                self.pareto[m][s][:,3] = self.pareto[m][s][:,3] - np.log(self.N)*float(len(self.models[m].bounds.keys()))
                self.pareto[m][s][:,4] = self.pareto[m][s][:,4] - np.log(self.N)*float(len(self.models[m].bounds.keys()))
                
                #self.removeDoublon()

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
            self.rank[s] = self.OWA((self.mixed[s][:,4:]), w)            
            i = np.argmax(self.rank[s])
            m = self.m_order[int(self.mixed[s][i,0])]            
            ind = self.pareto[m][s][(self.pareto[m][s][:,0] == self.mixed[s][i][1])*(self.pareto[m][s][:,1] == self.mixed[s][i][2])*(self.pareto[m][s][:,2] == self.mixed[s][i][3])][0]
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
        fig_par = figure(figsize = (12, 9))
        rcParams['ytick.labelsize'] = 8
        rcParams['xtick.labelsize'] = 8
        for m in self.pareto.iterkeys():
            for i in xrange(len(self.data[m].keys())):
                s = self.data[m].keys()[i]
                ax1 = fig_pareto.add_subplot(4,4,i+1)
                ax1.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,4], "-o", color = self.colors_m[m])
                #ax1.scatter(self.pareto[m][s][:,3], self.pareto[m][s][:,4], c = self.rank[m][s])            
        n_params_max = np.max([len(t) for t in [self.p_order[m.split("_")[0]] for m in self.pareto.keys()]])
        n_model = len(self.pareto.keys())
        for i in xrange(n_model):
            m = self.pareto.keys()[i]
            for j in xrange(len(self.p_order[m.split("_")[0]])):
                p = self.p_order[m.split("_")[0]][j]
                ax2 = fig_par.add_subplot(n_params_max, n_model, i+1+n_model*j)                
                for s in self.pareto[m].iterkeys():
                    y, x = np.histogram(self.pareto[m][s][:,5+j])
                    y = y/np.sum(y.astype("float"))
                    x = (x-(x[1]-x[0])/2)[1:]
                    ax2.plot(x, y, 'o-', linewidth = 2)
                    ax2.set_ylim(0, 1)
                    ax2.set_xlim(self.models[m.split("_")[0]].bounds[p][0],self.models[m.split("_")[0]].bounds[p][1])
                    ax2.set_xlabel(p)
                if j == 0:
                    ax2.set_title(m)
        fig_par.subplots_adjust(hspace = 0.8, top = 0.98, bottom = 0.1)
        fig_pareto.subplots_adjust(left = 0.08, wspace = 0.26, hspace = 0.26, right = 0.92, top = 0.96)
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
        x = np.array(self.models[m].reaction).flatten()
        y = np.array([self.human.subject['fmri'][s][i]['rt'][:,0][0:n_trials] for i in xrange(1, n_blocs+1)])                
        p = np.sum(np.array(self.models[m].pdf), 0)
        p = p/np.sum(p)
        tmp = np.cumsum(p)        
        f = lambda x: (x-np.sum(tmp<x)*tmp[np.sum(tmp<x)-1]+(np.sum(tmp<x)-1.0)*tmp[np.sum(tmp<x)])/(tmp[np.sum(tmp<x)]-tmp[np.sum(tmp<x)-1])
        w = []
        for i in [0.25, 0.5, 0.75]:
            if np.min(tmp)>i:
                w.append(0.0)
            else:
                w.append(f(i))        
        if (w[2]-w[0]):
            x = x*((np.percentile(y, 75)-np.percentile(y, 25))/(w[2]-w[0]))
        x = x-(w[1]-np.median(y))
        self.models[m].reaction = x.reshape(n_blocs, n_trials)

    def run(self, plot=True):
        nb_blocs = 4
        nb_trials = self.human.responses['fmri'].shape[1]
        cats = CATS(nb_trials)
        
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
            self.alignToMedian(m, s, nb_blocs, nb_trials)
            reaction = np.random.normal(self.models[m].reaction, np.array(self.models[m].sigma_test))
            for i in xrange(nb_blocs):
                self.beh['state'].append(self.models[m].state[i])
                self.beh['action'].append(self.models[m].action[i])
                self.beh['responses'].append(self.models[m].responses[i])
                self.beh['reaction'].append(list(reaction[i]))
        for k in self.beh.iterkeys():
            self.beh[k] = np.array(self.beh[k])
        self.beh['state'] = convertStimulus(self.beh['state'])
        self.beh['action'] = convertAction(self.beh['action'])
         
    
        if plot:                                                            
            pcr = extractStimulusPresentation(self.beh['responses'], self.beh['state'], self.beh['action'], self.beh['responses'])
            pcr_human = extractStimulusPresentation(self.human.responses['fmri'], self.human.stimulus['fmri'], self.human.action['fmri'], self.human.responses['fmri'])            
                                                            
            step, indice = getRepresentativeSteps(self.beh['reaction'], self.beh['state'], self.beh['action'], self.beh['responses'])
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
            ax3 = ax2.twinx()
            ax3.errorbar(range(1, len(rt_human[0])+1), rt_human[0], rt_human[1], linewidth = 2.5, elinewidth = 2.5, capsize = 1.0, linestyle = '--', color = 'grey', alpha = 0.7)
            show()


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
