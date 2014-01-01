#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and and plot multi objective results from Sferes 2 optimisation 


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
import matplotlib.pyplot as plt
from time import time

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test \n If none is provided, all files are loaded", default=False)
parser.add_option("-o", "--output", action="store", help="The output file of best parameters to test", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def loadData():
    model_in_folders = os.listdir(options.input)
    if len(model_in_folders) == 0:
        sys.exit("No model found in directory "+options.input)
    data = dict()
    for m in model_in_folders:
        data[m] = dict()
        list_subject = os.listdir(options.input+"/"+m)
        for s in list_subject:
            k = s.split("_")[-1].split(".")[0]
            data[m][k] = np.genfromtxt(options.input+"/"+m+"/"+s)
    if options.model and options.model in model_in_folders:
        return dict({options.model:data[options.model]})
    elif options.model and options not in model_in_folders:
        sys.exit("Model "+options.model+" is not found in directory "+options.input)
    else:
        return data
  
def OWA(value, w):
    # return utility 
    m,n = value.shape
    assert m>=n
    assert len(w) == n
    assert np.sum(w) == 1
    return np.sum(np.sort(value)*w, 1)

def Tchebychev(value, lambdaa, epsilon):
    m,n = value.shape
    assert m>=n
    assert len(lambdaa) == n
    assert np.sum(lambdaa) == 1
    assert epsilon < 1.0
    ideal = np.max(value, 0)
    nadir = np.min(value, 0)
    tmp = lambdaa*((ideal-value)/(ideal-nadir))
    return np.max(tmp, 1)+epsilon*np.sum(tmp,1)

def writing(data_):
    target = open(options.output, 'w')
    target.write(str(data_))
    target.close()


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# LOADING DATA
# -----------------------------------
data = loadData()

# -----------------------------------
# PLOTTING
# -----------------------------------
p_scale = dict({'qlearning':([0.0,1.0,0.0],[1.0,10.0,1.0]),
                'bayesian':([5.0,0.0,0.0],[15.0,0.01,2.3]),
                'fusion':([0.0,1.0,0.0,0.0,5.0,0.0,0.0],[1.0,6.0,1.0,0.01,15.0,40.0,40.0]),
                'keramati':()})
p_bounds = dict({'fusion':{"gamma":[0.0, 1.0],
                                "beta":[1.0, 6.0],
                                "alpha":[0.0, 2.0],
                                "length":[5, 20],
                                "threshold":[0.0, 40.0], 
                                "noise":[0.0, 0.01],
                                "gain":[0.0,40.0]},
                    'bayesian':{"length":[5, 20], 
                                "threshold":[0.0, 2.3], 
                                "noise":[0.0, 0.1]},
                    'qlearning':{"gamma":[0.0, 1.0],
                                "beta":[1.0, 10.0],
                                "alpha":[0.0, 2.0]}})

p_order = dict({'fusion':['alpha','beta','gamma','noise','length','threshold','gain'],
                'qlearning':['alpha','beta','gamma'],
                'bayesian':['length','noise','threshold'],
                'keramati':[]})
model_params = {'colors':{'bayesian':'red',
                          'fusion':'green',
                          'qlearning':'blue'}}
#ion()
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          

# -----------------------------------
# Espace des critères 
# -----------------------------------
ind_opt = dict()
fig1 = figure(figsize = (12, 9))
ion()
for m in data.iterkeys():
    ind_opt[m] = dict()
    for i in xrange(len(data[m].keys())):
        s = data[m].keys()[i]
        gen = data[m][s][:,0]
        ind = data[m][s][:,1] 
        values = data[m][s][:,2:4]
        pareto = data[m][s][:,2:4][gen == np.max(gen)]
        possible = data[m][s][:,4:][gen == np.max(gen)]
        ideal = np.max(pareto, 0)
        nadir = np.min(pareto, 0)
        ax1 = fig1.add_subplot(4,4,i+1)
        # PARETO FRONT
        uniq = np.array(list(set(tuple(r) for r in pareto)))        
        owa = OWA(uniq, [0.5, 0.5])
        ########################################
        tche = Tchebychev(uniq, [0.5, 0.5], 0.01)
        ########################################        
        ax1.scatter(uniq[:,0], uniq[:,1], c = tche)        
        #ax1.plot(ideal[0], ideal[1], 'o', markersize = 5, color = model_params['colors'][m])
        #ax1.plot(nadir[0], nadir[1], 'o', markersize = 5, color = model_params['colors'][m])
        #ax1.plot([nadir[0], ideal[0]], [nadir[1], ideal[1]], '--', linewidth = 3.0, color = model_params['colors'][m])
        ax1.plot(uniq[np.argmin(tche),0], uniq[np.argmin(tche),1], 'o', markersize = 15, color = model_params['colors'][m], label = m, alpha = 0.8)        
        # ax1.plot(uniq[np.argmax(owa),0], uniq[np.argmax(owa),1], 'o', markersize = 15)
        #xlim(0, -human.length['fmri'][s]*np.log(0.2))
        #ylim(0, human.length['fmri'][s])
        ax1.grid()
        # SAVING Optimal solutions        
        ind_opt[m][s] = possible[((pareto[:,0] == uniq[np.argmin(tche)][0])*(pareto[:,1] == uniq[np.argmin(tche)][1]))]
        
ax1.legend(loc='lower left', bbox_to_anchor=(1.15, 0.2), fancybox=True, shadow=True)
fig1.show()

# ----------------------------------
# Espaces des solutions
# ----------------------------------
solutions = dict()
for m in data.iterkeys():
    solutions[m] = dict()
    for s in data[m].iterkeys():
        solutions[m][s] = dict()
        l,n = ind_opt[m][s].shape
        assert n == len(p_scale[m][0]) == len(p_scale[m][1]) == len(p_order[m])
        solutions[m][s] = p_scale[m][0]+ind_opt[m][s]*p_scale[m][1]


fig2 = figure(figsize = (12, 9))
#rc.update(params)
n_params_max = np.max([len(t) for t in [p_order[m] for m in solutions]])
n_model = len(solutions.keys())
            
for i in xrange(n_model):
    m = solutions.keys()[i]
    for j in xrange(len(p_order[m])):
        p = p_order[m][j]
        subplot(n_params_max, n_model, i+1+n_model*j)
        for k in xrange(len(solutions[m].keys())):
            s = solutions[m].keys()[k]
            scatter(solutions[m][s][:,j], np.ones(len(solutions[m][s][:,j]))*(k+1))
        xlim(p_bounds[m][p][0], p_bounds[m][p][1])
fig2.show()            



# ---------------------------------
# Writing optimal solution 
# ---------------------------------

p_final = dict()

for m in solutions:
    p_final[m] = dict()
    for s in solutions[m]:
        p_final[m][s] = dict()
        tmp = np.mean(solutions[m][s],0)
        for i in xrange(len(p_order[m])):
            p = p_order[m][i]
            p_final[m][s][p] = np.round(tmp[i], 3)

if options.output:
    writing(p_final)


# # -----------------------------
# # TEsting solution
# # ------------------------------

# os.system("python subjectTest.py -i "+options.output+" -o sferes_fmri.pickle")
# os.system("python plot_test.py -i sferes_fmri.pickle")
# os.system("evince test.pdf")