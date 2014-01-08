#!/usr/bin/python
# encoding: utf-8
"""
paretoExp.py

load and explore pareto frontier for one model


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
def loadData(m):
    model_in_folders = os.listdir(options.input)
    if len(model_in_folders) == 0:
        sys.exit("No model found in directory "+options.input)
    data = dict()
    if m in model_in_folders:
        list_subject = os.listdir(options.input+"/"+m)
        for s in list_subject:
            k = s.split("_")[-1].split(".")[0]
            data[k] = np.genfromtxt(options.input+"/"+m+"/"+s)
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

def rescaling(order, scale):
    for s in data.keys():
        for p in order:        
            data[s][:,order.index(p)+4] = scale[0][order.index(p)]+data[s][:,order.index(p)+4]*scale[1][order.index(p)]        
    
def rankSolution(m, good):
    pareto = dict()
    for s in data.iterkeys():
        pareto[s] = data[s][data[s][:,0] == np.max(data[s][:,0])]
        tmp = np.tile(np.array([good[p] for p in p_order[m]]), (len(pareto[s]), 1))
        #score = np.vstack(np.sum(np.power(pareto[s][:,4:]-tmp,2),1))
        #pareto[s] = np.hstack((pareto[s], score))
        pareto[s] = pareto[s][np.argsort(pareto[s][:,4])]
    return pareto

def plot_both(order, pareto, good):
    tmp = np.array([pareto[s][-1][4:] for s in pareto.keys()])
    front = np.array([pareto[s][-1][2:4] for s in pareto.keys()])
    figure()
    for p in order:
        subplot(4,2,order.index(p)+1)
        scatter(np.arange(len(tmp)), tmp[:,order.index(p)])
        axhline(good[p], 0, 1, linewidth = 4, color = 'black')
        title(p)
    show()
    return tmp, front

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
m = 'fusion'
data = loadData(m)

# -----------------------------------
# PLOTTING
# -----------------------------------
p_scale = dict({'qlearning':([0.0,1.0,0.0],[1.0,10.0,1.0]),
                'bayesian':([5.0,0.0,0.0],[15.0,0.01,2.3]),
                'fusion':([0.0,1.0,0.0,0.0,5.0,0.0,0.0],[1.0,6.0,1.0,0.01,15.0,40.0,40.0]),
                'keramati':([0.0,1.0,0.0001,5.0,0.0,0.0,0.0],[1.0,10.0,0.001,20.0,2.3,0.01,1.0])})
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
                                "alpha":[0.0, 2.0]},
                    'keramati':{"gamma":[0.0, 1.0],
                                "beta":[1.0, 10.0],
                                "eta":[0.00001, 0.001],
                                "length":[5, 20],
                                "threshold":[0.0, 2.3], 
                                "noise":[0.0, 0.01],
                                "sigma":[0.0,1.0]}})

p_order = dict({'fusion':['alpha','beta','gamma','noise','length','threshold','gain'],
                'qlearning':['alpha','beta','gamma'],
                'bayesian':['length','noise','threshold'],
                'keramati':['gamma','beta','eta','length','threshold','noise','sigma']})

model_params = {'colors':{'bayesian':'red',
                          'fusion':'green',
                          'qlearning':'blue',
                          'keramati':'grey'}}
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
good = dict({'alpha': 0.8,
             'beta': 3.0,
             'gain': 2.0,
             'gamma': 0.4,
             'length': 10,
             'noise': 0.0001,
             'threshold': 4.0})

rescaling(p_order[m], p_scale[m])
pareto = rankSolution(m, good)
best, front = plot_both(p_order[m], pareto, good)

# ----------------------------------
# Espaces des solutions
# ----------------------------------

# # -----------------------------
# # TEsting solution
# # ------------------------------
p_final = dict({m:dict()})

for s in pareto.keys():
    p_final[m][s] = dict()
    for p in p_order[m]:
        p_final[m][s][p] = np.round(best[pareto.keys().index(s),p_order[m].index(p)], 3)

if options.output:
    writing(p_final)


# os.system("python subjectTest.py -i "+options.output+" -o test_pareto_front.pickle")
