#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

Grid-search for a mixture of Kalman and Bayesian Model
Kalman : beta, gamma
Bayesian : length, noise

WARNING : target function is bounded between the two models.

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization
from scipy.stats import norm
import datetime
import time
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def computeSpecialSingleCorrelation(human_vector, model1_vector, model2_vector):
    h = len(human_vector)
    m1 = len(model1_vector)
    m2 = len(model2_vector)
    ph = float(np.sum(human_vector == 1))/float(h)
    pm1 = float(np.sum(model1_vector == 1))/float(m1)
    pm2 = float(np.sum(model2_vector == 1))/float(m2)
    w = omegaFunc(ph, pm1, pm2)
    pm = w*pm1+(1-w)*pm2
    p = np.mean([pm, ph])
    if ph == pm:
        return 1.0
    elif pm == 0.0:
        z = (np.abs(ph-pm))/(np.sqrt(p*(1-p)*((1/float(h)))))
        return 1-(norm.cdf(z, 0, 1)-norm.cdf(-z, 0, 1))                
    else:
        z = (np.abs(ph-pm))/(np.sqrt(p*(1-p)*((1/float(h))+(1/float(m1)))))
        return 1-(norm.cdf(z, 0, 1)-norm.cdf(-z, 0, 1))


def omegaFunc(cible, freq1, freq2):
    print cible, freq1, freq2
    if cible == freq1 == freq2 == 0:
        return 0.5
    elif cible == freq1 == freq2 == 1:
        return 0.5
    else:
        w = float((cible-freq2)/(freq1-freq2))
        return w
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.9     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.7      
noise_width = 0.01

length_memory = 15

nb_trials = human.responses['meg'].shape[1]

nb_blocs = 46

cats = CATS()

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, 15, 0.01, 1.0)})

inter = 10
# -----------------------------------


xs = {}
ys = {}
label = {}
# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------
opt = Optimization(human, cats, nb_trials, nb_blocs)

data = np.zeros((inter, inter, inter, inter))
values = dict()
fall = dict()
fall['meg'] = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
for m in models.iterkeys():
    p = models[m].getAllParameters()
    for k in p.keys():
        values[k] = np.linspace(p[k][0], p[k][2], inter)
        
count = 0
for i in xrange(len(values['beta'])):
    for j in xrange(len(values['gamma'])):
        for k in xrange(len(values['lenght'])):
            for l in xrange(len(values['noise'])):
                time.sleep(2)
                count+=1; print str(count)+" | "+str(inter**4)
                models['kalman'].beta = values['beta'][i]
                models['kalman'].gamma = values['gamma'][j]
                models['bmw'].lenght_memory = values['lenght'][k]
                models['bmw'].noise = values['noise'][l]
                for m in models.keys():
                    opt.testModel(models[m])
                    models[m].state = convertStimulus(np.array(models[m].state))
                    models[m].action = convertAction(np.array(models[m].action))
                    models[m].responses = np.array(models[m].responses)
                    fall[m] = extractStimulusPresentation(models[m].responses, models[m].state, models[m].action, models[m].responses)
                tmp = 0
                for n in xrange(3):
                    for q in xrange(fall['bmw']['mean'].shape[1]):                        
                        if fall['bmw']['mean'][n, q] > fall['meg']['mean'][n, q] > fall['kalman']['mean'][n, q]:
                            tmp+=1
                        elif fall['bmw']['mean'][n, q] < fall['meg']['mean'][n, q] < fall['kalman']['mean'][n, q]:
                            tmp+=1
                        elif fall['bmw']['mean'][n, q] == fall['meg']['mean'][n, q] or fall['kalman']['mean'][n, q] == fall['meg']['mean'][n, q]:
                            tmp+=1
                if tmp == 3*fall['bmw']['mean'].shape[1]:
                    for m in models.iterkeys():
                        fall[m] = extractStimulusPresentation2(models[m].responses, models[m].state, models[m].action, models[m].responses)
                    v = 0.0
                    for n in [1,2,3]:
                        for q in xrange(opt.data_human[n].shape[1]):
                            v+=computeSpecialSingleCorrelation(opt.data_human[n][:,q], fall['bmw'][n][:,q], fall['kalman'][n][:,q])
                    data[i,j,k,l] = v
                else:
                    data[i,j,k,l] = -1
  
#######################  
#VERY UGLY#############
#######################
v = np.array([values['beta'],
              values['gamma'],
              values['lenght'],
              values['noise']])
crap = dict()
sorted_value = np.sort(np.unique(data.flatten()))[::-1]
for i in sorted_value:
    tmp = np.array(np.where(data == i))
    tmp2 = []
    for j, k in zip(xrange(4), ['beta', 'gamma', 'length', 'noise']):
        tmp2.append(v[j][tmp[j]])
    crap[i] = np.array(tmp2)
        
crap['values'] = values

output = open("../../../Dropbox/ISIR/Plot/superdatagrid_"+str(datetime.datetime.now()).replace(" ","_"), 'wb')
pickle.dump(crap, output)

output.close()




