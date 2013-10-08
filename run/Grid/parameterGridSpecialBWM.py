#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

Grid-search for Kalman and Bayesian Model
Kalman : beta, gamma
Bayesian : length, noise

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization
import datetime
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
def testModel():
    bww.initializeList()
    for i in xrange(nb_blocs):
        cats.reinitialize()
        bww.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)
    bww.state = convertStimulus(np.array(bww.state))
    bww.action = convertAction(np.array(bww.action))
    bww.responses = np.array(bww.responses)


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
noise = 0.001
length_memory = 10
threshold = 1

correlation = "Z"
inter = 10

nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]

cats = CATS()

bww = BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise, threshold)

# -----------------------------------

data = np.zeros((inter, inter, inter))
fall = dict()
values = dict()
# -----------------------------------

# -----------------------------------
# PARAMETERS Testing
# -----------------------------------
opt = Optimization(human, cats, nb_trials, nb_blocs)

tm = 0

p = bww.getAllParameters()    
for k in p.keys():
    values[k] = np.linspace(p[k][0], p[k][2], inter)
values['lenght'] = values['lenght'].astype(int)

count = 0
for i in xrange(len(values['lenght'])):
    for j in xrange(len(values['noise'])):
        for k in xrange(len(values['threshold'])):
            count+=1; print str(count)+" | "+str(inter**len(p.keys()))
            bww.lenght_memory = values['lenght'][i]
            bww.noise = values['noise'][j]
            bww.threshold = values['threshold'][k]
            testModel()
            fall = extractStimulusPresentation2(bww.responses, bww.state, bww.action, bww.responses)
            tmp = 0.0
            for n in [1,2,3]:
                if len(fall[n]) != 0:
                    for q in xrange(opt.data_human[n].shape[1]):
                        tmp += computeSingleCorrelation(opt.data_human[n][:,q], fall[n][:,q], "JSD")
            data[i,j,k] = tmp

v = np.array([values['lenght'],
              values['noise'],
              values['threshold']])
crap = dict()
sorted_value = np.sort(np.unique(data.flatten()))[::-1]
for i in sorted_value:
    tmp = np.array(np.where(data == i))
    tmp2 = []
    for j, k in zip(xrange(4), ['lenght','noise','threshold']):
        tmp2.append(v[j][tmp[j]])
    crap[i] = np.array(tmp2)

crap['values'] = values
crap['correlation'] = correlation
output = open("../../../Dropbox/ISIR/Plot/datagrid_bwm_"+str(datetime.datetime.now()).replace(" ", "_"), 'wb')
pickle.dump(crap, output)
output.close()

