#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test a dictionnary of parameters for each subject

run subjectTest.py -i data

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
from Selection import *
from matplotlib import *
from pylab import *
import pickle
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import leastsq
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------


# -----------------------------------
# FONCTIONS
# -----------------------------------
def _convertStimulus(s):
        return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y : (y - fitfunc(p, x))

def leastSquares(x, y):
    for i in xrange(len(x)):
        pinit = [1.0, -1.0]
        p = leastsq(errfunc, pinit, args = (x[i], y[i]), full_output = False)
        x[i] = fitfunc(p[0], x[i])
    return x    

def center(x):
    #x = x-np.mean(x)
    #x = x/np.std(x)
    x = x-np.median(x)
    x = x/(np.percentile(x, 75)-np.percentile(x, 25))
    return x

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 4
nb_trials = 39
nb_repeat = 2
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions),
                "mixture":CSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("parameters.pickle", 'r') as f:
  p_test = pickle.load(f)

colors = {'owa':'r','distance':'b','tche':'g'}

data = dict()


for o in p_test.iterkeys():

    mean_inferences = []
    super_state = []
    super_action = []
    super_response = []
    super_inf = []
    for s in p_test[o].iterkeys():    
        m = p_test[o][s].keys()[0]

        print "Testing "+s+" with "+m+" selected by "+o
        models[m].setAllParameters(p_test[o][s][m])
        models[m].startExp()
        for k in xrange(nb_repeat):
            for i in xrange(nb_blocs):
                cats.reinitialize()
                cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
                models[m].startBloc()
                for j in xrange(nb_trials):
                    state = cats.getStimulus(j)
                    action = models[m].chooseAction(state)
                    reward = cats.getOutcome(state, action)
                    models[m].updateValue(reward)                            
        super_state.append(convertStimulus(np.array(models[m].state)))
        super_action.append(np.array(models[m].action))
        super_response.append(np.array(models[m].responses))
        super_inf.append(np.array(models[m].pdf))            

    super_state = np.array(super_state).reshape(len(p_test[o].keys())*nb_repeat*nb_blocs,nb_trials)
    super_action = np.array(super_action).reshape(len(p_test[o].keys())*nb_repeat*nb_blocs,nb_trials)
    super_response = np.array(super_response).reshape(len(p_test[o].keys())*nb_repeat*nb_blocs,nb_trials)
    super_inf = np.array(super_inf).reshape(len(p_test[o].keys())*nb_repeat*nb_blocs,nb_trials)
    data[o] = extractStimulusPresentation(super_inf, super_state, super_action, super_response)
    

fig = figure()

ind = np.arange(10)
width = 0.3

for i in xrange(3):
    o = data.keys()[i]
    ax = subplot(3,1,i+1)
    for j in xrange(3):
        ax.bar(np.arange(len(data[o]['mean'][j]))+j*width, data[o]['mean'][j], width, yerr = data[o]['sem'][j])  


# for i in xrange(3):
#     ax = subplot(3,1,i+1)
#     # ax.bar(ind+i*width, data[o]['mean'], width, yerr = data[o]['var'], color = colors[o])
#     for o in data.keys():
#         ax.errorbar(range(len(data[o]['mean'][i])), data[o]['mean'][i], data[o]['sem'][i])

show()



# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_model.pickle") , 'wb') as handle:    
#      pickle.dump(super_data, handle)


# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rts_all_subjects.pickle") , 'wb') as handle:    
#      pickle.dump(super_rt, handle)
