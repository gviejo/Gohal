#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
singleModelTest.py

load and test a dictionnary of parameters for Q-learning and bayesian working memory


run singleModelTest.py

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
nb_repeat = 5
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions),
                "mixture":CSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("parameters_single.pickle", 'r') as f:
  p_test = pickle.load(f)


super_data = dict()

for m in p_test.iterkeys():    
    super_data[m] = dict()
    for o in p_test[m].iterkeys():
        super_data[m][o] = dict({'pcr':{},'rt':{}})

        super_state = []
        super_action = []
        super_response = []
        super_rt = []
        super_hrt = []

        for s in p_test[m][o].iterkeys():                
            print "Testing "+s+" with "+m+" selected by "+o
            models[m].setAllParameters(p_test[m][o][s][m])
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

            rtm = np.array(models[m].reaction)
            rt = np.array([human.subject['fmri'][s][i]['rt'][0:nb_trials,0] for i in range(1,nb_blocs+1)])            
            # rt = np.tile(rt, (nb_repeat, 1))                    
            # CENTER
            rtm = center(rtm)
            rt = center(rt)
            super_state.append(convertStimulus(np.array(models[m].state)))
            super_action.append(np.array(models[m].action))
            super_response.append(np.array(models[m].responses))
            super_rt.append(rtm)
            super_hrt.append(rt)
            
        super_state = np.array(super_state).reshape(len(p_test[m][o].keys())*nb_repeat*nb_blocs,nb_trials)
        super_action = np.array(super_action).reshape(len(p_test[m][o].keys())*nb_repeat*nb_blocs,nb_trials)
        super_response = np.array(super_response).reshape(len(p_test[m][o].keys())*nb_repeat*nb_blocs,nb_trials) 
        super_rt = np.array(super_rt).reshape(len(p_test[m][o].keys())*nb_repeat*nb_blocs,nb_trials)

        pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])

        pcr = extractStimulusPresentation(super_response, super_state, super_action, super_response)

        step, indice = getRepresentativeSteps(center(human.reaction['fmri']), human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
        rt_fmri = computeMeanRepresentativeSteps(step) 

        step, indice = getRepresentativeSteps(super_rt, super_state, super_action, super_response)
        rt_model = computeMeanRepresentativeSteps(step)

        super_data[m][o]['pcr']['model'] = pcr
        super_data[m][o]['pcr']['fmri'] = pcr_human
        super_data[m][o]['rt']['model'] = rt_model
        super_data[m][o]['rt']['fmri'] = rt_fmri

        # fig = figure(figsize = (15, 12))
        # colors = ['blue', 'red', 'green']
        # ax1 = fig.add_subplot(1,2,1)
        # for i in xrange(3):
        #     plot(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
        #     errorbar(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], pcr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
        #     plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    
        #     #errorbar(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], pcr_human['sem'][i], linewidth = 2, linestyle = ':', color = colors[i], alpha = 0.6)

        # ax1 = fig.add_subplot(1,2,2)
        # ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, color = 'grey', alpha = 0.5)
        # ax1.errorbar(range(1, len(rt_model[0])+1), rt_model[0], rt_model[1], linewidth = 2, color = 'black', alpha = 0.9)

        # show()



with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_single_model.pickle") , 'wb') as handle:    
     pickle.dump(super_data, handle)


# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rts_all_subjects.pickle") , 'wb') as handle:    
#      pickle.dump(super_rt, handle)
