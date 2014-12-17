#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test choice only extremum of pareto fronts

run subjectTestExtremum.py

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
nb_repeat = 10
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions),
                "mixture":CSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("extremum.pickle", 'r') as f:
    p_test = pickle.load(f)

# with open(os.path.expanduser("obj_choice.pickle"), 'r') as f:
#     best = pickle.load(f)

colors_m = dict({'fusion':'#F1433F',
                'bayesian': '#D5A253',
                'qlearning': '#6E8243',
                'selection':'#70B7BA',
                'mixture':'#3D4C53'})

entropy = {'Hb':{},'Hf':{}}

pcrm = dict({'s':[], 'a':[], 'r':[]})

# for m in best.iterkeys():
for m in ['qlearning']:
    # for s in best[m].iterkeys():                
    for s in p_test.keys():
        print "Testing "+s+" with "+m
        models[m].setAllParameters(p_test[s][m])
        models[m].startExp()
        for k in xrange(nb_repeat):
            for i in xrange(nb_blocs):
                cats.reinitialize()
                cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
                models[m].startBloc()
                for j in xrange(nb_trials):
                    state = cats.getStimulus(j)
                    action = models[m].chooseAction(state)
                    reward = cats.getOutcome(state, action, case='fmri')                    
                    models[m].updateValue(reward)            
                    # print state, action, reward
                    # print cats.incorrect
                    # sys.stdin.readline()        
                # if s == 'S2':
                #     sys.exit()
        state = convertStimulus(np.array(models[m].state))
        action = np.array(models[m].action)
        responses = np.array(models[m].responses)        
        pcrm['s'].append(state)
        pcrm['a'].append(action)
        pcrm['r'].append(responses)               
        hall = np.array(models[m].Hall)
        if hall[:,:,0].sum():
            entropy['Hb'][s] = {m:extractStimulusPresentation(hall[:,:,0], state, action, responses)}

        if hall[:,:,1].sum():
            entropy['Hf'][s] = {m:extractStimulusPresentation(hall[:,:,1], state, action, responses)}
        
        
        
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
for i in pcrm.iterkeys():    
    pcrm[i] = np.array(pcrm[i])
    pcrm[i] = np.reshape(pcrm[i], (pcrm[i].shape[0]*pcrm[i].shape[1], pcrm[i].shape[2]))
pcr = extractStimulusPresentation(pcrm['r'], pcrm['s'], pcrm['a'], pcrm['r'])    


meanHall = dict()
for h in entropy.keys():
    meanHall[h] = dict()
    model = np.unique([entropy[h][s].keys()[0] for s in entropy[h].iterkeys()])
    for m in model:
        subject = [s for s in entropy[h].keys() if entropy[h][s].keys()[0] == m]
        if len(subject) == 1:
            meanHall[h][m] = entropy[h][subject[0]][m]
        else:
            tmp = np.array([entropy[h][s][m]['mean'] for s in subject])
            meanHall[h][m] = {'mean':np.mean(tmp,0), 'sem':sem(tmp,0)}


#SAVING DATA
data = dict()
data['pcr'] = dict({'model':pcr,'fmri':pcr_human})    
data['Hb'] = meanHall['Hb']
data['Hf'] = meanHall['Hf']

fig = figure(figsize = (9,5))
# fig = figure()
colors = ['blue', 'red', 'green']
ax1 = fig.add_subplot(1,3,1)
for i in xrange(3):
    plot(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], pcr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    
    #errorbar(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], pcr_human['sem'][i], linewidth = 2, linestyle = ':', color = colors[i], alpha = 0.6)

ax2 = fig.add_subplot(1,3,2)
# for s in entropy['Hb'].iterkeys():
    # m = entropy['Hb'][s].keys()[0]
    # tmp = entropy['Hb'][s][m]
for m in meanHall['Hb'].iterkeys():
    tmp = meanHall['Hb'][m]    
    for i in xrange(3):
        x = range(1, len(tmp['mean'][i])+1)
        y = tmp['mean'][i]
        ax2.plot(x, y, linewidth=1.5, color = colors_m[m])        
        ax2.fill_between(x, y-tmp['sem'][i], y+tmp['sem'][i], facecolor = colors_m[m], alpha = 0.5)
ax3 = fig.add_subplot(1,3,3)
# for s in entropy['Hf'].iterkeys():
#     m = entropy['Hf'][s].keys()[0]
#     tmp = entropy['Hf'][s][m]
for m in meanHall['Hf'].iterkeys():
    tmp = meanHall['Hf'][m]
    for i in xrange(3):
        x = range(1, len(tmp['mean'][i])+1)
        y = tmp['mean'][i]
        ax3.plot(x, y, linewidth=1.5, color = colors_m[m])        
        ax3.fill_between(x, y-tmp['sem'][i], y+tmp['sem'][i], facecolor = colors_m[m], alpha = 0.5)

show()

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_choice_only.pickle") , 'wb') as handle:    
#      pickle.dump(data, handle)
