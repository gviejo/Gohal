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

def mutualInformation(x, y):
    np.seterr('ignore')
    bin_size = 2*(np.percentile(y, 75)-np.percentile(y, 25))*np.power(len(y), -(1/3.))
    py, edges = np.histogram(y, bins=np.arange(y.min(), y.max()+bin_size, bin_size))
    py = py/float(py.sum())
    yp = np.digitize(y, edges)-1
    px, edges = np.histogram(x, bins = np.linspace(x.min(), x.max()+0.00001, 15))
    px = px/float(px.sum())
    xp = np.digitize(x, edges)-1
    p = np.zeros((len(py), len(px)))
    for i in xrange(len(yp)): p[yp[i], xp[i]] += 1
    p = p/float(p.sum())
    tmp = np.log2(p/np.outer(py, px))
    tmp[np.isinf(tmp)] = 0.0
    tmp[np.isnan(tmp)] = 0.0
    return (np.sum(p*tmp), p)

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
                "selection":KSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("parameters.pickle", 'r') as f:
  p_test = pickle.load(f)


hrt = []
hrtm = []
mi = []
pmi = []
pcrm = dict({'s':[], 'a':[], 'r':[], 't':[]})
rt_all = []
rtm_all = []

for s in p_test.iterkeys():    
    m = p_test[s].keys()[0]
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
                reward = cats.getOutcome(state, action)
                models[m].updateValue(reward)    
    #MUTUAL Information
    rtm = np.array(models[m].reaction)
    rt = np.array([human.subject['fmri'][s][i]['rt'][0:nb_trials,0] for i in range(1,nb_blocs+1)])
    rt_all.append(rt.flatten())    
    rtm_all.append(rtm[0:nb_blocs].flatten())
    rt = np.tile(rt, (nb_repeat, 1))
    v, p = mutualInformation(rtm.flatten(), rt.flatten())
    mi.append(v)
    pmi.append(p)
    
    # CENTER
    rtm = center(rtm)    
    state = convertStimulus(np.array(models[m].state))
    action = np.array(models[m].action)
    responses = np.array(models[m].responses)
    step, indice = getRepresentativeSteps(rtm, state, action, responses)
    hrtm.append(computeMeanRepresentativeSteps(step)[0])
    pcrm['s'].append(state)
    pcrm['a'].append(action)
    pcrm['r'].append(responses)
    pcrm['t'].append(rtm)
  
    rt = center(rt)
    action = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,1] for i in range(1,nb_blocs+1)])
    responses = np.array([human.subject['fmri'][s][i]['sar'][0:nb_trials,2] for i in range(1,nb_blocs+1)])
    action = np.tile(action, (nb_repeat, 1))
    responses = np.tile(responses, (nb_repeat, 1))
    step, indice2 = getRepresentativeSteps(rt, state, action, responses)
    hrt.append(computeMeanRepresentativeSteps(step)[0])
    
rt_all = np.array(rt_all)
rtm_all = np.array(rtm_all)
mi = np.array(mi)
pmi = np.array(pmi)
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
for i in pcrm.iterkeys():
    pcrm[i] = np.array(pcrm[i])
    pcrm[i] = np.reshape(pcrm[i], (pcrm[i].shape[0]*pcrm[i].shape[1], pcrm[i].shape[2]))
pcr = extractStimulusPresentation(pcrm['r'], pcrm['s'], pcrm['a'], pcrm['r'])

ht = np.reshape(human.reaction['fmri'], (14, 4*39))
ht = np.array(map(center, ht)).reshape(14*4, 39)
step, indice = getRepresentativeSteps(ht, human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 

step, indice = getRepresentativeSteps(pcrm['t'], pcrm['s'], pcrm['a'], pcrm['r'])
rt = computeMeanRepresentativeSteps(step)



fig = figure(figsize = (15, 12))
colors = ['blue', 'red', 'green']
ax1 = fig.add_subplot(4,4,1)
for i in xrange(3):
    plot(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], pcr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    
    #errorbar(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], pcr_human['sem'][i], linewidth = 2, linestyle = ':', color = colors[i], alpha = 0.6)

ax1 = fig.add_subplot(4,4,2)
ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, color = 'grey', alpha = 0.5)
ax1.errorbar(range(1, len(rt[0])+1), rt[0], rt[1], linewidth = 2, color = 'black', alpha = 0.9)

for i, s in zip(xrange(14), p_test.keys()):
  ax1 = fig.add_subplot(4,4,i+3)
  ax1.plot(hrt[i], 'o-')
  #ax2 = ax1.twinx()
  ax1.plot(hrtm[i], 'o--', color = 'green')
  ax1.set_title(s+" "+p_test[s].keys()[0])

#show()
fig2 = figure(figsize = (15, 12))
for i, s in zip(xrange(14), p_test.keys()):
     subplot(4,4,i+1)
     imshow(pmi[i], origin = 'lower', interpolation = 'nearest')
     title(s + " " + str(mi[i]))

ind = np.argsort(rt_all, 1)
fig3 = figure(figsize = (15, 12))
for i, s in zip(xrange(14), p_test.keys()):
    subplot(4,4,i+1)
    plot(rt_all[i][ind[i]])
    plot(rtm_all[i][ind[i]])

show()

# fig.savefig('fig_mean_rt_all_sub_representative_steps.pdf', bbox_inches='tight')

# os.system("evince fig_mean_rt_all_sub_representative_steps.pdf")

