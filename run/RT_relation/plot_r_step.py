#!/usr/bin/python
# encoding: utf-8
"""
to plot representative steps for meg and fmri

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *

from matplotlib import *
from pylab import *
from HumanLearning import HLearning

# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (12, 8))
ion()
for case, i in zip(['meg', 'fmri'], [1,2]):    
    state = human.stimulus[case]
    action = human.action[case]
    responses = human.responses[case]
    reaction = human.reaction[case]    
    step, indice = getRepresentativeSteps(reaction, state, action, responses)
    rt = computeMeanRepresentativeSteps(step) 
    step, indice = getRepresentativeSteps(responses, state, action, responses)
    y = computeMeanRepresentativeSteps(step)

    ax1 = subplot(2,2,i)
    ind = np.arange(1, len(rt[0])+1)
    ax1.plot(ind, y[0], linewidth = 2, color = 'blue')
    ax1.errorbar(ind, y[0], y[1], linewidth = 2, color = 'blue')    
    ax2 = ax1.twinx()
    ax2.plot(ind, rt[0], linewidth = 2, color = 'green', linestyle = '--')
    ax2.errorbar(ind, rt[0], rt[1], linewidth = 2, color = 'green', linestyle = '--')
    ax1.grid()
    ax1.set_ylabel("PCR %")    
    ax2.set_ylabel("Reaction time (s)")
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax1.set_xticks(range(2, 15, 2))
    ax1.set_xlim(0, 15)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(case)
    if case == 'meg':
        ax2.set_yticks([0.46, 0.50, 0.54])
        ax2.set_ylim(0.43, 0.56)
    elif case == 'fmri':
        ax2.set_yticks([0.68, 0.72, 0.76, 0.80])

    

# -----------------------------------

for case, p in zip(['meg', 'fmri'], [3,4]):
    state = human.stimulus[case]
    action = human.action[case]
    responses = human.responses[case]
    reaction = human.reaction[case]    
    step, indice = getRepresentativeSteps(reaction, state, action, responses)
    m,n = state.shape
    state_ = [[],[],[]]
    action_ = [[],[],[]]
    responses_ = [[],[],[]]
    reaction_ = [[],[],[]]
    indice_ = [[],[],[]]
    size = []    
    for j in xrange(m):
        order = searchStimOrder(state[j], action[j], responses[j])   
        for k in xrange(3):
            ind = state[j] == order[k]
            size.append(np.sum(ind))
            state_[k].append(state[j][ind])
            action_[k].append(action[j][ind])
            responses_[k].append(responses[j][ind])
            reaction_[k].append(reaction[j][ind])      
            indice_[k].append(indice[j][ind])
    smin = np.min(size)
    for i in xrange(3):
        for j in xrange(len(state_[i])):
            state_[i][j] = state_[i][j][:smin]
            action_[i][j] = action_[i][j][:smin]
            responses_[i][j] = responses_[i][j][:smin]
            reaction_[i][j] = reaction_[i][j][:smin]
            indice_[i][j] = indice_[i][j][:smin]
    state_ = np.array(state_)
    action_ = np.array(action_)
    responses_ = np.array(responses_)
    reaction_ = np.array(reaction_)
    indice_ = np.array(indice_)
    mean_reaction_time = []
    for i in xrange(3):
        ind = np.unique(indice_[i])
        mean_reaction_time.append(np.zeros((3, len(ind))))
        mean_reaction_time[i][0] = ind
        for j in xrange(len(ind)):
            mean_reaction_time[i][1, j] = np.mean(reaction_[i][np.where(indice_[i] == ind[j])])
            mean_reaction_time[i][2, j] = np.var(reaction_[i][np.where(indice_[i] == ind[j])])

    ax1 = subplot(2,2,p)
    for rt in mean_reaction_time:        
        ax1.plot(rt[0], rt[1], 'o-', linewidth = 2)
        ax1.errorbar(rt[0], rt[1], rt[2], linewidth = 2, linestyle = 'o-')
        ax1.grid()        
        ax1.set_ylabel("Reaction time (s)")        
        ax1.set_xticks(range(2, 15, 2))        
        ax1.set_title(case)
        ax1.set_xlim(0, 15)
        
            

show()