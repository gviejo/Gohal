#!/usr/bin/python
# encoding: utf-8
"""
plot reaction when subject do a mistake at the previous step
vs when subject do a correct responses

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
def testRelation(var1, var2):
    #KS, p = stats.ks_2samp(var1, var2)    
    #KS, p = stats.kruskal(var1, var2)
    KS, p = stats.mannwhitneyu(var1, var2)
    return p

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# ----------------------------------

figure(figsize = (10, 10))
ion()
bar_kwargs = {'linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
data = dict()
for case,p,t in zip(['meg', 'fmri'], [[1,2], [3,4]],[1,2]):
    subplot(3,2,t)
    state = human.stimulus[case]
    action = human.action[case]
    responses = human.responses[case]
    reaction = human.reaction[case]    
    step, indice = getRepresentativeSteps(reaction, state, action, responses)
    correct = []
    incorrect = []
    m,n = indice.shape
    for i in xrange(m):
        for j in xrange(n):
            if responses[i,j-1] == 0:
                incorrect.append(reaction[i,j])
            elif responses[i,j-1] == 1:
                correct.append(reaction[i,j])        
    
    p_value = testRelation(np.array(correct), np.array(incorrect))    

    m = np.array([[np.mean(correct), np.mean(incorrect)],
                  [np.var(correct), np.var(incorrect)]])
    data[case] = m
    nbins = 20
    hist(correct, nbins, color= 'green', **bar_kwargs)
    hist(incorrect, nbins, color ='red', **bar_kwargs)    
    title(case, fontsize = 20)
    grid()
    annotate("p = "+str(np.round(p_value, 4)), xy=(0.75,0.90), xycoords='axes fraction')
    xlabel("Reaction time (s)")
    ylabel("#")

data = dict()
for case,p,t in zip(['meg', 'fmri'], [[1,2], [3,4]],[3,4]):
    subplot(3,2,t)
    state = human.stimulus[case]
    action = human.action[case]
    responses = human.responses[case]
    reaction = human.reaction[case]    
    step, indice = getRepresentativeSteps(reaction, state, action, responses)
    correct = []
    incorrect = []
    m,n = indice.shape
    for i in xrange(m):
        for j in xrange(n):
            if indice[i,j] > 5 and indice[i,j-1] > 5 and responses[i,j-1] == 0:
                incorrect.append(reaction[i,j])
            elif indice[i,j] > 5 and indice[i,j-1] > 5 and responses[i,j-1] == 1:
                correct.append(reaction[i,j])        
    
    p_value = testRelation(np.array(correct), np.array(incorrect))    

    m = np.array([[np.mean(correct), np.mean(incorrect)],
                  [np.var(correct), np.var(incorrect)]])
    data[case] = m
    nbins = 20
    hist(correct, nbins, color= 'green', **bar_kwargs)
    hist(incorrect, nbins, color ='red', **bar_kwargs)        
    grid()
    annotate("p = "+str(np.round(p_value, 4)), xy=(0.75,0.90), xycoords='axes fraction')
    xlabel("Reaction time (s)")
    ylabel("#")

data = dict()
for case,p,t in zip(['meg', 'fmri'], [[1,2], [3,4]],[5,6]):
    subplot(3,2,t)
    state = human.stimulus[case]
    action = human.action[case]
    responses = human.responses[case]
    reaction = human.reaction[case]    
    step, indice = getRepresentativeSteps(reaction, state, action, responses)
    m = np.zeros((2, np.max(indice)))
    v = np.zeros((2, np.max(indice)))
    p_values_ = np.zeros(np.max(indice))
    for k in xrange(6, int(np.max(indice))+1):        
        ind = np.where(indice == k)
        correct = []
        incorrect = []
        for i,j in zip(ind[0], ind[1]):        
            if responses[i,j-1] == 0:
                incorrect.append(reaction[i,j])
            elif responses[i,j-1] == 1:
                correct.append(reaction[i,j])        
        m[0,k-1] = np.mean(correct)
        m[1,k-1] = np.mean(incorrect)
        v[0,k-1] = np.var(correct)
        v[1,k-1] = np.var(incorrect)
        p_values_[k-1] = testRelation(np.array(correct), np.array(incorrect))
    
    width = 0.3
    ind = np.arange(1, np.max(indice)+1)
    #bar(ind, m[0], width = width, color = 'green', **bar_kwargs)
    #errorbar(ind+width/2, m[0], yerr = v[0], **err_kwargs)
    #bar(ind+width+0.1, m[1], width = width, color = 'red', **bar_kwargs)
    #errorbar(ind+width+0.1+width/2, m[1], yerr = v[1], **err_kwargs)
    plot(ind, m[0], linewidth = 2, color = 'green', linestyle = 'o-')
    errorbar(ind, m[0], v[0], linewidth = 2, color = 'green')
    plot(ind, m[1], linewidth = 2, color = 'red', linestyle = 'o-')
    errorbar(ind, m[1], v[1], linewidth = 2, color = 'red')
    xlim(5, 18)
    xlabel("Representative steps")
    ylabel("Reaction time (s)")
    grid()


subplots_adjust(hspace = 0.35)
show()
    


    

