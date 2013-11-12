#!/usr/bin/python
# encoding: utf-8
"""
for benoit

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *

from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from scipy import stats
from sklearn.decomposition import PCA
# -----------------------------------
# FONCTIONS
# -----------------------------------
def computeDistanceMatrix():
    m, n = indice.shape    
    distance = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            if indice[i, j] > 1:
                distance[i,j] = j-np.where(state[i,0:j] == state[i,j])[0][-1]

    return distance

def computePCA():
    m, n = indice.shape
    tmp = np.zeros((m, 15))
    for i in xrange(m):
        for j in xrange(15):
            tmp[i, j] = np.mean(reaction[i][indice[i] == j+1])
    return tmp

def testRelationCorrectIncorrect():
    P = np.zeros(len(correct))
    P_discret = np.zeros(len(correct))
    for i in xrange(len(correct)):
        KS, p = stats.ks_2samp(correct[i], incorrect[i])
        #KS, p = stats.kruskal(correct[i], incorrect[i])
        P[i] = p
    P_discret[P < 0.01] = 3
    P_discret[(P > 0.01)*(P < 0.05)] = 2
    P_discret[(P > 0.05)*(P < 0.1)] = 1
    P_discret[P > 0.1] = 0
    return P, P_discret


# -----------------------------------
# -----------------------------------
# Parameters
# -----------------------------------
case = 'meg'
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------
state = human.stimulus[case]
action = human.action[case]
responses = human.responses[case]
reaction = human.reaction[case]

pcr_human = extractStimulusPresentation(responses, state, action, responses)


step, indice = getRepresentativeSteps(reaction, state, action, responses)

rt_human = computeMeanRepresentativeSteps(step) 

distance = computeDistanceMatrix()
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (12, 8))
ion()

######################################
# Plot 1 : RT vs responses 
correct = []
incorrect = []
m,n = distance.shape
for i in xrange(m):
    for j in xrange(n):
        if indice[i, j] > 5 and responses[i,j-1] == 1:
            correct.append(reaction[i,j])
        elif indice[i,j] > 5 and responses[i,j-1] == 0:
            incorrect.append(reaction[i,j])
correct = np.array(correct)
incorrect = np.array(incorrect)            
KS, p = stats.ks_2samp(correct,incorrect)


mean_correct = [np.mean(correct), np.var(correct)]
mean_incorrect = [np.mean(incorrect), np.var(incorrect)]

ax2 = subplot(2,1,1)
width = 0.4
bar_kwargs = {'width':width,'linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
ax2.p1 = bar(1, mean_correct[0], color = 'green', label = 'correct', **bar_kwargs)
ax2.errorbar(1+width/2, mean_correct[0], yerr=mean_correct[1], **err_kwargs)
ax2.p2 = bar(2, mean_incorrect[0], color='red', label = 'incorrect', **bar_kwargs)
ax2.errorbar(2+width/2, mean_incorrect[0], yerr = mean_incorrect[1], **err_kwargs)
xlim(0, np.max(distance))
ylim(0.0, 1.0)
ax2.plot([1.25, 2.15], [0.7, 0.7], linewidth = 2, color = 'black')
ax2.text(1.6, 0.74, "***")
ylabel("Reaction time")
xticks([1+width/2, 2+width/2], ['Correct', 'Incorrect'])
grid()

######################################
# Plot 2 : RT vs distance depending on responses
correct = np.array([reaction[np.where((distance == i) & (responses == 1) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
incorrect = np.array([reaction[np.where((distance == i) & (responses == 0) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
mean_correct = np.array([np.mean(reaction[np.where((distance == i) & (responses == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_correct = np.array([sem(reaction[np.where((distance == i) & (responses == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
mean_incorrect = np.array([np.mean(reaction[np.where((distance == i) & (responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_incorrect = np.array([sem(reaction[np.where((distance == i) & (responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

P, P_discret = testRelationCorrectIncorrect()

ax = subplot(2,1,2)
ind = np.arange(len(mean_correct))
labels = range(1, len(mean_correct)+1)
width = 0.4
bar_kwargs = {'width':width,'linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
ax.p1 = bar(ind, mean_correct, color = 'green', **bar_kwargs)
ax.errorbar(ind+width/2, mean_correct, yerr=var_correct, **err_kwargs)
ax.p2 = bar(ind+width, mean_incorrect, color = 'red', **bar_kwargs)
ax.errorbar(ind+3*width/2, mean_incorrect, yerr=var_incorrect, **err_kwargs)
d = -0.001; top = 0.8
for i in xrange(len(P_discret)):
    if P_discret[i] == 1:
        ax.plot([i+0.15, i+width+0.15], [top, top], linewidth = 2, color = 'black')
        ax.text(i+0.25, top+d, "*"*P_discret[i])
    elif P_discret[i] == 2:
        ax.plot([i+0.15, i+width+0.15], [top, top], linewidth = 2, color = 'black')
        ax.text(i+0.22, top+d, "*"*P_discret[i])
    elif P_discret[i] == 3:
        ax.plot([i+0.15, i+width+0.15], [top, top], linewidth = 2, color = 'black')
        ax.text(i+0.20, top+d, "*"*P_discret[i])
grid()
xlim(0, np.max(distance))
ylim(0.0, 1.0)
xlabel("Distance")
ylabel("Reaction time")
xticks(ind+width/2, labels, color = 'k')



#savefig('rt_relation_res.pdf', bbox_inches='tight')

show()

