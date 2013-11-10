#!/usr/bin/python
# encoding: utf-8
"""
explore relation between reaction time, stimulus and action choice

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *

from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from time import time
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

def testRelation(data):
    n = len(data)
    P = np.eye(n)
    P_discret = np.eye(n)
    for i in xrange(n):
        for j in xrange(n):
            if j < i:
                KS, p = stats.ks_2samp(tmp[j], tmp[i])
                #KS, p = stats.kruskal(tmp[j], tmp[i])
                #KS, p = stats.mannwhitneyu(tmp[j], tmp[i])
                P[i, j] = p
    P_discret[P < 0.01] = 3
    P_discret[(P > 0.01)*(P < 0.05)] = 2
    P_discret[(P > 0.05)*(P < 0.1)] = 1
    P_discret[P > 0.1] = 0
    P_discret[np.triu_indices(n)] = 0
    return P, P_discret

def plotStars(data):
    n = len(data)
    #pos = [0.6, 0.68, 0.75, 0.82, 0.97]
    pos = np.arange(0.6, 5, 0.08)
    d = -0.001
    # side by side 
    for i, j in zip(xrange(1, n), xrange(n-1)):
        if data[i, j] != 0:            
            ax.plot([j+width-0.15, i+0.15], [pos[0], pos[0]], linewidth = 2, color = 'black')
            ax.text(j+width+0.15, pos[0]+d, "*"*data[i,j])
    # two side
    for i,j in zip(xrange(2, n), xrange(n-2)):
        if data[i,j] != 0 and j%2 != 0:
            ax.plot([j+width-0.15, i+0.15], [pos[1], pos[1]], linewidth = 2, color = 'black')
            ax.text(j+2*width+0.15, pos[1]+d, "*"*data[i,j])
        elif data[i,j] != 0 and j%2 == 0:
            ax.plot([j+width-0.15, i+0.15], [pos[2], pos[2]], linewidth = 2, color = 'black')
            ax.text(j+2*width+0.15, pos[2]+d, "*"*data[i,j])
    # 
    for i,j in zip(xrange(3, n), xrange(n-3)):
        if data[i,j] != 0 and j%2 != 0:
            ax.plot([j+width-0.15, i+0.15], [pos[3], pos[3]], linewidth = 2, color = 'black')
            ax.text(j+3*width+0.15, pos[3]+d, "*"*data[i,j])
        elif data[i,j] != 0 and j%2 == 0:
            ax.plot([j+width-0.15, i+0.15], [pos[4], pos[4]], linewidth = 2, color = 'black')
            ax.text(j+3*width+0.15, pos[4]+d, "*"*data[i,j])
    #
    for i,j in zip(xrange(4, n), xrange(n-4)):
        if data[i,j] != 0 and j%2 != 0:
            ax.plot([j+width-0.15, i+0.15], [pos[5], pos[5]], linewidth = 2, color = 'black')
            ax.text(j+4*width+0.15, pos[5]+d, "*"*data[i,j])
        elif data[i,j] != 0 and j%2 == 0:
            ax.plot([j+width-0.15, i+0.15], [pos[6], pos[6]], linewidth = 2, color = 'black')
            ax.text(j+4*width+0.15, pos[6]+d, "*"*data[i,j])
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
# Plot 2
# -----------------------------------
stim1 = []
for i in [2,6]:
    stim1.append([])
    for j in xrange(1, int(np.max(distance[np.where((state==1)&(indice==i))]))+1):
        ind = np.where((distance == j)&(indice == i)&(state == 1))
        stim1[-1].append([j, np.mean(reaction[ind]), sem(reaction[ind])])
    stim1[-1] = np.array(stim1[-1])
stim2 = []
for i in [2,3,4,6]:
    stim2.append([])
    for j in xrange(1, int(np.max(distance[np.where((state==2)&(indice==i))]))+1):
        ind = np.where((distance == j)&(indice == i)&(state == 2))
        stim2[-1].append([j, np.mean(reaction[ind]), sem(reaction[ind])])
    stim2[-1] = np.array(stim2[-1])
stim3 = []
for i in [2,3,4,5,6]:    
    stim3.append([])
    for j in xrange(1, int(np.max(distance[np.where((state==3)&(indice==i))]))+1):
        ind = np.where((distance == j)&(indice == i)&(state == 3))
        stim3[-1].append([j, np.mean(reaction[ind]), sem(reaction[ind])])
    stim3[-1] = np.array(stim3[-1])

figure(figsize = (12, 8))
ion()
width = 1/7.
colors = dict({1:'b', 2:'r', 3:'g', 4:'m', 5:'y', 6:'k'})
#bar_kwargs = {'linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
labels = range(1, 6)

ax1 = subplot(3,1,1)
for i,j in zip(xrange(len(stim1)), [2,6]):
    ind = np.arange(5)+width*(j-2)
    bar(ind, stim1[i][0:5,1], width = width, linewidth = 2, zorder = 5, color = colors[j], label = "Step "+str(j))
    errorbar(ind+(width/2), stim1[i][0:5,1], yerr=stim1[i][0:5,2], **err_kwargs)
    xticks(ind+width/2, labels, color = 'k')
legend()
title("One error")
ylim(0.0, 1.0)
xlim(0, 5.8)
xlabel("Distance")
ylabel("Reaction time")
grid()

ax2 = subplot(3,1,2)
for i,j in zip(xrange(len(stim2)), [2,3,4,6]):
    ind = np.arange(5)+width*(j-2)    
    bar(ind, stim2[i][0:5,1], width = width, linewidth = 2, zorder = 5, color = colors[j], label = "Step "+str(j))
    errorbar(ind+width/2, stim2[i][0:5,1], yerr=stim2[i][0:5,2], **err_kwargs)
    xticks(ind+width/2, labels, color = 'k')    
legend()
title("Three error")
ylim(0.0, 1.0)
xlim(0, 5.8)
xlabel("Distance")
ylabel("Reaction time")
grid()


subplot(3,1,3)
for i,j in zip(xrange(len(stim3)), [2,3,4,5,6]):
    ind = np.arange(5)+width*(j-2)    
    ind = ind[0:len(stim3[i][0:5, 1])]
    bar(ind, stim3[i][0:5,1], width = width, linewidth = 2, zorder = 5, color = colors[j], label = "Step "+str(j))
    errorbar(ind+width/2, stim3[i][0:5,1], yerr=stim3[i][0:5,2], **err_kwargs)
    xticks(ind+width/2, labels, color = 'k')        
legend()
title("Four error")
ylim(0.0, 1.0)
xlim(0, 5.8)
xlabel("Distance")
ylabel("Reaction time")
grid()

subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.45, right = 0.86)
savefig('rt_relation_first.pdf', bbox_inches='tight')
show()

