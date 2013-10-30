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
from scipy import stats

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

def plotStars(data, pos):
    n = len(data)
    #pos = [0.6, 0.68, 0.75, 0.82, 0.97]    
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
        if data[i,j] != 0 and j%2 == 0:
            ax.plot([j+width-0.15, i+0.15], [pos[5], pos[5]], linewidth = 2, color = 'black')
            ax.text(j+4*width+0.15, pos[5]+d, "*"*data[i,j])
        elif data[i,j] != 0 and j%2 != 0:
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
tmp = np.array([reaction[np.where((distance == i) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
mean_plot1 = np.array([np.mean(reaction[np.where((distance == i) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_plot1 = np.array([sem(reaction[np.where((distance == i) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

pvalue, stars = testRelation(tmp)
# -----------------------------------
# Plot
# -----------------------------------
ind = np.arange(len(mean_plot1))
width = 0.5
labels = range(1,8)

bar_kwargs = {'width':width,'color':'y','linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}

fig = figure(figsize = (10, 4))
ion()
ax = subplot(1,2,2)

ax.p1 = bar(ind, mean_plot1, **bar_kwargs)
ax.errs = errorbar(ind+width/2, mean_plot1, yerr=var_plot1, **err_kwargs)
if case == 'meg':
    plotStars(stars, pos = np.arange(0.6, 5, 0.06))
else:
    plotStars(stars, pos = np.arange(1, 5, 0.09))
grid()
xlabel("Distance")
ylabel("Reaction time (ms)")
xticks(ind+width/2, labels, color = 'k')
if case == 'meg':   
    ylim(0, 0.95)
else:
    ylim(0, 1.6)
title("Consolidation")

# -----------------------------------
# Plot 2 First step
# -----------------------------------

tmp = np.array([reaction[np.where((distance == i) & (1 < indice) & (indice < 6))] for i in xrange(1, int(np.max(distance))+1)])
mean_plot1 = np.array([np.mean(reaction[np.where((distance == i) & (1 < indice) & (indice < 6))]) for i in xrange(1, int(np.max(distance))+1)])
var_plot1 = np.array([sem(reaction[np.where((distance == i) & (1 < indice) & (indice < 6))]) for i in xrange(1, int(np.max(distance))+1)])

pvalue, stars = testRelation(tmp)

ind = np.arange(len(mean_plot1))
width = 0.5
labels = range(1,8)

bar_kwargs = {'width':width,'color':'y','linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}


ion()
ax = subplot(1,2,1)

ax.p1 = bar(ind, mean_plot1, **bar_kwargs)
ax.errs = errorbar(ind+width/2, mean_plot1, yerr=var_plot1, **err_kwargs)
if case == 'meg':
    plotStars(stars, pos = np.ones(10)*0.6)
else:
    plotStars(stars, pos = np.arange(1, 5, 0.09))
grid()
xlabel("Distance")
ylabel("Reaction time (ms)")
xticks(ind+width/2, labels, color = 'k')
if case == 'meg':
    ylim(0, 0.95)
else:
    ylim(0, 1.6)
title("Acquisition")


show()


