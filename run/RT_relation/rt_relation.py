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
        #KS, p = stats.ks_2samp(correct[i], incorrect[i])
        KS, p = stats.kruskal(correct[i], incorrect[i])
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
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
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


# # -----------------------------------
# # Plot
# # -----------------------------------
# figure(figsize = (12, 8))
# ion()
# ######################################
# # Plot 1 : mean RT vs distance between correct
# tmp = np.array([reaction[np.where((distance == i) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
# mean_plot1 = np.array([np.mean(reaction[np.where((distance == i) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
# var_plot1 = np.array([sem(reaction[np.where((distance == i) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

# subplot(2,2,1)
# plot(range(1, len(mean_plot1)+1), mean_plot1, linewidth = 3, linestyle = '-', color = 'blue')
# errorbar(range(1, len(mean_plot1)+1), mean_plot1, var_plot1, linewidth = 3, linestyle = '-', color = 'blue')    
# grid()
# xlim(0, np.max(distance)+2)
# ylim(0.3, 0.7)
# xlabel("Distance")
# ylabel("Reaction time (ms)")

# ######################################
# # Plot 2 : RT vs distance depending on responses
# correct = np.array([reaction[np.where((distance == i) & (responses == 1) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
# incorrect = np.array([reaction[np.where((distance == i) & (responses == 0) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
# mean_correct = np.array([np.mean(reaction[np.where((distance == i) & (responses == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
# var_correct = np.array([sem(reaction[np.where((distance == i) & (responses == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
# mean_incorrect = np.array([np.mean(reaction[np.where((distance == i) & (responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
# var_incorrect = np.array([sem(reaction[np.where((distance == i) & (responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

# P, P_discret = testRelationCorrectIncorrect()

# ax = subplot(2,2,2)
# ind = np.arange(len(mean_correct))
# labels = range(1, len(mean_correct)+1)
# width = 0.4
# bar_kwargs = {'width':width,'linewidth':2,'zorder':5}
# err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
# ax.p1 = bar(ind, mean_correct, color = 'green', **bar_kwargs)
# ax.errorbar(ind+width/2, mean_correct, yerr=var_correct, **err_kwargs)
# ax.p2 = bar(ind+width, mean_incorrect, color = 'red', **bar_kwargs)
# ax.errorbar(ind+3*width/2, mean_incorrect, yerr=var_incorrect, **err_kwargs)
# d = -0.001; top = 0.8
# for i in xrange(len(P_discret)):
#     if P_discret[i] == 1:
#         ax.plot([i+0.15, i+width+0.15], [top, top], linewidth = 2, color = 'black')
#         ax.text(i+0.25, top+d, "*"*P_discret[i])
#     elif P_discret[i] == 2:
#         ax.plot([i+0.15, i+width+0.15], [top, top], linewidth = 2, color = 'black')
#         ax.text(i+0.22, top+d, "*"*P_discret[i])
#     elif P_discret[i] == 3:
#         ax.plot([i+0.15, i+width+0.15], [top, top], linewidth = 2, color = 'black')
#         ax.text(i+0.20, top+d, "*"*P_discret[i])
# grid()
# xlim(0, np.max(distance))
# ylim(0.0, 1.0)
# xlabel("Distance")
# ylabel("Reaction time")
# xticks(ind+width/2, labels, color = 'k')


# #######################################
# # Plot 3 : RT vs position for each distances
# pca = PCA(n_components=1)
# data = []
# plot3 = dict()
# plot3_pca = dict()
# #d = [1,5]
# d = xrange(1, 6)

# for i in d:
#     data.append([])
#     plot3[i] = list()
#     plot3_pca[i] = list()
#     for j in xrange(1, 18):
#         data[-1].append(reaction[np.where((distance == i) & (indice == j))])
#         if len(reaction[np.where((distance == i) & (indice == j))]):
#             tmp = pca.fit_transform(np.vstack(reaction[np.where((distance == i) & (indice == j))]))
#             plot3_pca[i].append([j, np.mean(tmp), np.var(tmp)])
#             plot3[i].append([j, np.mean(reaction[np.where((distance == i) & (indice == j))]), np.var(reaction[np.where((distance == i) & (indice == j))])])

#     plot3[i] = np.array(plot3[i])
#     plot3_pca[i] = np.array(plot3_pca[i])
    

# ax1 = subplot(2,2,3)
# for i in d:
#     c = np.random.rand(3,)
#     plot(plot3[i][:,0], plot3[i][:,1], '-', linewidth = 3, label = "D : "+str(i), color = c)
#     errorbar(plot3[i][:,0], plot3[i][:,1], plot3[i][:,2], linestyle = '-', linewidth = 3, color = c)

# legend()
# grid()
# ylabel("Reaction time")
# xlabel("Representative Steps")

# msize = 8.0
# mwidth = 2.5
# ax1.plot(1, 0.455, 'x', color = 'blue', markersize=msize, markeredgewidth=mwidth)
# ax1.plot(1, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
# ax1.plot(1, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
# ax1.plot(2, 0.455, 'o', color = 'blue', markersize=msize)
# ax1.plot(2, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
# ax1.plot(2, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
# ax1.plot(3, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
# ax1.plot(3, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
# ax1.plot(4, 0.4445, 'o', color = 'red', markersize=msize)
# ax1.plot(4, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
# ax1.plot(5, 0.435, 'o', color = 'green', markersize=msize)
# for i in xrange(6,16,1):
#     ax1.plot(i, 0.455, 'o', color = 'blue', markersize=msize)
#     ax1.plot(i, 0.4445, 'o', color = 'red', markersize=msize)
#     ax1.plot(i, 0.435, 'o', color = 'green', markersize=msize)


# #######################################
# # Plot 4 : fifth representative steps
# fifth = []
# for i in xrange(1,6):
#     ind = np.where((indice == 5)&(distance == i))
#     fifth.append([i, np.mean(reaction[ind]), np.var(reaction[ind])])
# fifth = np.array(fifth)
# width = 0.5
# colors = dict({1:'b', 2:'r', 3:'g', 4:'m', 5:'y', 6:'k'})
# #bar_kwargs = {'linewidth':2,'zorder':5}
# err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
# labels = range(1, 6)

# subplot(2,2,4)
# bar(fifth[:,0], fifth[:,1], width = width, linewidth = 2, zorder = 5, color = colors[5])
# errorbar(fifth[:,0]+width/2, fifth[:,1], yerr = fifth[:,2],  **err_kwargs)
# xticks(fifth[:,0]+width/2, labels, color = 'k')

# savefig('rt_relation.pdf', bbox_inches='tight')

######################################
# Second figure rt & perf on presentation time
######################################


perf = extractStimulusPresentation(responses, state, action, responses)
# PCA for reaction time
rt = extractStimulusPresentation(reaction, state, action, responses)



colors = ['blue', 'red', 'green']
figure(figsize = (6, 8))
ion()
ind = np.arange(1, len(perf['mean'][0])+1)
for i in xrange(3):
    ax1 = subplot(3,1,i+1)
    ax1.plot(ind, perf['mean'][i], linewidth = 2, color =colors[i])
    ax1.errorbar(ind, perf['mean'][i], perf['sem'][i], linewidth = 2, color = colors[i])    
    ax2 = ax1.twinx()
    ax2.plot(ind, rt['mean'][i], linewidth = 2, color =colors[i], linestyle = '--')
    ax2.errorbar(ind, rt['mean'][i], rt['sem'][i], linewidth = 2, color = colors[i], linestyle = '--')
    ax1.grid()
    ax1.set_ylabel("PCR %")    
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax1.set_xticks(range(2, 15, 2))
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Reaction time (s)")
    ax2.set_yticks([0.46, 0.50, 0.54])
    ax2.set_ylim(0.43, 0.56)
ax1.set_xlabel("Trial")
    
    
    
show()



