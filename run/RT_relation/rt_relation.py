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
# Plot 1 : mean RT vs distance between correct
tmp = np.array([reaction[np.where((distance == i) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
mean_plot1 = np.array([np.mean(reaction[np.where((distance == i) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_plot1 = np.array([sem(reaction[np.where((distance == i) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

subplot(2,2,1)
plot(range(1, len(mean_plot1)+1), mean_plot1, linewidth = 3, linestyle = '-', color = 'blue')
errorbar(range(1, len(mean_plot1)+1), mean_plot1, var_plot1, linewidth = 3, linestyle = '-', color = 'blue')    
grid()
xlim(0, np.max(distance)+2)
ylim(0.3, 1.0)
xlabel("Distance")
ylabel("Reaction time (ms)")

######################################
# Plot 2 : RT vs distance depending on responses
correct = np.array([reaction[np.where((distance == i) & (responses == 1) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
incorrect = np.array([reaction[np.where((distance == i) & (responses == 0) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
mean_correct = np.array([np.mean(reaction[np.where((distance == i) & (responses == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_correct = np.array([sem(reaction[np.where((distance == i) & (responses == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
mean_incorrect = np.array([np.mean(reaction[np.where((distance == i) & (responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_incorrect = np.array([sem(reaction[np.where((distance == i) & (responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

subplot(2,2,2)
# for i in xrange(len(correct)):
#     plot(np.ones(len(correct[i]))*(i+1), correct[i], 'o', alpha = 0.4, color = 'green')
# for i in xrange(len(incorrect)):
#     plot(np.ones(len(incorrect[i]))*(i+1.4), incorrect[i], 'o', alpha = 0.4, color = 'red')
plot(range(1, len(mean_correct)+1), mean_correct, linewidth = 3, linestyle = '-', color = 'green')
errorbar(range(1, len(mean_correct)+1), mean_correct, var_correct, linewidth = 3, linestyle = '-', color = 'green')    
plot(range(1, len(mean_incorrect)+1), mean_incorrect, linewidth = 3, linestyle = '-', color = 'red')
errorbar(range(1, len(mean_incorrect)+1), mean_incorrect, var_incorrect, linewidth = 3, linestyle = '-', color = 'red')    
grid()
xlim(0, np.max(distance)+2)
ylim(0.3, 1.0)
xlabel("Distance")
ylabel("Reaction time")
legend()

#######################################
# Plot 3 : RT vs position for each distances
pca = PCA(n_components=1)
data = []
plot3 = []
plot3_pca = []

for i in xrange(1, 6 ):
    data.append([])
    plot3.append([])    
    plot3_pca.append([])
    for j in xrange(5, 18):
        data[-1].append(reaction[np.where((distance == i) & (indice == j))])
        if len(reaction[np.where((distance == i) & (indice == j))]):
            tmp = pca.fit_transform(np.vstack(reaction[np.where((distance == i) & (indice == j))]))
            plot3_pca[-1].append([j, np.mean(tmp), np.var(tmp)])
            plot3[-1].append([j, np.mean(reaction[np.where((distance == i) & (indice == j))]), np.var(reaction[np.where((distance == i) & (indice == j))])])

    plot3[-1] = np.array(plot3[-1])
    plot3_pca[-1] = np.array(plot3_pca[-1])
    

subplot(2,2,3)
for i in xrange(len(plot3)):
    c = np.random.rand(3,)
    plot(plot3[i][:,0], plot3[i][:,1], '-', linewidth = 3, label = "D : "+str(i+1), color = c)
    errorbar(plot3[i][:,0], plot3[i][:,1], plot3[i][:,2], linestyle = '-', linewidth = 3, color = c)
legend()
grid()
ylabel("Reaction time")
xlabel("Representative Steps")
show()



