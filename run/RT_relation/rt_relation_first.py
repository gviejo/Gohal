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
stim1 = []
for i in [2,6]:
    stim1.append([])
    for j in xrange(1, int(np.max(distance[np.where((state==1)&(indice==2))]))+1):
        stim1[-1].append([j, np.mean(reaction[np.where((distance == j)&(indice == i)&(state == 1))])])
    stim1[-1] = np.array(stim1[-1])
stim2 = []
for i in [2,3,4,6]:
    stim2.append([])
    for j in xrange(1, int(np.max(distance[np.where((state==2)&(indice==i))]))+1):
        stim2[-1].append([j, np.mean(reaction[np.where((distance == j)&(indice == i)&(state == 2))])])
    stim2[-1] = np.array(stim2[-1])
stim3 = []
for i in [2,3,4,5,6]:    
    stim3.append([])
    for j in xrange(1, int(np.max(distance[np.where((state==3)&(indice==i))]))+1):
        stim3[-1].append([j, np.mean(reaction[np.where((distance == j)&(indice == i)&(state == 3))])])
    stim3[-1] = np.array(stim3[-1])

figure(figsize = (12, 8))
ion()
subplot(3,1,1)
for i,j in zip(xrange(len(stim1)), [2,6]):
    plot(stim1[i][:,0], stim1[i][:,1], "o-", linewidth = 2.0, label = 'Ind : '+str(j))
legend()

subplot(3,1,2)
for i,j in zip(xrange(len(stim2)), [2,3,4,6]):
    plot(stim2[i][:,0], stim2[i][:,1], "o-", linewidth = 2.0, label = 'Ind : '+str(j))
legend()

subplot(3,1,3)
for i,j in zip(xrange(len(stim3)), [2,3,4,5,6]):
    plot(stim3[i][:,0], stim3[i][:,1], "o-", linewidth = 2.0,  label = 'Ind : '+str(j))
legend()

show()

