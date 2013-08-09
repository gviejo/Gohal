#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory Entropy Evolution

3d plot of the evolution of entropy for differents levels of inference

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Models import BayesianWorkingMemory
from matplotlib import *
from Sweep import Optimization
from HumanLearning import HLearning
from matplotlib import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
noise = 0.0106
length_memory = 10

#nb_trials = 42
nb_trials = 42
nb_blocs = 46
cats = CATS()

bmw = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise)

opt = Optimization(human, cats, nb_trials, nb_blocs)

data = dict()
pcr = dict()
rs = []
for i in [1,2,3]:
    data[i] = []
    pcr[i] = []


# -----------------------------------

for l in xrange(3,30):
    # New memory length allowed
    print "Memory size :", l
    bmw.lenght_memory = l
    
    # -----------------------------------
    # SESSION MODELS
    # -----------------------------------
    opt.testModel(bmw)
    # -----------------------------------

    # -----------------------------------
    #order data
    # -----------------------------------
    bmw.state = convertStimulus(np.array(bmw.state))
    bmw.action = convertAction(np.array(bmw.action))
    bmw.responses = np.array(bmw.responses)
    bmw.reaction = np.array(bmw.reaction)
    rt = getRepresentativeSteps(bmw.reaction, bmw.state, bmw.action, bmw.responses)
    m_rt, sem_rt = computeMeanRepresentativeSteps(rt[0])
    rs.append(m_rt)
    rt2 = extractStimulusPresentation(bmw.reaction, bmw.state, bmw.action, bmw.responses)['mean']
    r = extractStimulusPresentation(bmw.responses, bmw.state, bmw.action, bmw.responses)['mean']
    for i in [1,2,3]:
        data[i].append(rt2[i-1])
        pcr[i].append(r[i-1])

# -----------------------------------
rs = np.array(rs)
for i in [1,2,3]:
    data[i] = np.array(data[i])
    pcr[i] = np.array(pcr[i])
# -----------------------------------
# Plot
# -----------------------------------

ticks_size = 15
legend_size = 15
title_size = 20
label_size = 19

fig = figure(figsize=plt.figaspect(0.5))

rc('legend',**{'fontsize':legend_size})
tick_params(labelsize = ticks_size)

for i in [1,2,3]:
    ax = fig.add_subplot(1,3,i, projection='3d')
    xs = np.arange(data[i].shape[1])
    ys = np.arange(data[i].shape[0])
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, data[i], rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    xlabel('Trials')
    ylabel('Lenght')
    title(i)

figure()
subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=-0.0)
for i in [1,2,3]:
    subplot(2,3,i)
    imshow(data[i], interpolation='nearest', origin='lower' )
    xlabel('Trials')
    ylabel('Lenght')
    title(i)    

for i in [1,2,3]:
    subplot(2,3,i+3)
    imshow(pcr[i], interpolation='nearest', origin='lower' )
    xlabel('Trials')
    ylabel('PCR')
    title(i)    

figure()
imshow(rs,  interpolation='nearest', origin='lower' )
xlabel('Representative Steps')
ylabel('Length')

show()

