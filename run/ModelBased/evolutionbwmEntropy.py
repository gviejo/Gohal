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
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def createStimulusList(i,j):
    n_states = len(cats.states)    
    s = np.tile(np.array(cats.states), ((nb_trials/n_states)+1, 1))
    map(np.random.shuffle, s)
    return s.flatten()

def modelTest(stimuli_list):
    for i in xrange(nb_blocs):
        cats.reinitialize()
        bww.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)
    bww.state = convertStimulus(np.array(bww.state))
    bww.action = convertAction(np.array(bww.action))
    bww.responses = np.array(bww.responses)
    bww.reaction = np.array(bww.reaction)

# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
beta = 1.7
length_memory = 100
noise_width = 0.01

correlation = "Z"

nb_trials = 42
nb_blocs = 100

cats = CATS(nb_trials)


bww = BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0)
bww.setEntropyEvolution(nb_blocs, nb_trials)
# -----------------------------------

# -----------------------------------
# Training session
# -----------------------------------
modelTest(createStimulusList(0,0))
#----------------------------------

#----------------------------------
# DATA Extraction
#---------------------------------
bad = dict({1:[1,2,3],
            2:[4,5,6],
            3:[7,8,9],
            4:[10,11,12],
            5:[13,14,15],
            6:[7,8,9,12,13,14,15,16,17]})
for i in xrange(7,16):
    bad[i] = range(12,100)


crap = np.reshape(np.arange(bww.reaction.shape[0]*bww.reaction.shape[1]), bww.reaction.shape)

step, indice = getRepresentativeSteps(crap, bww.state, bww.action, bww.responses)
order = extractStimulusPresentation2(crap, bww.state, bww.action, bww.responses)

superstuf = {i:{j:np.array([bww.entropy[np.where(crap == k)][0] for k in np.intersect1d(order[i].flatten(), step[j])]) for j in step.keys()} for i in order.iterkeys()}

for i in superstuf.iterkeys():
    for j in superstuf[i].iterkeys():
        if len(superstuf[i][j]) > 0:
            size = np.sum(superstuf[i][j] <> 0, axis = 1)
            tmp = []
            for k in np.unique(size):
                if k in bad[j]:
                    tmp.append(np.mean(superstuf[i][j][size == k][:,0:k], axis = 0))
            superstuf[i][j] = tmp

#----------------------------------
# Plot
#----------------------------------
ion()
for f in [(1,6),(6,11)]:
    figure(figsize = (16, 12))
    count = 1
    for i in superstuf.iterkeys():
        for j in range(f[0],f[1]):
            subplot(3,5,count)
            tmp = superstuf[i][j]
            for k in tmp:
                plot(range(1, len(k)+1), k, 'o-')
            grid()
            count+=1
            #ylim(0.5,5)
    for i,j in zip(xrange(1,6), xrange(f[0],f[1])):
        subplot(3,5,i)
        title("RStep "+str(j))
    for i in [1,6,11]:
        subplot(3,5,i)
        ylabel('Entropy')
    for i in xrange(11,16):
        subplot(3,5,i)
        xlabel('Inference Level')
        
        
    
subplots_adjust(left = 0.08, wspace = 0.4, right = 0.86, hspace = 0.35)


show()

"""
figure()
subplot(3,5,1)
bar(range(1, len(gain[5]['mean'])+1), gain[5]['mean'], yerr = gain[5]['var'], width = 0.3)
xlabel('Inference Level')
xticks(range(1, len(gain[5]['mean'])+1))
ylabel('Entropy Gain')
title("1 Error search")

gain = dict()
n_inferences_max = [5]
for l in n_inferences_max:
    data = {i:list() for i in xrange(1,l+1)}
    for i in xrange(1,4):
        tmp = np.reshape(np.where(bww.state[:,0:l+1] == i)[1], (nb_blocs, 2))
    distance = tmp[:,1]-tmp[:,0]
    for j in xrange(nb_blocs):
        data[distance[j]].append(bww.entropy[j,tmp[j,1],0]-bww.entropy[j,tmp[j,1],distance[j]])
    gain[l] = dict({'mean':[np.mean(data[k]) for k in data.iterkeys()],
                    'var':[np.var(data[k]) for k in data.iterkeys()]})
"""
