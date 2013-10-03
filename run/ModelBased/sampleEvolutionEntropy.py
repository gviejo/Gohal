#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory Entropy Evolution

script to analyse the eoluation of entropy under differents conditions


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
def createStimulusList(s, d, e):
    """Parameters are :
    - stimulus : {'s1','s2','s3'}
    - distance between two presentations
    - essai 
    """
    n_states = len(cats.states)    
    slist = np.tile(np.array(cats.states), ((nb_trials/n_states)+1, 1))
    #map(np.random.shuffle, slist)
    return slist.flatten()

def singleTest(stimuli_list, order):
    cats.reinitialize()
    cats.order = order
    cats.stimuli = stimuli_list
    bww.initialize()
    for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)    
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)
            #if np.sum(cats.incorrect[2]) == -3.0:
             #   sys.exit()
    bww.state = convertStimulus(np.array(bww.state))
    bww.action = convertAction(np.array(bww.action))
    bww.responses = np.array(bww.responses)
    bww.reaction = np.array(bww.reaction)

def modelTest(stimuli_list):
    for i in xrange(stimuli_list.shape(0)):
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
noise_width = 0.0

correlation = "Z"

nb_trials = 42
nb_blocs = 1

cats = CATS(nb_trials)


bww = BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0)
bww.setEntropyEvolution(1, nb_trials)
# -----------------------------------

slist = np.tile(np.array(cats.states), ((nb_trials/len(cats.states))+1, 1)).flatten()

# -----------------------------------
# Training session
# -----------------------------------
singleTest(slist, {'s1':1,'s2':3,'s3':4})
#----------------------------------

indice = dict({i:np.where(bww.state[0] == i)[0] for i in [1,2,3]})

params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}
rcParams.update(params)
colorscode = {1:'blue',
              2:'green',
              3:'red'}
ion()
for i in [1,2,3]:
    figure(str(i), figsize=(16,10))
    #----------------------------------
    # DATA Extraction
    #---------------------------------
    for j in xrange(0,nb_trials/3):
        ent = bww.entropy[0,indice[i][j]][bww.entropy[0,indice[i][j]] <> 0]
        ind = indice[i]
        #----------------------------------
        # Plot
        #----------------------------------
        subplot(4,4,j+1)
        plot(range(0, len(ent)), ent, 'o-', markersize = 1.2)
        xlabel("Nb inferences", fontsize = 10)
        ylabel("Entropy", fontsize = 10)
        legend()
        #grid()
        ylim(0,2.5)
        xlim(0,42)
        r = bww.responses[0][indice[i]]        
        for k,l in zip(range(0,45,3)[1:j+1], r[0:len(range(0,45,3)[1:j+1])][::-1]):
            if l == 0:
                axvline(k, 0, 1, linestyle = '--', color = 'red')
            elif l == 1:
                axvline(k, 0, 1, linestyle = '--', color = 'green')                
        
        
    subplots_adjust(left=0.08, wspace=0.3, right = 0.86)
    savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig_entropy_stim'+str(i)+'.pdf', bbox_inches = 'tight')
#show()        
