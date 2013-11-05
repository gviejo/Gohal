#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory entropy


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Models import BayesianWorkingMemory
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from time import time


# -----------------------------------
# FONCTIONS
# -----------------------------------
def testModel():
    bww.initializeList()
    for i in xrange(nb_blocs):
        sys.stdout.write("\r Blocs : %i" % i); sys.stdout.flush()                    
        cats.reinitialize()
        bww.initialize()
        entropy.append([])
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)
            entropy[-1].append(bww.computeInformationGain())
    bww.state = convertStimulus(np.array(bww.state))
    bww.action = convertAction(np.array(bww.action))
    bww.responses = np.array(bww.responses)
    bww.reaction = np.array(bww.reaction)
    bww.entropies = np.array(bww.entropies)    
    bww.choice = np.array(bww.choice)


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
noise = 0.0
length_memory = 10
#threshold = 1.2
threshold = 1.0

nb_trials = 42
nb_blocs = 100
cats = CATS(nb_trials)

bww = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise, threshold)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------

entropy = []

testModel()

entropy = np.array(entropy)
entropy = (computeEntropy(np.ones(5)*0.2, beta) - entropy)/computeEntropy(np.ones(5)*0.2, beta)
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(bww.responses, bww.state, bww.action, bww.responses)
pcr_human = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

#bww.reaction = (computeEntropy(np.ones(5)*0.2, beta) - bww.reaction)/computeEntropy(np.ones(5)*0.2, beta)
ent = extractStimulusPresentation(entropy, bww.state, bww.action, bww.responses)

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (9,4))
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          
#rcParams.update(params)                  
colors = ['blue', 'red', 'green']
subplot(1,2,1)
for i in xrange(3):
    plot(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(pcr['mean'][i])+1), pcr['mean'][i], pcr['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], linewidth = 2.5, linestyle = '--', color = colors[i], alpha = 0.7)    
    #errorbar(range(1, len(pcr_human['mean'][i])+1), pcr_human['mean'][i], pcr_human['sem'][i], linewidth = 2, linestyle = ':', color = colors[i], alpha = 0.6)
    ylabel("Probability correct responses")
    legend(loc = 'lower right')
    xticks(range(2,11,2))
    xlabel("Trial")
    xlim(0.8, 10.2)
    ylim(-0.05, 1.05)
    yticks(np.arange(0, 1.2, 0.2))
    title('A')
    grid()

subplot(1,2,2)
for i in xrange(3):
    plot(range(1, len(ent['mean'][i])+1), ent['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(ent['mean'][i])+1), ent['mean'][i], ent['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
    ylabel("Information")
    #legend(loc = 'lower right')
    xticks(range(2,11,2))
    xlabel("Trial")
    xlim(0.8, 10.2)
    #ylim(-0.05, 1.05)
    #yticks(np.arange(0, 1.2, 0.2))
    title('B')
    grid()


subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)

#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testKQL.pdf', bbox_inches='tight')

show()

