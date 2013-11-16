#!/usr/bin/python
# encoding: utf-8
"""
Test for Bayesian Memory :
based on bayesian inference to calcul p(a/s)

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
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = bww.chooseAction(state)
            reward = cats.getOutcome(state, action)
            bww.updateValue(reward)

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
nb_blocs = 200
cats = CATS(nb_trials)

bww = BayesianWorkingMemory("test", cats.states, cats.actions, length_memory, noise, threshold)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
bww.initializeList()
bww.initialize()
#ys.exit()
t1 = time()
testModel()
t2 = time()

print "\n"
print t2-t1
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(bww.responses, bww.state, bww.action, bww.responses)
pcr_human = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

step, indice = getRepresentativeSteps(bww.reaction, bww.state, bww.action, bww.responses)
rt = computeMeanRepresentativeSteps(step)

step, indice = getRepresentativeSteps(human.reaction['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

rt_human = computeMeanRepresentativeSteps(step) 

step, indice = getRepresentativeSteps(bww.entropies, bww.state, bww.action, bww.responses)
ent = computeMeanRepresentativeSteps(step)

cho2 = extractStimulusPresentation(bww.choice, bww.state, bww.action, bww.responses)
step, indice = getRepresentativeSteps(bww.entropies, bww.state, bww.action, bww.responses)
cho = computeMeanRepresentativeSteps(step)


# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

# Probability of correct responses
figure(figsize = (11,8))
ion()
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}          
#rcParams.update(params)                  
colors = ['blue', 'red', 'green']
subplot(2,2,1)
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


ax1 = plt.subplot(2,2,2)
ax1.plot(range(1, len(rt_human[0])+1), rt_human[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.6)
ax1.errorbar(range(1, len(rt_human[0])+1), rt_human[0], rt_human[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.6)

ax2 = ax1.twinx()
ax2.plot(range(1, len(rt[0])+1), rt[0], linewidth = 2, linestyle = '-', color = 'black')
ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
ax2.set_ylabel("Inference Level")
ax2.set_ylim(-1, 11)
##
msize = 8.0
mwidth = 2.5
ax1.plot(1, 0.455, 'x', color = 'blue', markersize=msize, markeredgewidth=mwidth)
ax1.plot(1, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax1.plot(1, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(2, 0.455, 'o', color = 'blue', markersize=msize)
ax1.plot(2, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax1.plot(2, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(3, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
ax1.plot(3, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(4, 0.4445, 'o', color = 'red', markersize=msize)
ax1.plot(4, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
ax1.plot(5, 0.435, 'o', color = 'green', markersize=msize)
for i in xrange(6,16,1):
    ax1.plot(i, 0.455, 'o', color = 'blue', markersize=msize)
    ax1.plot(i, 0.4445, 'o', color = 'red', markersize=msize)
    ax1.plot(i, 0.435, 'o', color = 'green', markersize=msize)

##
ax1.set_ylabel("Reaction time (ms)")
ax1.grid()
ax1.set_xlabel("Representative steps")
ax1.set_xticks([1,5,10,15])
ax1.set_yticks([0.46, 0.50, 0.54])
ax1.set_ylim(0.43, 0.56)
ax1.set_title('B')

################
subplot(2,2,3)
for i in xrange(3):
    plot(range(1, len(cho2['mean'][i])+1), cho2['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(cho2['mean'][i])+1), cho2['mean'][i], cho2['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])    
    legend(loc = 'lower right')
    xticks(range(2,11,2))
    xlabel("Trial")
    ylabel("$P(Choice)^{Decision}$")
    xlim(0.8, 10.2)
    ylim(-0.05, 1.05)
    yticks(np.arange(0, 1.2, 0.2))
    title('A')
    grid()

################
subplot(2,2,4)
plot(range(1, len(cho[0])+1), cho[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.6)
errorbar(range(1, len(cho[0])+1), cho[0], cho[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.6)


subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)
#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testBWM3.pdf', bbox_inches='tight')



# figure(figsize = (11,8))

# step, indice = getRepresentativeSteps(human.reaction['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
# rt = human.extractRTSteps('meg', step, indice)
# tmp = np.reshape(rt, (12, 4, 15))
# for i in xrange(12):
#     subplot(3,4,i+1)
#     plot(range(1, 16), np.transpose(tmp[i]), 'o-', linewidth = 1.4)
#     axvline(5, 0, 1, linewidth =2)
#     grid()
#     ylim(0, 1)

show()
