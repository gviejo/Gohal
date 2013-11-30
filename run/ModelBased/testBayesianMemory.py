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
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
noise = 0.0001
length_memory = 8
#threshold = 1.2
threshold = 1.0

nb_trials = 48
nb_blocs = 300
cats = CATS(nb_trials)

bww = BayesianWorkingMemory("v2", cats.states, cats.actions, length_memory, noise, threshold)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
bww.initializeList()
bww.initialize()
#sys.exit()
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
step, indice = getRepresentativeSteps(bww.responses, bww.state, bww.action, bww.responses)
y = computeMeanRepresentativeSteps(step)
distance = computeDistanceMatrix(bww.state, indice)

correct = np.array([bww.reaction[np.where((distance == i) & (bww.responses == 1) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
incorrect = np.array([bww.reaction[np.where((distance == i) & (bww.responses  == 0) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
mean_correct = np.array([np.mean(bww.reaction[np.where((distance == i) & (bww.responses  == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_correct = np.array([sem(bww.reaction[np.where((distance == i) & (bww.responses  == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
mean_incorrect = np.array([np.mean(bww.reaction[np.where((distance == i) & (bww.responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_incorrect = np.array([sem(bww.reaction[np.where((distance == i) & (bww.responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

step, indice = getRepresentativeSteps(human.reaction['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
rt_meg = computeMeanRepresentativeSteps(step) 
step, indice = getRepresentativeSteps(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
indice_meg = indice
y_meg = computeMeanRepresentativeSteps(step)
distance_meg = computeDistanceMatrix(human.stimulus['meg'], indice)

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 
step, indice = getRepresentativeSteps(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
y_fmri = computeMeanRepresentativeSteps(step)



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
    xticks(range(2,len(pcr['mean'][i])+1,2))
    xlabel("Trial")
    xlim(0.8, len(pcr['mean'][i])+1.02)
    ylim(-0.05, 1.05)
    yticks(np.arange(0, 1.2, 0.2))
    title('A')
    grid()


ax1 = plt.subplot(2,2,2)
ax1.plot(range(1, len(rt_meg[0])+1), rt_meg[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
ax1.errorbar(range(1, len(rt_meg[0])+1), rt_meg[0], rt_meg[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)

ax2 = ax1.twinx()
ax2.plot(range(1, len(rt[0])+1), rt[0], linewidth = 2, linestyle = '-', color = 'black')
ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
ax2.set_ylabel("Inference Level")
ax2.set_ylim(-5, 10)
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
ax1.set_ylabel("Reaction time (s)")
ax1.grid()
ax1.set_xlabel("Representative steps")
ax1.set_xticks([1,5,10,15])
ax1.set_yticks([0.46, 0.50, 0.54])
ax1.set_ylim(0.43, 0.56)
ax1.set_title('B')

################

ind = np.arange(1, len(rt[0])+1)
ax5 = subplot(2,2,3)
for i,j,k,l,m in zip([y, y_meg, y_fmri], 
                   ['blue', 'grey', 'grey'], 
                   ['BWW', 'MEG', 'FMRI'],
                   [1.0, 0.9, 0.9], 
                   ['-', '--', ':']):
    ax5.plot(ind, i[0], linewidth = 2, color = j, label = k, alpha = l, linestyle = m)
    ax5.errorbar(ind, i[0], i[1], linewidth = 2, color = j, alpha = l, linestyle = m)

ax5.grid()
ax5.set_ylabel("PCR %")    
ax5.set_yticks(np.arange(0, 1.2, 0.2))
ax5.set_xticks(range(2, 15, 2))
ax5.set_ylim(-0.05, 1.05)
ax5.legend(loc = 'lower right')

        
################
ax6 = subplot(4,2,6)
ind = np.arange(len(mean_correct))
labels = range(1, len(mean_correct)+1)
width = 0.4
bar_kwargs = {'width':width,'linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
ax6.p1 = bar(ind, mean_correct, color = 'green', **bar_kwargs)
ax6.errorbar(ind+width/2, mean_correct, yerr=var_correct, **err_kwargs)
ax6.p2 = bar(ind+width, mean_incorrect, color = 'red', **bar_kwargs)
ax6.errorbar(ind+3*width/2, mean_incorrect, yerr=var_incorrect, **err_kwargs)

grid()
xlim(0, np.max(distance))
#ylim(0.0, 2.0)
#xlabel("Distance")
ylabel("Inference Level")
xticks(ind+width/2, labels, color = 'k')
title("BWM")

###############

correct = np.array([human.reaction['meg'][np.where((distance_meg == i) & (human.responses['meg'] == 1) & (indice_meg > 5))] for i in xrange(1, int(np.max(distance_meg))+1)])
incorrect = np.array([human.reaction['meg'][np.where((distance_meg == i) & (human.responses['meg'] == 0) & (indice_meg > 5))] for i in xrange(1, int(np.max(distance_meg))+1)])
mean_correct = np.array([np.mean(human.reaction['meg'][np.where((distance_meg == i) & (human.responses['meg'] == 1) & (indice_meg > 5))]) for i in xrange(1, int(np.max(distance_meg))+1)])
var_correct = np.array([sem(human.reaction['meg'][np.where((distance_meg == i) & (human.responses['meg'] == 1) & (indice_meg > 5))]) for i in xrange(1, int(np.max(distance_meg))+1)])
mean_incorrect = np.array([np.mean(human.reaction['meg'][np.where((distance_meg == i) & (human.responses['meg'] == 0) & (indice_meg > 5))]) for i in xrange(1, int(np.max(distance_meg))+1)])
var_incorrect = np.array([sem(human.reaction['meg'][np.where((distance_meg == i) & (human.responses['meg'] == 0) & (indice_meg > 5))]) for i in xrange(1, int(np.max(distance_meg))+1)])

ax = subplot(4,2,8)
ind = np.arange(len(mean_correct))
labels = range(1, len(mean_correct)+1)
width = 0.4
bar_kwargs = {'width':width,'linewidth':2,'zorder':5, 'alpha':0.8}
err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
ax.p1 = bar(ind, mean_correct, color = 'green', **bar_kwargs)
ax.errorbar(ind+width/2, mean_correct, yerr=var_correct, **err_kwargs)
ax.p2 = bar(ind+width, mean_incorrect, color = 'red', **bar_kwargs)
ax.errorbar(ind+3*width/2, mean_incorrect, yerr=var_incorrect, **err_kwargs)

grid()
xlim(0, np.max(distance_meg))
ylim(0.0, 1.0)
xlabel("Distance")
ylabel("Reaction time")
xticks(ind+width/2, labels, color = 'k')
title("MEG")


################
subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)
#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testBWM3.pdf', bbox_inches='tight')


show()
