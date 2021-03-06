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

def inference(i):
    tmp = model.p_a_s[i]*np.vstack(model.p_s[i])
    return model.p_r_as[i] * np.reshape(np.repeat(tmp, 2, axis = 1), model.p_r_as[i].shape)
# -----------------------------------
# FONCTIONS
# -----------------------------------
def testModel():
    model.startExp()
    for i in xrange(nb_blocs):
        cats.reinitialize()
        model.startBloc()        
        for j in xrange(nb_trials):
            sys.stdout.write("\r Bloc : %s | Trial : %i" % (i,j)); sys.stdout.flush()
            state = cats.getStimulus(j)
            action = model.chooseAction(state)
            reward = cats.getOutcome(state, action)            
            model.updateValue(reward)
    model.state = convertStimulus(np.array(model.state))
    model.action = convertAction(np.array(model.action))
    model.responses = np.array(model.responses)
    model.reaction = np.array(model.reaction)
    model.entropies = np.array(model.entropies)    

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
parameters = dict({'length':10,
                    'noise':0.0,
                    'threshold':0.4,
                    'sigma':0.1})

nb_trials = 39
nb_blocs = 100
cats = CATS(nb_trials)

model = BayesianWorkingMemory(cats.states, cats.actions, parameters)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------

t1 = time()
testModel()
t2 = time()

print "\n"
print t2-t1
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])

step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
rt = computeMeanRepresentativeSteps(step)
step, indice = getRepresentativeSteps(model.responses, model.state, model.action, model.responses)
y = computeMeanRepresentativeSteps(step)
distance = computeDistanceMatrix(model.state, indice)

correct = np.array([model.reaction[np.where((distance == i) & (model.responses == 1) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
incorrect = np.array([model.reaction[np.where((distance == i) & (model.responses  == 0) & (indice > 5))] for i in xrange(1, int(np.max(distance))+1)])
mean_correct = np.array([np.mean(model.reaction[np.where((distance == i) & (model.responses  == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_correct = np.array([sem(model.reaction[np.where((distance == i) & (model.responses  == 1) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
mean_incorrect = np.array([np.mean(model.reaction[np.where((distance == i) & (model.responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])
var_incorrect = np.array([sem(model.reaction[np.where((distance == i) & (model.responses == 0) & (indice > 5))]) for i in xrange(1, int(np.max(distance))+1)])

step, indice = getRepresentativeSteps(human.reaction['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
rt_meg = computeMeanRepresentativeSteps(step) 
step, indice = getRepresentativeSteps(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
indice_meg = indice
y_meg = computeMeanRepresentativeSteps(step)
distance_meg = computeDistanceMatrix(human.stimulus['meg'], indice)

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 
step, indice = getRepresentativeSteps(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
indice_fmri = indice
y_fmri = computeMeanRepresentativeSteps(step)
distance_fmri = computeDistanceMatrix(human.stimulus['fmri'], indice)


# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

# Probability of correct responses
figure(figsize = (11,8))

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
ax1.plot(range(1, len(rt_fmri[0])+1), rt_fmri[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)

ax2 = ax1.twinx()
ax2.plot(range(1, len(rt[0])+1), rt[0], linewidth = 2, linestyle = '-', color = 'black')
ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
ax2.set_ylabel("Inference Level")
#ax2.set_ylim(-5, 10)
# ##
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

##
ax1.set_ylabel("Reaction time (s)")
ax1.grid()
ax1.set_xlabel("Representative steps")
#ax1.set_xticks([1,5,10,15])
#ax1.set_yticks([0.46, 0.50, 0.54])
#ax1.set_ylim(0.43, 0.56)
ax1.set_title('B')

################

# ind = np.arange(1, len(rt[0])+1)
# ax5 = subplot(2,2,3)
# for i,j,k,l,m in zip([y, y_meg, y_fmri], 
#                    ['blue', 'grey', 'grey'], 
#                    ['model', 'MEG', 'FMRI'],
#                    [1.0, 0.9, 0.9], 
#                    ['-', '--', ':']):
#     ax5.plot(ind, i[0], linewidth = 2, color = j, label = k, alpha = l, linestyle = m)
#     ax5.errorbar(ind, i[0], i[1], linewidth = 2, color = j, alpha = l, linestyle = m)

# ax5.grid()
# ax5.set_ylabel("PCR %")    
# ax5.set_yticks(np.arange(0, 1.2, 0.2))
# ax5.set_xticks(range(2, 15, 2))
# ax5.set_ylim(-0.05, 1.05)
# ax5.legend(loc = 'lower right')

        
# ################
# ax6 = subplot(4,2,6)
# ind = np.arange(len(mean_correct))
# labels = range(1, len(mean_correct)+1)
# width = 0.4
# bar_kwargs = {'width':width,'linewidth':2,'zorder':5}
# err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
# ax6.p1 = bar(ind, mean_correct, color = 'green', **bar_kwargs)
# ax6.errorbar(ind+width/2, mean_correct, yerr=var_correct, **err_kwargs)
# ax6.p2 = bar(ind+width, mean_incorrect, color = 'red', **bar_kwargs)
# ax6.errorbar(ind+3*width/2, mean_incorrect, yerr=var_incorrect, **err_kwargs)

# grid()
# xlim(0, np.max(distance))
# #ylim(0.0, 2.0)
# #xlabel("Distance")
# ylabel("Inference Level")
# xticks(ind+width/2, labels, color = 'k')
# title("BWM")

# ###############

# correct = np.array([human.reaction['fmri'][np.where((distance_fmri == i) & (human.responses['fmri'] == 1) & (indice_fmri > 5))] for i in xrange(1, int(np.max(distance_fmri))+1)])
# incorrect = np.array([human.reaction['fmri'][np.where((distance_fmri == i) & (human.responses['fmri'] == 0) & (indice_fmri > 5))] for i in xrange(1, int(np.max(distance_fmri))+1)])
# mean_correct = np.array([np.mean(human.reaction['fmri'][np.where((distance_fmri == i) & (human.responses['fmri'] == 1) & (indice_fmri > 5))]) for i in xrange(1, int(np.max(distance_fmri))+1)])
# var_correct = np.array([sem(human.reaction['fmri'][np.where((distance_fmri == i) & (human.responses['fmri'] == 1) & (indice_fmri > 5))]) for i in xrange(1, int(np.max(distance_fmri))+1)])
# mean_incorrect = np.array([np.mean(human.reaction['fmri'][np.where((distance_fmri == i) & (human.responses['fmri'] == 0) & (indice_fmri > 5))]) for i in xrange(1, int(np.max(distance_fmri))+1)])
# var_incorrect = np.array([sem(human.reaction['fmri'][np.where((distance_fmri == i) & (human.responses['fmri'] == 0) & (indice_fmri > 5))]) for i in xrange(1, int(np.max(distance_fmri))+1)])

# ax = subplot(4,2,8)
# ind = np.arange(len(mean_correct))
# labels = range(1, len(mean_correct)+1)
# width = 0.4
# bar_kwargs = {'width':width,'linewidth':2,'zorder':5, 'alpha':0.8}
# err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}
# ax.p1 = bar(ind, mean_correct, color = 'green', **bar_kwargs)
# ax.errorbar(ind+width/2, mean_correct, yerr=var_correct, **err_kwargs)
# ax.p2 = bar(ind+width, mean_incorrect, color = 'red', **bar_kwargs)
# ax.errorbar(ind+3*width/2, mean_incorrect, yerr=var_incorrect, **err_kwargs)

# grid()
# xlim(0, np.max(distance_fmri))
# ylim(0.0, 1.0)
# xlabel("Distance")
# ylabel("Reaction time")
# xticks(ind+width/2, labels, color = 'k')
# title("fmri")


################
subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)
#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testBWM3.pdf', bbox_inches='tight')


show()
