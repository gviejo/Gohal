#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

to test collins model

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Selection import CSelection
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from time import time
from scipy.optimize import leastsq

# -----------------------------------
# FONCTIONS
# -----------------------------------
def center(x):
    #x = x-np.mean(x)
    #x = x/np.std(x)
    x = x-np.median(x)
    x = x/(np.percentile(x, 75)-np.percentile(x, 25))
    return x

def testModel():
    model.startExp()
    for i in xrange(nb_blocs):
        cats.reinitialize()
        model.startBloc()
        for j in xrange(nb_trials):
            # sys.stdout.write("\r Bloc : %s | Trial : %i" % (i,j)); sys.stdout.flush()                    
            state = cats.getStimulus(j)
            action = model.chooseAction(state)
            reward = cats.getOutcome(state, action)
            model.updateValue(reward)

    model.state = convertStimulus(np.array(model.state))
    model.action = np.array(model.action)
    model.responses = np.array(model.responses)
    model.reaction = np.array(model.reaction)
    model.reaction = center(model.reaction)

#     model.weights = np.array(model.weights)
#     model.p_wm =np.array(model.p_wm)
#     model.p_rl =np.array(model.p_rl)

# # -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
parameters = dict({'noise':0.0001,
                    'length':10,
                    'alpha':0.8,
                    'beta':3.0,
                    'gamma':0.5,                    
                    'threshold':4.0,
                    'sigma':0.5,
                    'gain':2.0})




                    
                            

nb_trials = 39
nb_blocs = 4
cats = CATS(nb_trials)

model = CSelection(cats.states, cats.actions, parameters)

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
t1 = time()
testModel()
t2 = time()
# sys.exit()
print "\n"
print t2-t1
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])

# w = extractStimulusPresentation(model.weights, model.state, model.action, model.responses)
# wm = extractStimulusPresentation(model.p_wm, model.state, model.action, model.responses)
# rl = extractStimulusPresentation(model.p_rl, model.state, model.action, model.responses)

human.reaction['fmri'] = center(human.reaction['fmri'])

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 

step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
rt = computeMeanRepresentativeSteps(step)

rt = np.array(rt)

# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
figure(figsize = (9,7))
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

subplot(2,2,3)
for i in xrange(3):
    plot(range(1, len(w['mean'][i])+1), w['mean'][i], linewidth = 2, linestyle = '-', color = colors[i])
    xlim(0.8, 10.2)

subplot(2,2,4)
for i in xrange(3):
    plot(range(1, len(wm['mean'][i])+1), wm['mean'][i], linewidth = 2, linestyle = '-', color = colors[i])
    plot(range(1, len(rl['mean'][i])+1), rl['mean'][i], linewidth = 2, linestyle = '--', color = colors[i])
    xlim(0.8, 10.2)

ax1 = plt.subplot(2,2,2)
ax1.plot(range(1, len(rt_fmri[0])+1), rt_fmri[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)

#ax2 = ax1.twinx()
ax1.plot(range(1, len(rt[0])+1), rt[0], linewidth = 2, linestyle = '-', color = 'black')
#ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
#ax2.set_ylabel("Inference Level")
#x2.set_ylim(-5, 15)
ax1.grid()
############

subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)

#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testSelection.pdf', bbox_inches='tight')
#savefig('/home/viejo/Desktop/figure_guillaume_a_tord.pdf', bbox_inches='tight')
show()
