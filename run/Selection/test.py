#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
Test for Fusion Selection


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from Selection import FSelection
from Models import *
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from time import time
from scipy.optimize import leastsq

# -----------------------------------
# FONCTIONS
# -----------------------------------
def _convertStimulus(s):
        return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

def center(x):
    #x = x-np.mean(x)
    #x = x/np.std(x)
    x = x-np.median(x)
    x = x/(np.percentile(x, 75)-np.percentile(x, 25))
    return x

def testModel():
    rt = np.zeros((nb_blocs, 15))
    for k in xrange(nb_blocs):                
        for i in xrange(4):
            cats.reinitialize()
            cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
            model.startBloc()
            for j in xrange(nb_trials):
                sys.stdout.write("\r Bloc : %s | Trial : %i" % (k,i)); sys.stdout.flush()                    
                state = cats.getStimulus(j)                
                action = model.chooseAction(state)                   
                reward = cats.getOutcome(state, action, case = 'fmri')
                model.updateValue(reward)
        reaction = np.array(model.reaction[-4:])
        reaction = reaction - 0.23355812
        reaction = reaction / 0.32967988
        state = convertStimulus(np.array(model.state[-4:]))
        action = np.array(model.action[-4:])
        responses = np.array(model.responses[-4:])
        step, indice = getRepresentativeSteps(reaction, state, action, responses)
        rt[k] = computeMeanRepresentativeSteps(step)[0]
        for i in xrange(-4,0,1):
            model.reaction[i] = list(reaction[i])
    return rt
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------
s = 'S9'

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
# parameters = {'alpha': 0.98,
#                 'beta': 4.75,
#                 'gain': 2.31,
#                 'length': 14.925,
#                 'noise': 1.0,
#                 'sigma': 0.0,
#                 'threshold': 0.0001}

parameters = {'alpha': 0.61835852000000002,
               'beta': 4.2165891999999996,
               'gain': 2.7591210999999998,
               'length': 1.0,
               'noise': 0.10000000000000001,
               'sigma': 0.092075579245000011,
               'threshold': 1.0}


nb_trials = 39
nb_blocs = 100
cats = CATS(nb_trials)

model = FSelection(cats.states, cats.actions, parameters)
# model = QLearning(cats.states, cats.actions, parameters)
# model = BayesianWorkingMemory(cats.states, cats.actions, {'length':1,'threshold':0.1,'noise':0.0,'sigma':0.1})
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
rt = testModel()
# -----------------------------------
#order data
# -----------------------------------
model.reaction = np.array(model.reaction).reshape(nb_blocs, 4*nb_trials)
# model.reaction = np.array(map(center, list(model.reaction)))
# plot(np.mean(model.reaction, 0), 'o-')

plot(np.mean(rt, 0))

show()
sys.exit()

# tirage = model.reaction.flatten()

# plot(model.reaction.flatten(), 'o-')
# show()




pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
pcr_human = extractStimulusPresentation(human.responses['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])

# thr = extractStimulusPresentation(model.thr, model.state, model.action, model.responses)
# thr_free = extractStimulusPresentation(model.thr_free, model.state, model.action, model.responses)

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 

figure()
[plot(rt[i]) for i in xrange(nb_blocs)]
plot(np.mean(rt,0), linewidth = 5)


# fitfunc = lambda p, x: p[0] + p[1] * x
# errfunc = lambda p, x, y : (y - fitfunc(p, x))
# p = leastsq(errfunc, [1.0, -1.0], args = (rt[0], rt_fmri[0]), full_output = False)
# rt[0] = fitfunc(p[0], rt[0])
#rt[1] = fitfunc(p[0], rt[1])
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

# # subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)

# # #savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testSelection.pdf', bbox_inches='tight')
# # #savefig('/home/viejo/Desktop/figure_guillaume_a_tord.pdf', bbox_inches='tight')
show()
