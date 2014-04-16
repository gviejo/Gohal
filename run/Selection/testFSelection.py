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
from matplotlib import *
from pylab import *
from HumanLearning import HLearning
from time import time
from scipy.optimize import leastsq

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
    


# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
very_good_parameters = dict({'noise':0.0001,
                    'length':10,
                    'alpha':0.8,
                    'beta':3.0,
                    'gamma':0.4,
                    'threshold':4.0,
                    'gain':2.0})

parameters = dict({'noise':0.2,
                    'length':7,
                    'alpha':0.9,
                    'beta':3.5,
                    'gamma':0.5,
                    'gain':0.6,
                    'threshold':1.5,
                    'sigma':0.01})
                            

nb_trials = 39
nb_blocs = 50
cats = CATS(nb_trials)

model = FSelection(cats.states, cats.actions, parameters)

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

# thr = extractStimulusPresentation(model.thr, model.state, model.action, model.responses)
# thr_free = extractStimulusPresentation(model.thr_free, model.state, model.action, model.responses)

step, indice = getRepresentativeSteps(human.reaction['fmri'], human.stimulus['fmri'], human.action['fmri'], human.responses['fmri'])
rt_fmri = computeMeanRepresentativeSteps(step) 

step, indice = getRepresentativeSteps(model.reaction, model.state, model.action, model.responses)
rt = computeMeanRepresentativeSteps(step)

rt = np.array(rt)
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y : (y - fitfunc(p, x))
p = leastsq(errfunc, [1.0, -1.0], args = (rt[0], rt_fmri[0]), full_output = False)
rt[0] = fitfunc(p[0], rt[0])
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

subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)

#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testSelection.pdf', bbox_inches='tight')
#savefig('/home/viejo/Desktop/figure_guillaume_a_tord.pdf', bbox_inches='tight')
show()
