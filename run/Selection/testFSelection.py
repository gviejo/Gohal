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
    x = x-np.median(x)
    x = x/(np.percentile(x,75)-np.percentile(x,25))
    return x

def testModel():
    model.startExp()
    for k in xrange(nb_repeat):
        for i in xrange(nb_blocs):
            cats.reinitialize()
            cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri']['S14'][i+1]['sar'][:,0]))
            model.startBloc()
            for j in xrange(nb_trials):
                sys.stdout.write("\r Bloc : %s | Trial : %i" % (i,j)); sys.stdout.flush()                    
                state = cats.getStimulus(j)
                action = model.chooseAction(state)
                reward = cats.getOutcome(state, action, case='fmri')
                model.updateValue(reward)
    
    rtm = np.array(model.reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
    state = convertStimulus(np.array(model.state)).reshape(nb_repeat, nb_blocs, nb_trials)
    action = np.array(model.action).reshape(nb_repeat, nb_blocs, nb_trials)
    responses = np.array(model.responses).reshape(nb_repeat, nb_blocs, nb_trials)
    for i in xrange(nb_repeat):
        rtm[i] = center(rtm[i])
        step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i])
        rt[i] = computeMeanRepresentativeSteps(step)[0]
    model.state = convertStimulus(np.array(model.state))
    model.action = np.array(model.action)
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

parameters = dict({'noise':0.06,
                    'length':10,
                    'alpha':1.0,
                    'beta':3.5,
                    'gamma':0.6,
                    'gain':2055.86,
                    'threshold':271.5,
                    'sigma':1.05})

with open("../Sferes/fmri/S14.pickle", "rb") as f:
    data = pickle.load(f)                      

nb_trials = 39
nb_blocs = 4
nb_repeat = 10
cats = CATS(nb_trials)

model = FSelection(cats.states, cats.actions, parameters)
rt = np.zeros((nb_repeat, 15))
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
# ax1.plot(range(1, len(rt_fmri[0])+1), rt_fmri[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
# ax1.errorbar(range(1, len(rt_fmri[0])+1), rt_fmri[0], rt_fmri[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
ax1.plot(data['mean'][0], linewidth=2, linestyle='--', color='grey', alpha=0.9)
# ax2 = ax1.twinx()
ax1.plot(np.mean(rt,0), linewidth = 2, linestyle = '-', color = 'black')
# ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
#ax2.set_ylabel("Inference Level")
#x2.set_ylim(-5, 15)
ax1.grid()
############

subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)

#savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testSelection.pdf', bbox_inches='tight')
#savefig('/home/viejo/Desktop/figure_guillaume_a_tord.pdf', bbox_inches='tight')
show()
