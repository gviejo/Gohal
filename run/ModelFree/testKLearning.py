#!/usr/bin/python
# encoding: utf-8
"""
Test for Klearning :
Plot probability of correct responses and reaction time

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import KalmanQLearning
from matplotlib import *
from pylab import *
from time import time

# -----------------------------------
# FONCTIONS
# -----------------------------------
def testModel():
    kalman.startExp()
    for i in xrange(nb_blocs):
        cats.reinitialize()
        kalman.startBloc()
        for j in xrange(nb_trials):
            sys.stdout.write("\r Bloc : %s | Trial : %i" % (i,j)); sys.stdout.flush()
            state = cats.getStimulus(j)
            action = kalman.chooseAction(state)
            reward = cats.getOutcome(state, action)
            kalman.updateValue(reward)
    kalman.state = convertStimulus(np.array(kalman.state))
    kalman.action = convertAction(np.array(kalman.action))
    kalman.responses = np.array(kalman.responses)
    kalman.reaction = np.array(kalman.reaction)
    
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
parameters = dict({'gamma':0.43,
                    'eta':0.0001,
                    'beta':2.2})

nb_trials = 42
nb_blocs = 100
cats = CATS()

kalman = KalmanQLearning(cats.states, cats.actions, parameters)

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
pcr = extractStimulusPresentation(kalman.responses, kalman.state, kalman.action, kalman.responses)
pcr_human = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])


entropy = extractStimulusPresentation(kalman.reaction, kalman.state, kalman.action, kalman.responses)

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
    plot(range(1, len(entropy['mean'][i])+1), entropy['mean'][i], linewidth = 2, linestyle = '-', color = colors[i], label= 'Stim '+str(i+1))    
    errorbar(range(1, len(entropy['mean'][i])+1), entropy['mean'][i], entropy['sem'][i], linewidth = 2, linestyle = '-', color = colors[i])
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

savefig('../../../Dropbox/ISIR/JournalClub/images/fig_testKQL.pdf', bbox_inches='tight')

show()
