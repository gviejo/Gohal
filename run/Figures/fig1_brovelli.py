#!/usr/bin/python
# encoding: utf-8
"""
scripts to plot figure pour le rapport IAD
figure 1 : performances des sujets / RT on representatieve steps

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
#from pylab import *
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from HumanLearning import HLearning
from Models import QLearning
from Models import KalmanQLearning
from Models import TreeConstruction
from matplotlib import *
from pylab import *
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def iterationStep(iteration, models, display = True):
    state = cats.getStimulus(iteration)

    for m in models.itervalues():
        action = m.chooseAction(state)
        reward = cats.getOutcome(state, action, m.name)
        if m.__class__.__name__ == 'TreeConstruction':
            m.updateTrees(state, reward)
        else:
            m.updateValue(reward)


# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------

# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
# -----------------------------------


# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
#order data
# -----------------------------------
data = dict()
data['pcr'] = dict()
for i in human.directory.keys():
    print i
    data['pcr'][i] = extractStimulusPresentation(human.responses[i], human.stimulus[i], human.action[i], human.responses[i])
# -----------------------------------
data['rt'] = dict()
for i in human.directory.keys():
    print i
    data['rt'][i] = dict()
    step, indice = getRepresentativeSteps(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
    data['rt'][i]['mean'], data['rt'][i]['sem'] = computeMeanRepresentativeSteps(step) 
# -----------------------------------
data['rt2'] = dict()
for i in human.directory.keys():
    print i
    data['rt2'][i] = extractStimulusPresentation(human.reaction[i], human.stimulus[i], human.action[i], human.responses[i])
    
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

fig1 = figure(figsize=(8, 4))
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}
          
#rcParams.update(params)                  
subplot(121)
m = data['pcr']['meg']['mean']
s = data['pcr']['meg']['sem']
dashes = ['-', '--', ':']
colors = ['blue','red','green']
for i in xrange(3):
    errorbar(range(1, len(m[i])+1), m[i], s[i], linestyle = dashes[0], color = colors[i])
    plot(range(1, len(m[i])+1), m[i], linestyle = dashes[0], color = colors[i], linewidth = 2, label = 'Stim '+str(i+1))
grid()
legend(loc = 'lower right')
xticks(range(2,11,2))
xlabel("Trial")
xlim(0.8, 10.2)
ylim(-0.05, 1.05)
yticks(np.arange(0, 1.2, 0.2))
ylabel('Probability Correct Responses')
title('A')

subplot(122)
m = data['rt']['meg']['mean']
s = data['rt']['meg']['sem']
errorbar(range(1, len(m)+1), m, s, color = 'black', marker = 'o')
###
msize = 8.0
mwidth = 2.5
plot(1, 0.455, 'x', color = 'blue', markersize=msize, markeredgewidth=mwidth)
plot(1, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
plot(1, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
plot(2, 0.455, 'o', color = 'blue', markersize=msize)
plot(2, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
plot(2, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
plot(3, 0.4445, 'x', color = 'red', markersize=msize,markeredgewidth=mwidth)
plot(3, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
plot(4, 0.4445, 'o', color = 'red', markersize=msize)
plot(4, 0.435, 'x', color = 'green', markersize=msize,markeredgewidth=mwidth)
plot(5, 0.435, 'o', color = 'green', markersize=msize)
for i in xrange(6,16,1):
    plot(i, 0.455, 'o', color = 'blue', markersize=msize)
    plot(i, 0.4445, 'o', color = 'red', markersize=msize)
    plot(i, 0.435, 'o', color = 'green', markersize=msize)

###
grid()
xlabel("Representative steps")
xticks([1,5,10,15])
yticks([0.46, 0.50, 0.54])
ylim(0.43, 0.56)
ylabel("Reaction time (s)")
title('B')



subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86)
fig1.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig1.pdf', bbox_inches='tight')
show()
