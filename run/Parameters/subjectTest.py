#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test a dictionnary of parameters for each subject

run subjectTest.py -i kalman.txt -m kalman -s "S1"

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *

import matplotlib.pyplot as plt
from time import time
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the parameters file to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
parser.add_option("-s", "--subject", action="store", help="Which subject to plot \n Ex : -s S1", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def testParameters():
    p = eval(open(options.input, 'r').read())
    model.initializeList()
    for s in p.keys():
        for i in p[s].iterkeys():
            model.setParameter(i, p[s][i])

        for i in xrange(nb_blocs):
            sys.stdout.write("\r Sujet : %s | Blocs : %i" % (s,i)); sys.stdout.flush()                    
            cats.reinitialize()
            model.initialize()
            for j in xrange(nb_trials):
                state = cats.getStimulus(j)
                action = model.chooseAction(state)
                reward = cats.getOutcome(state, action)
                model.updateValue(reward)

    model.state = convertStimulus(np.array(model.state))
    model.action = convertAction(np.array(model.action))
    model.responses = np.array(model.responses)
    model.reaction = np.array(model.reaction)
    return p

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.630     # discount factor
init_cov = 10   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.6666   
alpha = 0.5
noise = 0.01
length_memory = 8
threshold = 1.2

nb_trials = human.responses['meg'].shape[1]
nb_blocs = 10
cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bwm_v1':BayesianWorkingMemory('v1', cats.states, cats.actions, length_memory, noise, threshold),
               'bwm_v2':BayesianWorkingMemory('v2', cats.states, cats.actions, length_memory, noise, threshold),
               'qlearning':QLearning('q', cats.states, cats.actions, gamma, alpha, beta)
               })

model = models[options.model]

# ------------------------------------
# Parameter testing
# ------------------------------------


t1 = time()
p = testParameters()
t2 = time()

print "\n"
print t2-t1
# -----------------------------------


# -----------------------------------
#order data
# -----------------------------------
pcr = extractStimulusPresentation(model.responses, model.state, model.action, model.responses)
pcr_human = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])

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
y_fmri = computeMeanRepresentativeSteps(step)



# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

# Probability of correct responses
figure(figsize = (9,4))
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
subplot(1,2,1)
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


ax1 = plt.subplot(1,2,2)
ax1.plot(range(1, len(rt_meg[0])+1), rt_meg[0], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)
ax1.errorbar(range(1, len(rt_meg[0])+1), rt_meg[0], rt_meg[1], linewidth = 2, linestyle = ':', color = 'grey', alpha = 0.9)

ax2 = ax1.twinx()
ax2.plot(range(1, len(rt[0])+1), rt[0], linewidth = 2, linestyle = '-', color = 'black')
ax2.errorbar(range(1,len(rt[0])+1), rt[0], rt[1], linewidth = 2, linestyle = '-', color = 'black')
ax2.set_ylabel("Inference Level")
ax2.set_ylim(-5, 15)
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


################
subplots_adjust(left = 0.08, wspace = 0.3, hspace = 0.35, right = 0.86)
savefig('../../../Dropbox/ISIR/B2V_council/images/fig_subject'+options.model+'.pdf', bbox_inches='tight')


show()










