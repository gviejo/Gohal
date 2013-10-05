#!/usr/bin/python
# encoding: utf-8
"""
New model selection for Brovelli task based on Keramati
Modification are :
      - reward rate depends on the stimulus
      - threshold for the number of inferences in the bayesian working memory
      
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *
from Sweep import Optimization
from Selection import KSelection
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

def testModel():
    for i in xrange(nb_blocs):
        sys.stdout.write("\r Testing model | Blocs : %i" % i); sys.stdout.flush()                       
        cats.reinitialize()
        selection.initialize()
        for j in xrange(nb_trials):
            state = cats.getStimulus(j)
            action = selection.chooseAction(state)
            reward = cats.getOutcome(state, action)
            selection.updateValue(reward)
    selection.state = convertStimulus(np.array(selection.state))
    selection.action = convertAction(np.array(selection.action))
    selection.responses = np.array(selection.responses)
    selection.rrate = np.array(selection.rrate)
    selection.vpi = np.array(selection.vpi)
    selection.model_used = np.array(selection.model_used)
    selection.n_inf = np.array(selection.n_inf)
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001            # variance of evolution noise v
var_obs = 0.05          # variance of observation noise n
gamma = 0.95            # discount factor
init_cov = 10           # initialisation of covariance matrice
kappa = 0.1             # unscentered transform parameters
beta = 5.5              # temperature for kalman soft-max
noise = 0.000            # variance of white noise for working memory
length_memory = 100     # size of working memory
threshold = 0.2         # inference threshold
sigma = 0.00002           # updating rate of the average reward


nb_trials = human.responses['meg'].shape[1]
nb_blocs = human.responses['meg'].shape[0]


cats = CATS(nb_trials)

selection = KSelection(KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
                       BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise, threshold),
                       sigma)
                       
# -----------------------------------

# -----------------------------------
# SESSION MODELS
# -----------------------------------
testModel()
# -----------------------------------

# -----------------------------------
#order data
# -----------------------------------
data = dict()
data['meg'] = extractStimulusPresentation(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])
data['keramati'] = extractStimulusPresentation(selection.responses, selection.state, selection.action, selection.responses) 
data['used'] = extractStimulusPresentation(selection.model_used, selection.state, selection.action, selection.responses)
data['r'] = extractStimulusPresentation(selection.rrate, selection.state, selection.action, selection.responses)
tmp =getRepresentativeSteps(selection.n_inf, selection.state, selection.action, selection.responses)
data['rt'] = computeMeanRepresentativeSteps(tmp[0])


data['vpi'] = dict()
action = extractStimulusPresentation2(selection.action-1, selection.state, selection.action, selection.responses)
#state = extractStimulusPresentation2(np.tile(np.arange(1,nb_trials+1), (nb_blocs,1)), selection.state, selection.action, selection.responses)
state = extractStimulusPresentation2(selection.state, selection.state, selection.action, selection.responses)
#first stimulus

data['vpi']['s1'] = dict({i:np.mean(np.reshape(selection.vpi[:,:,action[1][:,i-1]][:,:,0][selection.state == np.vstack(state[1][:,0])], (nb_blocs, 14))[:,0:10], axis = 0) for i in [1,2]})
# second stimulus
data['vpi']['s2'] = dict({i:np.mean(np.reshape(selection.vpi[:,:,action[2][:,i-1]][:,:,0][selection.state == np.vstack(state[2][:,0])], (nb_blocs, 14))[:,0:10], axis = 0) for i in [1,2,3,4]})
# third stimulus
data['vpi']['s3'] = dict({i:np.mean(np.reshape(selection.vpi[:,:,action[3][:,i-1]][:,:,0][selection.state == np.vstack(state[3][:,0])], (nb_blocs, 14))[:,0:10], axis = 0) for i in [1,2,3,4,5]})


# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------
ion()
fig = figure(figsize=(14, 8))
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':10,
          'ytick.labelsize':10,
          'text.usetex':False}
dashes = ['-', '--', ':']
rcParams.update(params)

for i in xrange(3):
    subplot(3,3,i+1)
    plot(range(1, len(data['keramati']['mean'][i])+1), data['keramati']['mean'][i], linewidth = 2, color = 'black', label = 'simulation')
    errorbar(range(1, len(data['keramati']['mean'][i])+1), data['keramati']['mean'][i], data['keramati']['sem'][i], linewidth = 2, color = 'black')
    plot(range(1, len(data['meg']['mean'][i])+1), data['meg']['mean'][i], linewidth = 2, color = 'black', linestyle = '--', label = 'MEG')
    errorbar(range(1, len(data['meg']['mean'][i])+1), data['meg']['mean'][i], data['meg']['sem'][i], linewidth = 2, color = 'black', linestyle = '--')
    legend(loc = 'lower right')
    grid()
    ylim(0,1)
    xlim(0,10)
    ylabel("Probability Correct Response")
    title("Stimulus "+str(i+1))

for i,j in zip([4,5,6], xrange(3)):
    subplot(3,3,i)
    plot(range(1, len(data['r']['mean'][j])+1),data['r']['mean'][j], linewidth = 2, linestyle = '--')
    for k in data['vpi']['s'+str(j+1)]:
        plot(range(1, len(data['vpi']['s'+str(j+1)][k])+1),data['vpi']['s'+str(j+1)][k], linewidth = 2, label = "VPI("+str(k)+")")
    grid()
    ylabel("Decision", fontsize = 12)
    xlim(0,10)
    legend()

for i,j in zip([7,8,9], xrange(3)):
    subplot(3,3,i)
    plot(range(1, len(data['used']['mean'][j])+1),data['used']['mean'][j], 'o-', linewidth = 2)
    grid()
    xlim(0,10)
    legend()
    ylabel("$N^{Based}_a / (N^{Free}_a+N^{Based}_a$", fontsize = 12)
    xlabel("Trials", fontsize = 12)
subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86, hspace = 0.35)
"""
subplot(4,2,7)
#rrorbar(range(1, len(data['rt'][0])+1), data['rt'][0], data['rt'][1], 'o-', linewidth = 2)
plot(range(1, len(data['rt'][0])+1), data['rt'][0], 'o-', linewidth = 2)
xlabel("Representative Steps")
ylabel("$Length * R(B/F)r")
grid()
"""
figtext(0.04, 0.92, 'A', fontsize = 15)
figtext(0.04, 0.61, 'B', fontsize = 15)
figtext(0.04, 0.33, 'C', fontsize = 15)

#fig.savefig('../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig8.pdf', bbox_inches='tight')
show()
