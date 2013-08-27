#!/usr/bin/python
# encoding: utf-8
"""
scripts to plot figure pour le rapport IAD
figure3 : evolution de w / correlation /performances

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from ColorAssociationTasks import CATS_MODELS
from HumanLearning import HLearning
from Models import *
from Sweep import Sweep_performances
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
        m.updateValue(reward)
# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------


# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
gamma = 0.6      # discount factor
init_cov = 10    # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters
beta = 1.7
length_memory = 10
noise_width = 0.01
correlation = "Z"

nb_trials = 42
nb_blocs = 42

cats = CATS(nb_trials)

models = dict({'kalman':KalmanQLearning('kalman', cats.states, cats.actions, gamma, beta, eta, var_obs, init_cov, kappa),
               'bmw':BayesianWorkingMemory('bmw', cats.states, cats.actions, length_memory, noise_width, 1.0)})

cats = CATS_MODELS(nb_trials, models.keys())

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
sweep = Sweep_performances(human, cats, nb_trials, nb_blocs)
data = dict()
data['human'] = extractStimulusPresentation2(human.responses['meg'], human.stimulus['meg'], human.action['meg'], human.responses['meg'])


# -----------------------------------

w = dict({1:[],2:[],3:[]})
fall = dict({'kalman':dict({1:[],
                            2:[],
                            3:[]}),
             'bmw':dict({1:[],
                         2:[],
                         3:[]})})
sim = dict({'kalman':dict({1:[],
                            2:[],
                            3:[]}),
             'bmw':dict({1:[],
                         2:[],
                         3:[]})})


for l in xrange(10):
        print l
        [m.initializeList() for m in models.itervalues()]
	# -----------------------------------
	# Learning
	# -----------------------------------
        for i in xrange(nb_blocs):
	    #sys.stdout.write("\r Blocs : %i " % i); sys.stdout.flush()                        
	    cats.reinitialize()
	    [m.initialize() for m in models.itervalues()]
	    for j in xrange(nb_trials):
		iterationStep(j, models, False)

	# -----------------------------------



	# -----------------------------------
	# Comparison with Human Learnig
	# -----------------------------------
	for m in models.itervalues():
	    m.state = convertStimulus(np.array(m.state))
	    m.action = convertAction(np.array(m.action))
	    m.responses = np.array(m.responses)
	    data[m.name] = extractStimulusPresentation2(m.responses, m.state, m.action, m.responses)

        for m in ['kalman', 'bmw']:
            for i in [1,2,3]:
                tmp = np.mean(data[m][i], axis = 0)
                fall[m][i].append(tmp)

	f = dict()  
	for i in [1,2,3]:
	    tmp = []
	    for j in xrange(len(data.keys())):
		f[data.keys()[j]] = j
		tmp.append(np.mean(data[data.keys()[j]][i], axis = 0))
	    f[i] = np.array(tmp)
	# -----------------------------------

	# -----------------------------------
        # Computation of similiraty
	# -----------------------------------
        for m in ['kalman', 'bmw']:
            tmp = sweep.computeCorrelation(data[m], correlation)
            for i in [1, 2, 3]:
                sim[m][i].append(tmp[i])
            

	# -----------------------------------

	# -----------------------------------
	# Computation of weight
	# -----------------------------------
	for i in [1,2,3]:
	    tmp = f[i]
            w[i].append([])
	    for j in xrange(tmp.shape[1]):
		if tmp[0,j] == tmp[1,j] == tmp[2,j] == 0.0:
		    w[i][-1].append(0.0)
		else:
		    v = (tmp[2,j]-tmp[1,j])/(tmp[0,j]-tmp[1,j])
		    if v < 0.0:
			w[i][-1].append(0.0)
		    elif v > 1.0:
			w[i][-1].append(1.0)
		    else:
			w[i][-1].append(v)
        

        
	# -----------------------------------
mw = np.array([np.mean(w[i], 0) for i in [1,2,3]])
sw = np.array([np.var(w[i], 0) for i in [1,2,3]])

mk = np.array([np.mean(fall['kalman'][i], 0) for i in [1,2,3]])
sk = np.array([np.var(fall['kalman'][i], 0) for i in [1,2,3]])
mb = np.array([np.mean(fall['bmw'][i], 0) for i in [1,2,3]])
sb = np.array([np.var(fall['bmw'][i], 0) for i in [1,2,3]])
mh = np.array([np.mean(data['human'][i], 0) for i in [1,2,3]])
msk = np.array([np.mean(sim['kalman'][i], 0) for i in [1, 2, 3]])
ssk = np.array([np.var(sim['kalman'][i], 0) for i in [1, 2, 3]])
msb = np.array([np.mean(sim['bmw'][i], 0) for i in [1, 2,3]])
ssb = np.array([np.var(sim['bmw'][i], 0) for i in [1,2,3]])

# -----------------------------------
# Plot
# -----------------------------------
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}
#rcParams.update(params)
dashes = ['-', '--', ':']

fig = figure(figsize=(10,10))

subplot(1,1,1)
for i in range(3):
    plot(range(1, len(mw[i])+1), mw[i], linestyle = dashes[i], linewidth = 2, color = 'black', label = 'Stim '+str(i))
    errorbar(range(1, len(mw[i])+1), mw[i], sw[i], linestyle = dashes[i], linewidth = 2, color = 'black')
ylim(0.0, 1.05)
ylabel("w", fontsize = 11)
xlabel("Trial")
yticks(np.arange(0, 1.2, 0.2))
xticks(range(2,11,2))
xlim(0.8, 10.2)
grid()

legend(loc = 'lower right')

#subplots_adjust(left = 0.09, wspace = 0.25, hspace = 0.25, right = 0.86)
#fig.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig6.pdf', bbox_inches='tight')
show()




