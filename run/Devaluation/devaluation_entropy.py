#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
devaluation_entropy.py

1) load the parameters from the sferes folder

2) generate new block with devaluation

3) test for a few subjects and plot the entropy

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from Selection import *
from matplotlib import *
from pylab import *
import pickle
import matplotlib.pyplot as plt


# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 1
nb_trials = 80

cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
                "qlearning":QLearning(cats.states, cats.actions),
                "bayesian":BayesianWorkingMemory(cats.states, cats.actions),
                "selection":KSelection(cats.states, cats.actions),
                "mixture":CSelection(cats.states, cats.actions)})
# ------------------------------------

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("../Sferes/parameters.pickle", 'r') as f:
	p_test = pickle.load(f)
# tmp = dict({k:[] for k in p_test['distance']['S9']['fusion'].keys()})
# for o in p_test.iterkeys():
# 	for s in p_test[o].iterkeys():
# 		for m in p_test[o][s].iterkeys():
# 			for p in p_test[o][s][m].iterkeys():
# 				tmp[p].append(p_test[o][s][m][p])
# fig3 = figure()
# ind = 1
# for p in tmp.keys():
# 	ax = fig3.add_subplot(len(tmp.keys()),1,ind)
# 	ax.hist(tmp[p])
# 	ax.set_title(p)
# 	ind+=1

# -----------------------------------
operator = 'owa'
model = models['fusion']
devaluation_time = [2,8,16]
colors = ['blue', 'red', 'green']
alpha = 0.6
fig1 = figure()
s_to_plot = np.random.choice(list(set(p_test[operator].keys())-set(['S2'])), 4)
subplot_positions = np.arange(1,len(devaluation_time)*len(s_to_plot)+1).reshape(len(s_to_plot),len(devaluation_time))


for d in xrange(len(devaluation_time)):
	ax = {k:subplot(3,len(devaluation_time),d+1+k*len(devaluation_time)) for k in xrange(3)}	
	for s in xrange(len(s_to_plot)):
		Hb = []
		Hf = []
		N = []
		model.startExp()
		model.setAllParameters(p_test[operator][s_to_plot[s]]['fusion'])
		cats.reinitialize()
		cats.set_devaluation_interval(devaluation_time[d])
		model.startBloc()
		for j in xrange(nb_trials):
			state = cats.getStimulus(j)				
			action = model.chooseAction(state)
			reward = cats.getOutcome(state, action)				
			model.updateValue(reward)
			Hf.append(model.Hf)
			N.append(model.nb_inferences)
			Hb.append(model.Hb)			
			#sys.stdin.readline()

		states = convertStimulus(np.array(model.state))
		actions = np.array(model.action)
		responses = np.array(model.responses)
		Hbs = extractStimulusPresentation(np.array([Hb]),states,actions,responses)
		Hfs = extractStimulusPresentation(np.array([Hf]),states,actions,responses)
		
		[ax[i].plot(Hbs['mean'][i], 'o-', color = colors[i], alpha = alpha) for i in range(3)]
		[ax[i].plot(Hfs['mean'][i], '*--', color = colors[i], alpha = alpha) for i in range(3)]
	

plt.savefig('devaluation_entropy.pdf')
os.system("evince devaluation_entropy.pdf")