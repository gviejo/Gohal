#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
devaluation.py

1) load the parameters from the sferes folder

2) generate new block with devaluation

3) test for each set of parameters

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
nb_blocs = 10
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

fig1 = figure()




ind = 1
for d in devaluation_time:
	model.startExp()	
	for s in list(set(p_test[operator].keys())-set(['S2'])):
	# for s in p_test[operator].keys():
	# for s in ['S9']:
		model.setAllParameters(p_test[operator][s]['fusion'])
		for i in xrange(nb_blocs):
			cats.reinitialize()
			cats.set_devaluation_interval(d)
			model.startBloc()
			for j in xrange(nb_trials):
				# print cats.asso				
				state = cats.getStimulus(j)				
				# print state
				action = model.chooseAction(state)
				# print action, cats.actions.index(action)
				reward = cats.getOutcome(state, action)				
				model.updateValue(reward)

				# print reward
				#sys.stdin.readline()

	states = convertStimulus(np.array(model.state))
	actions = np.array(model.action)
	responses = np.array(model.responses)
	reaction = np.array(model.reaction)

	pcr = extractStimulusPresentation(responses, states, actions, responses)
	step, indice = getRepresentativeSteps(reaction, states, actions, responses, 28)
	rtm = computeMeanRepresentativeSteps(step)
	rtm2 = extractStimulusPresentation(reaction, states, actions, responses)
	# hbs = extractStimulusPresentation(np.array(model.Hbs), states, actions, responses)
	# hfs = extractStimulusPresentation(np.array(model.Hfs), states, actions, responses)

	ax = fig1.add_subplot(len(devaluation_time),2,ind)
	[ax.errorbar(range(len(pcr['mean'][i])), pcr['mean'][i], pcr['sem'][i]) for i in range(3)]
	# ax = fig2.add_subplot(len(devaluation_time),3,ind)
	# ax.plot(range(len(hbs['mean'][0])), hbs['mean'][0], 'o-')#, hbs['sem'][0], color = 'red')
	# ax.plot(range(len(hfs['mean'][0])), hfs['mean'][0], 'o--')#, hfs['sem'][0], color = 'red', linestyle = '--')
	# ax.set_ylim(0,-np.log2(0.2))
	# ax2 = fig2.add_subplot(2,3,ind+3)
	# ax2.plot(range(len(rtm2['mean'][0])), rtm2['mean'][0], '*-')
	ind+=1
	ax = fig1.add_subplot(len(devaluation_time),2,ind)
	ax.errorbar(range(len(rtm[0])), rtm[0], rtm[1])
	ind+=1

	# ax = fig2.add_subplot(2,3,ind)
	# ax.plot(range(len(hbs['mean'][1])), hbs['mean'][1], 'o-')#, hbs['sem'][1], color = 'green')
	# ax.plot(range(len(hfs['mean'][1])), hfs['mean'][1], 'o--')#, hfs['sem'][1], color = 'green')
	# ax.set_ylim(0,-np.log2(0.2))
	# ax2 = fig2.add_subplot(2,3,ind+3)
	# ax2.plot(range(len(rtm2['mean'][1])), rtm2['mean'][1], '*-')

	# ax = fig2.add_subplot(2,3,ind)
	# ax.plot(range(len(hbs['mean'][2])), hbs['mean'][2], 'o-')#, hbs['sem'][2], color = 'blue')
	# ax.plot(range(len(hfs['mean'][2])), hfs['mean'][2], 'o--')#, hfs['sem'][2], color = 'blue')
	# ax.set_ylim(0,-np.log2(0.2))
	# ax2 = fig2.add_subplot(2,3,ind+3)
	# ax2.plot(range(len(rtm2['mean'][2])), rtm2['mean'][2], '*-')
	
	


plt.savefig('devaluation.pdf')
os.system("evince devaluation.pdf")
