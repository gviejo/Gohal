#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
hand_search.py

for new model of fusion

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



# ------------------------------------

# ------------------------------------
# Parameter testing
# ------------------------------------
parameters= {'alpha': 0.5,
			 'beta': 5.0,
			 'gain': 6.0,
			 'gamma': 0.1,
			 'length': 7,
			 'noise': 0.0001,
			 'sigma': 1.0,
			 'threshold': 4.5,
			 'reward': 0.5}


# -----------------------------------
nb_blocs = 1
nb_trials = 40

cats = CATS(nb_trials)
model = FSelection(cats.states, cats.actions, parameters)



Hbs = []
Hfs = []

model.startExp()
for i in xrange(nb_blocs):
	Hbs.append([])
	Hfs.append([])
	cats.reinitialize()
	model.startBloc()
	# cats.set_devaluation_interval(5)
	for j in xrange(nb_trials):
		# print cats.asso				
		state = cats.getStimulus(j)				
		# print state
		action = model.chooseAction(state)
		Hbs[-1].append(model.Hb)
		Hfs[-1].append(model.Hf)
		# print action, cats.actions.index(action)
		reward = cats.getOutcome(state, action)				
		model.updateValue(reward)

		# print reward
		#sys.stdin.readline()

states = convertStimulus(np.array(model.state))
actions = np.array(model.action)
responses = np.array(model.responses)
reaction = np.array(model.reaction)
Hbs = np.array(Hbs)
Hfs = np.array(Hfs)

pcr = extractStimulusPresentation(responses, states, actions, responses)
step, indice = getRepresentativeSteps(reaction, states, actions, responses, 28)
rtm = computeMeanRepresentativeSteps(step)
rtm2 = extractStimulusPresentation(reaction, states, actions, responses)


fig1 = figure()

ax = fig1.add_subplot(3,2,1)
[ax.errorbar(range(len(pcr['mean'][i])), pcr['mean'][i], pcr['sem'][i]) for i in range(3)]

ax = fig1.add_subplot(3,2,2)
ax.plot(rtm[0], 'o-')

Hbmean = extractStimulusPresentation(Hbs, states, actions, responses)
Hfmean = extractStimulusPresentation(Hfs, states, actions, responses)


for i,c in zip([4,5,6],['blue','green','red']):
	ax = fig1.add_subplot(3,3,i)
	ax.plot(Hbmean['mean'][i-4], '-o', color = c)
	ax.plot(Hfmean['mean'][i-4], '--*', color = c)
	ax.set_ylim(0,model.max_entropy)


for i,c in zip([7,8,9],['blue','green','red']):
	ax = fig1.add_subplot(3,3,i)
	ax.plot(rtm2['mean'][i-7], '-o', color = c)
show()


# plt.savefig('devaluation_test.pdf')
# os.system("evince devaluation_test.pdf")
