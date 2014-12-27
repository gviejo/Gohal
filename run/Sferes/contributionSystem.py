#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and test a dictionnary of parameters for each subject

run subjectTest.py -i data

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
from Selection import *
from matplotlib import *
from pylab import *
import pickle
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import leastsq
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------


# -----------------------------------
# FONCTIONS
# -----------------------------------
def _convertStimulus(s):
		return (s == 1)*'s1'+(s == 2)*'s2' + (s == 3)*'s3'

fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y : (y - fitfunc(p, x))

def leastSquares(x, y):
	for i in xrange(len(x)):
		pinit = [1.0, -1.0]
		p = leastsq(errfunc, pinit, args = (x[i], y[i]), full_output = False)
		x[i] = fitfunc(p[0], x[i])
	return x    

def center(x, o, s, m):    
    x = x-timing[o][s][m][0]
    x = x/timing[o][s][m][1]
    return x

# -----------------------------------

# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nb_blocs = 4
nb_trials = 39
nb_repeat = 10
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions),
				"qlearning":QLearning(cats.states, cats.actions),
				"bayesian":BayesianWorkingMemory(cats.states, cats.actions),
				"selection":KSelection(cats.states, cats.actions),
				"mixture":CSelection(cats.states, cats.actions)})

# ------------------------------------
# Parameter testing
# ------------------------------------
with open("parameters.pickle", 'r') as f:
  p_test = pickle.load(f)

with open("timing.pickle", 'rb') as f:
    timing = pickle.load(f)


groups = {}
for o in p_test.iterkeys():
	groups[o] = dict()
	for s in p_test[o].iterkeys():
		m = p_test[o][s].keys()[0]
		if m in groups[o].keys():
			groups[o][m].append(s)
		else:
			groups[o][m] = [s]

colors = {'owa':'r','distance':'b','tche':'g'}

colors_m = dict({'fusion':'#F1433F',
				'bayesian': '#D5A253',
				'qlearning': '#6E8243',
				'selection':'#70B7BA',
				'mixture':'#3D4C53'})


data = {}


for o in p_test.iterkeys():
	data[o] = {'Hb':{},'Hf':{}}
	entropy = dict()
	for g in groups[o].iterkeys():
		entropy[g] = dict({'Hb':{}, 'Hf':{}, 'N':{}})			
		for s in groups[o][g]:			
			m = p_test[o][s].keys()[0]
			print "Testing "+s+" with "+m+" selected by "+o+" in group "+g
			models[m].setAllParameters(p_test[o][s][m])	        
			models[m].startExp()
			for k in xrange(nb_repeat):
				for i in xrange(nb_blocs):
					cats.reinitialize()
					cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
					models[m].startBloc()
					for j in xrange(nb_trials):
						state = cats.getStimulus(j)
						action = models[m].chooseAction(state)
						reward = cats.getOutcome(state, action, case='fmri')
						models[m].updateValue(reward)                                    
			# rtm = np.array(models[m].reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
			# state = convertStimulus(np.array(models[m].state)).reshape(nb_repeat, nb_blocs, nb_trials)
	  #       action = np.array(models[m].action).reshape(nb_repeat, nb_blocs, nb_trials)
	  #       responses = np.array(models[m].responses).reshape(nb_repeat, nb_blocs, nb_trials)
	  #       tmp = np.zeros((nb_repeat, 15))
	  #       for i in xrange(nb_repeat):
	  #           rtm[i] = center(rtm[i], o, s, m)
   #  	        step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i])
   #      	    tmp[i] = computeMeanRepresentativeSteps(step)[0]        			
			state = convertStimulus(np.array(models[m].state))
			action = np.array(models[m].action)
			responses = np.array(models[m].responses)        
			hall = np.array(models[m].Hall)
			N = np.array(models[m].pdf)

			data[o]['Hb'][s] = {m:extractStimulusPresentation(hall[:,:,0], state, action, responses)}        
			data[o]['Hf'][s] = {m:extractStimulusPresentation(hall[:,:,1], state, action, responses)}

			entropy[g]['Hb'][s] = extractStimulusPresentation(hall[:,:,0], state, action, responses)
			entropy[g]['Hf'][s] = extractStimulusPresentation(hall[:,:,1], state, action, responses)
			entropy[g]['N'][s] = extractStimulusPresentation(N, state, action, responses)
	
	fig = figure()

	axes = {'Hb':{i:fig.add_subplot(2,3,i+1) for i in xrange(3)}, 'Hf':{i:fig.add_subplot(2,3,i+4) for i in xrange(3)}}

	for g in entropy.iterkeys():
		for h in ['Hb', 'Hf']:
			for s in entropy[g][h].iterkeys():
				for i in xrange(3):
					y = entropy[g][h][s]['mean'][i]
					x = range(1, len(y)+1)
					e = entropy[g][h][s]['sem'][i]
					axes[h][i].plot(x, y, linewidth = 2, label = s, color = colors_m[g]) 
					axes[h][i].fill_between(x, y-e, y+e, facecolor = colors_m[g], alpha = 0.1)
					axes[h][i].set_ylim(0,np.log2(5))
					# if h == 'Hb':
					# 	ax2 = axes[h][i].twinx()
					# 	ax2.plot(x, entropy[g]['N'][s]['mean'][i], '--', color = colors_m[g], linewidth = 3)
	legend()
	show()
		# meanHall = dict()
		# for h in entropy.keys():
		#     meanHall[h] = dict()
		#     model = np.unique([entropy[h][s].keys()[0] for s in entropy[h].iterkeys()])
		#     for m in model:
		#         subject = [s for s in entropy[h].keys() if entropy[h][s].keys()[0] == m]
		#         if len(subject) == 1:
		#             meanHall[h][m] = entropy[h][subject[0]][m]
		#         else:
		#             tmp = np.array([entropy[h][s][m]['mean'] for s in subject])
		#             meanHall[h][m] = {'mean':np.mean(tmp,0), 'sem':sem(tmp,0)}

		# fig = figure(figsize = (9,5))

		# ax2 = fig.add_subplot(1,3,2)
		# # for s in entropy['Hb'].iterkeys():
		#     # m = entropy['Hb'][s].keys()[0]
		#     # tmp = entropy['Hb'][s][m]
		# for m in meanHall['Hb'].iterkeys():
		#     tmp = meanHall['Hb'][m]    
		#     for i in xrange(3):
		#         x = range(1, len(tmp['mean'][i])+1)
		#         y = tmp['mean'][i]
		#         ax2.plot(x, y, linewidth=1.5, color = colors_m[m])        
		#         ax2.fill_between(x, y-tmp['sem'][i], y+tmp['sem'][i], facecolor = colors_m[m], alpha = 0.5)
		# ax2.set_ylim(0,np.log2(5))
		# ax3 = fig.add_subplot(1,3,3)
		# # for s in entropy['Hf'].iterkeys():
		# #     m = entropy['Hf'][s].keys()[0]
		# #     tmp = entropy['Hf'][s][m]
		# for m in meanHall['Hf'].iterkeys():
		#     tmp = meanHall['Hf'][m]
		#     for i in xrange(3):
		#         x = range(1, len(tmp['mean'][i])+1)
		#         y = tmp['mean'][i]
		#         ax3.plot(x, y, linewidth=1.5, color = colors_m[m])        
		#         ax3.fill_between(x, y-tmp['sem'][i], y+tmp['sem'][i], facecolor = colors_m[m], alpha = 0.5)
		# ax3.set_ylim(0,np.log2(5))
		# show()

			
			