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

o = 'tche'

groups = {}
for s in p_test[o].iterkeys():
	m = p_test[o][s].keys()[0]
	if m in groups.keys():
		groups[m].append(s)
	else:
		groups[m] = [s]

colors = {'owa':'r','distance':'b','tche':'g'}

colors_m = dict({'fusion':'#F1433F',
				'bayesian': '#D5A253',
				'qlearning': '#6E8243',
				'selection':'#70B7BA',
				'mixture':'#3D4C53'})
data = {}
# Start with selection 
# m = 'selection'
# for s in groups['selection']:
# 	print "Testing "+s+" with "+m+" selected by "+o
# 	models[m].setAllParameters(p_test[o][s][m])
# 	models[m].startExp()
# 	count = np.zeros((nb_repeat,nb_blocs,nb_trials))
# 	for k in xrange(nb_repeat):
# 		for i in xrange(nb_blocs):
# 			cats.reinitialize()
# 			cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
# 			models[m].startBloc()			
# 			for j in xrange(nb_trials):
# 				state = cats.getStimulus(j)
# 				action = models[m].chooseAction(state)
# 				count[k,i,j] = models[m].used
# 				reward = cats.getOutcome(state, action, case='fmri')
# 				models[m].updateValue(reward)                                    
# 	state = convertStimulus(np.array(models[m].state))
# 	action = np.array(models[m].action)
# 	responses = np.array(models[m].responses)        
# 	count = count.reshape(nb_blocs*nb_repeat, nb_trials)
# 	data[s] = extractStimulusPresentation(count, state, action, responses)
# 	# step, indice = getRepresentativeSteps(count, state, action, responses)
# 	# data[s] = computeMeanRepresentativeSteps(step)
# 	sys.exit()
# # Then mixture
# m = 'mixture'
# for s in groups['mixture']:
# 	print "Testing "+s+" with "+m+" selected by "+o
# 	models[m].setAllParameters(p_test[o][s][m])
# 	models[m].startExp()
# 	weight = np.zeros((nb_repeat,nb_blocs,nb_trials))
# 	for k in xrange(nb_repeat):
# 		for i in xrange(nb_blocs):
# 			cats.reinitialize()
# 			cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
# 			models[m].startBloc()			
# 			for j in xrange(nb_trials):
# 				state = cats.getStimulus(j)
# 				action = models[m].chooseAction(state)
# 				weight[k,i,j] = models[m].w[models[m].current_state]
# 				reward = cats.getOutcome(state, action, case='fmri')
# 				models[m].updateValue(reward)                                    
# 	state = convertStimulus(np.array(models[m].state))
# 	action = np.array(models[m].action)
# 	responses = np.array(models[m].responses)        
# 	weight = weight.reshape(nb_blocs*nb_repeat, nb_trials)
# 	weight = (weight*2.0)-1.0
# 	data[s] = extractStimulusPresentation(weight, state, action, responses)
# 	# step, indice = getRepresentativeSteps(weight, state, action, responses)
# 	# data[s] = computeMeanRepresentativeSteps(step)

# And fusion to finish
m = 'fusion'
for s in groups['fusion']:
	print "Testing "+s+" with "+m+" selected by "+o
	models[m].setAllParameters(p_test[o][s][m])
	models[m].startExp()
	entropy = np.zeros((nb_repeat,nb_blocs,nb_trials))
	for k in xrange(nb_repeat):
		for i in xrange(nb_blocs):
			cats.reinitialize()
			cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
			models[m].startBloc()			
			for j in xrange(nb_trials):
				state = cats.getStimulus(j)
				action = models[m].chooseAction(state)
				entropy[k,i,j] = models[m].Hf-models[m].Hb
				reward = cats.getOutcome(state, action, case='fmri')
				models[m].updateValue(reward)                                    
	state = convertStimulus(np.array(models[m].state))
	action = np.array(models[m].action)
	responses = np.array(models[m].responses)        
	entropy = entropy.reshape(nb_blocs*nb_repeat, nb_trials)
	sys.exit()
	entropy = (entropy+models[m].max_entropy)/(2*models[m].max_entropy)
	entropy = (entropy*2.0)-1.0
	data[s] = extractStimulusPresentation(entropy, state, action, responses)
	# step, indice = getRepresentativeSteps(entropy, state, action, responses)
	# data[s] = computeMeanRepresentativeSteps(step)	

# figure()
# for i in xrange(3):
# 	subplot(1,3,i+1)
# for s in data.keys():
# 	plot(data[s][0])
# show()

data['groups'] = groups

with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_contribution_dual_models.pickle") , 'wb') as handle:    
    pickle.dump(data, handle)



# # for o in p_test.iterkeys():
# for o in ['tche']:
# 	data[o] = {'Hb':{},'Hf':{}}
# 	entropy = dict()
# 	for g in groups[o].iterkeys():
# 		entropy[g] = dict({'Hb':{}, 'Hf':{}, 'N':{}})			
# 		for s in groups[o][g]:			
# 			m = p_test[o][s].keys()[0]
# 			print "Testing "+s+" with "+m+" selected by "+o+" in group "+g
# 			models[m].setAllParameters(p_test[o][s][m])	        
# 			models[m].startExp()
# 			for k in xrange(nb_repeat):
# 				for i in xrange(nb_blocs):
# 					cats.reinitialize()
# 					cats.stimuli = np.array(map(_convertStimulus, human.subject['fmri'][s][i+1]['sar'][:,0]))
# 					models[m].startBloc()
# 					for j in xrange(nb_trials):
# 						state = cats.getStimulus(j)
# 						action = models[m].chooseAction(state)
# 						reward = cats.getOutcome(state, action, case='fmri')
# 						models[m].updateValue(reward)                                    
# 			# rtm = np.array(models[m].reaction).reshape(nb_repeat, nb_blocs, nb_trials)                        
# 			# state = convertStimulus(np.array(models[m].state)).reshape(nb_repeat, nb_blocs, nb_trials)
# 	  #       action = np.array(models[m].action).reshape(nb_repeat, nb_blocs, nb_trials)
# 	  #       responses = np.array(models[m].responses).reshape(nb_repeat, nb_blocs, nb_trials)
# 	  #       tmp = np.zeros((nb_repeat, 15))
# 	  #       for i in xrange(nb_repeat):
# 	  #           rtm[i] = center(rtm[i], o, s, m)
#    #  	        step, indice = getRepresentativeSteps(rtm[i], state[i], action[i], responses[i])
#    #      	    tmp[i] = computeMeanRepresentativeSteps(step)[0]        			
# 			state = convertStimulus(np.array(models[m].state))
# 			action = np.array(models[m].action)
# 			responses = np.array(models[m].responses)        
# 			hall = np.array(models[m].Hall)
# 			N = np.array(models[m].pdf)

# 			data[o]['Hb'][s] = {m:extractStimulusPresentation(hall[:,:,0], state, action, responses)}        
# 			data[o]['Hf'][s] = {m:extractStimulusPresentation(hall[:,:,1], state, action, responses)}

# 			entropy[g]['Hb'][s] = extractStimulusPresentation(hall[:,:,0], state, action, responses)
# 			entropy[g]['Hf'][s] = extractStimulusPresentation(hall[:,:,1], state, action, responses)
# 			entropy[g]['N'][s] = extractStimulusPresentation(N, state, action, responses)
	
# 	fig = figure()

# 	axes = {'Hb':{i:fig.add_subplot(2,3,i+1) for i in xrange(3)}, 'Hf':{i:fig.add_subplot(2,3,i+4) for i in xrange(3)}}

# 	for g in entropy.iterkeys():
# 		for h in ['Hb', 'Hf']:
# 			for s in entropy[g][h].iterkeys():
# 				for i in xrange(3):
# 					y = entropy[g][h][s]['mean'][i]
# 					x = range(1, len(y)+1)
# 					e = entropy[g][h][s]['sem'][i]
# 					axes[h][i].plot(x, y, linewidth = 2, color = colors_m[g]) 
# 					axes[h][i].fill_between(x, y-e, y+e, facecolor = colors_m[g], alpha = 0.1)
# 					axes[h][i].set_ylim(0,np.log2(5))
# 					# if h == 'Hb':
# 					# 	ax2 = axes[h][i].twinx()
# 					# 	ax2.plot(x, entropy[g]['N'][s]['mean'][i], '--', color = colors_m[g], linewidth = 3)
# 	axes['Hb'][0].set_ylabel("Entropy BWM")
# 	axes['Hf'][0].set_ylabel("Entropy Q-L")
# 	for i in xrange(3): axes['Hf'][i].set_xlabel("Trials")
# 	for i in xrange(3): axes['Hb'][i].set_title("Stimulus "+str(i+1))

# 	# fig2 = figure()
# 	# axes = {i:fig2.add_subplot(1,3,i+1) for i in xrange(3)}
# 	# for m in groups['tche'].iterkeys():
# 	# 	for s in groups['tche'][m]:
# 	# 		hb = entropy[m]['Hb'][s]['mean']
# 	# 		hf = entropy[m]['Hf'][s]['sem']
# 	# 		r = hb-hf
# 	# 		for i in xrange(3):
# 	# 			x = range(1, len(r[i])+1)
# 	# 			axes[i].plot(x, r[i], linewidth=2, color = colors_m[m])

# 	# legend()
# 	tight_layout()
# 	savefig("contributionSystem.pdf")
# 	os.system("evince contributionSystem.pdf")
			
			
