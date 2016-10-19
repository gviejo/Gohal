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
nb_repeat = 100
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

for m in ['fusion', 'selection', 'mixture']:
	sstate = []
	aaction = []
	rresponses = []
	eentropy = []
	w = []
	vpi = []
	rrate = []
	data[m] = {}
	nb_sujet = 0
	for s in groups[m]:
		print "Testing "+s+" with "+m+" selected by "+o
		models[m].setAllParameters(p_test[o][s][m])
		models[m].startExp()
		entropy = np.zeros((nb_repeat,nb_blocs,nb_trials,6))
		for k in xrange(nb_repeat):
			for i in xrange(nb_blocs):
				cats.reinitialize()
				models[m].startBloc()			
				for j in xrange(nb_trials):	
					state = cats.getStimulus(j)
					action = models[m].chooseAction(state)					
					entropy[k,i,j,0] = models[m].h_bayes_only
					entropy[k,i,j,1] = models[m].h_ql_only
					entropy[k,i,j,2] = models[m].Hl
					entropy[k,i,j,3] = models[m].nb_inferences
					reward = cats.getOutcome(state, action, case='fmri')
					models[m].updateValue(reward)                                    

		state = convertStimulus(np.array(models[m].state))
		action = np.array(models[m].action)
		responses = np.array(models[m].responses)        
		entropy = entropy.reshape(nb_blocs*nb_repeat, nb_trials, 6)		
		sstate.append(state)
		aaction.append(action)
		rresponses.append(responses)
		eentropy.append(entropy)
		if m == 'mixture':
			w.append(np.array(models[m].weights))
		elif m == 'selection':
			vpi.append(np.array(models[m].vpi))
			rrate.append(np.array(models[m].rrate))
		nb_sujet+=1

	sstate = np.array(sstate).reshape(nb_sujet*nb_repeat*nb_blocs,39)
	aaction = np.array(aaction).reshape(nb_sujet*nb_repeat*nb_blocs,39)
	rresponses = np.array(rresponses).reshape(nb_sujet*nb_repeat*nb_blocs,39)
	eentropy = np.array(eentropy).reshape(nb_sujet*nb_repeat*nb_blocs,39,6)

	hb = computeMeanRepresentativeSteps(getRepresentativeSteps(eentropy[:,:,0], sstate, aaction, rresponses)[0])
	hf = computeMeanRepresentativeSteps(getRepresentativeSteps(eentropy[:,:,1], sstate, aaction, rresponses)[0])
	hl = computeMeanRepresentativeSteps(getRepresentativeSteps(eentropy[:,:,2], sstate, aaction, rresponses)[0])
	data[m] = {'hb':hb,'hf':hf,'hl':hl}

	if m == 'mixture':
		w = np.array(w).reshape(nb_sujet*nb_repeat*nb_blocs,39)
		wm = computeMeanRepresentativeSteps(getRepresentativeSteps(w, sstate, aaction, rresponses)[0])
		data[m]['w'] = wm
	elif m == 'selection':
		vpi = np.array(vpi).reshape(nb_sujet*nb_repeat*nb_blocs,39)
		rrate = np.array(rrate).reshape(nb_sujet*nb_repeat*nb_blocs,39)
		data[m]['vpi'] = computeMeanRepresentativeSteps(getRepresentativeSteps(vpi, sstate, aaction, rresponses)[0])
		data[m]['rrate'] = computeMeanRepresentativeSteps(getRepresentativeSteps(rrate, sstate, aaction, rresponses)[0])
	

	

figure()
pl = 1
for m in data.keys():	
	subplot(2,3,pl)
	for h in ['hb', 'hf', 'hl']:
		errorbar(range(15), data[m][h][0], data[m][h][1], label = h)

	legend()
	# twinx()
	# plot(data[m][s]['N'][0], '--', color = 'black', linewidth = 3)		
	title(m+" "+s)
	pl+=1
	# ylim(0,10)

subplot(2,3,5)
errorbar(range(15), data['selection']['vpi'][0], data['selection']['vpi'][1], label = 'vpi')
errorbar(range(15), data['selection']['rrate'][0], data['selection']['rrate'][1], label = 'r')
legend()

subplot(2,3,6)
errorbar(range(15), data['mixture']['w'][0], data['mixture']['w'][1])

show()


with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/beh_contribution.pickle") , 'wb') as handle:    
     pickle.dump(data, handle)
