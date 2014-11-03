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

def center(x):
	#x = x-np.mean(x)
	#x = x/np.std(x)
	x = x-np.median(x)
	x = x/(np.percentile(x, 75)-np.percentile(x, 25))
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
nb_repeat = 5
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


subjects = p_test['distance'].keys()

colors = {'owa':'r','distance':'b','tche':'g'}

colors_m = dict({'fusion':'#F1433F',
				'bayesian': '#D5A253',
				'qlearning': '#6E8243',
				'selection':'#70B7BA',
				'mixture':'#3D4C53'})

colors = dict({0:'red',1:'green',2:'blue'})
styles = dict({'Hb':'o-','Hf':'o--', 'N':'-'})
operators = ['owa', 'distance', 'tche']

for s in subjects:
	data = {}	
	for o in operators:
		data[o] = {'Hb':{},'Hf':{}, 'N':{}}		
		m = p_test[o][s].keys()[0]
		print "Testing "+s+" with "+m+" selected by "+o
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
		state = convertStimulus(np.array(models[m].state))
		action = np.array(models[m].action)
		responses = np.array(models[m].responses)        
		entropy = np.array(models[m].Hall)
		N = np.array(models[m].pdf)
		data[o]['Hb'] = {m:extractStimulusPresentation(entropy[:,:,0], state, action, responses)}        
		data[o]['Hf'] = {m:extractStimulusPresentation(entropy[:,:,1], state, action, responses)}
		data[o]['N'] = {m:extractStimulusPresentation(N, state, action, responses)}

	fig = figure()
	fig.canvas.set_window_title(s)
	ind = 1	
	for h in data[o].keys():			
		for o in operators:		
			for i in xrange(3):
				ax = fig.add_subplot(3,9,ind)				
				ind+=1
				y = data[o][h]['fusion']['mean'][i]
				e = data[o][h]['fusion']['sem'][i]
				x = range(1, len(y)+1)
				ax.plot(x, y, styles[h], linewidth=1, color = colors[i])								
				ax.fill_between(x, y-e, y+e, alpha = 0.3)
				ax.set_title(i)				
				if h in ['Hb', 'Hf']:
					ax.plot()
					ax.set_ylim(0,np.log2(5))
					ax.set_yticks([])
					ax.set_xticks([])
				else:
					ax.set_yticks([0,5,10,15])
					ax.set_xticks([])
	fig.subplots_adjust(left=0.0, right=1.0)
	fig.savefig("contribution_"+s+".pdf")
	

os.system("evince contribution_*.pdf")



			
			
