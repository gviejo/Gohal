#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# encoding: utf-8
"""
exp1.py

scripts to Keramati first figure pour IAD report
figure 5 : moderate training | extensive training

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
from fonctions import *
from Selection import Keramati
from Models import KalmanQLearning
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
parser.add_option("-o", "--output", action="store", help="The name of the output file to store the data", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------
def transitionRules(state, action):
    if state == 's0' and action == 'pl':
        return 's1'
    else:
        return 's0'

    
# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
beta = 1.0       # rate of exploration
gamma = 0.95     # discount factor
sigma = 0.02     # updating rate of the average reward
rau = 0.1        # update rate of the reward function
tau = 0.08       # time step for graph exploration

phi = 0.5        # update rate of the transition function
depth = 3        # depth of search when computing the goal value
init_cov = 1.0   # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_iter_test = 500

nb_iter_mod = 100
deval_mod_time = 40
nb_iter_ext = 300
deval_ext_time = 240

nb_blocs = 100

states = ['s0', 's1']
actions = ['pl', 'em']
rewards = createQValuesDict(states, actions)
kalman = KalmanQLearning("", states, actions, gamma, beta, eta, var_obs, init_cov, kappa)
selection = Keramati(kalman, depth, phi, rau, sigma, tau)

meandata = dict({'mod':dict({'vpi':list(),
                             'r':list(),
                             'p':list(),
                             'q':list()}),
                 'ext':dict({'vpi':list(),
                             'r':list(),
                             'p':list(),
                             'q':list()})})

for i in xrange(nb_blocs):
        data = dict({'mod':dict({'vpi':list(),
                                 'r':list(),
                                 'p':list(),
                                 'q':list()}),
                     'ext':dict({'vpi':list(),
                                 'r':list(),
                                 'p':list(),
                                 'q':list()})})

	# -----------------------------------
	# Training + devaluation
	# -----------------------------------

	for exp, nb_trials, deval_time in zip(['mod','ext'], [nb_iter_mod, nb_iter_ext], [deval_mod_time, deval_ext_time]):
	    kalman.initialize()
	    selection.initialize()
	    state = 's0'
	    rewards[0][rewards[('s1','em')]] = 1.0
	    print exp, nb_trials, deval_time
	    for i in xrange(nb_trials):
		#Setting Reward
		if i == deval_time:
		    rewards[0][rewards[('s1','em')]] = 0.0
                    selection.rfunction[0][selection.rfunction[('s1', 'em')]] = -1.0
		#Learning
		while True:
		    action = selection.chooseAction(state)        
		    next_state = transitionRules(state, action)
		    selection.updateValues(rewards[0][rewards[(state, action)]], next_state)
		    #sys.stdin.read(1)
		    if state == 's1' and action == 'em':
		        #Retrieving data
		        data[exp]['vpi'].append(computeVPIValues(kalman.values[0][kalman.values['s0']],kalman.covariance['cov'].diagonal()[kalman.values['s0']]))
		        data[exp]['r'].append(selection.rrate[-1])                                                       
		        data[exp]['p'].append(testQValues(states, selection.values, kalman.beta, 0, nb_iter_test))
		        data[exp]['q'].append(kalman.values[0][kalman.values[('s0','pl')]]-kalman.values[0][kalman.values[('s0','em')]])                
		        state = next_state
		        break
		    else:
		        state = next_state
		
	    data[exp]['vpi'] = np.array(data[exp]['vpi'])
	    data[exp]['r'] = np.array(data[exp]['r'])*selection.tau
	    data[exp]['p'] = np.array(data[exp]['p'])
	    data[exp]['q'] = np.array(data[exp]['q'])

            for s in ['vpi', 'r', 'p']:                 
                meandata[exp][s].append(data[exp][s])

for i in ['mod', 'ext']:
    meandata[i]['vpi'] = np.mean(meandata[i]['vpi'], 0)
    meandata[i]['r'] = np.mean(meandata[i]['r'], 0)
    meandata[i]['p'] = np.mean(meandata[i]['p'], 0)

# -----------------------------------
# Plot
# -----------------------------------\
params = {'backend':'pdf',
          'axes.labelsize':10,
          'text.fontsize':10,
          'legend.fontsize':10,
          'xtick.labelsize':8,
          'ytick.labelsize':8,
          'text.usetex':False}

dashes = ['--', '-.', '-']
colors = ['black', 'grey']
#rcParams.update(params)

fig1 = figure(figsize = (15,9))

subplot(221)
for s in ['s0']:
    for a,i in zip(actions, range(len(actions))):
        plot(meandata['mod']['vpi'][:,selection.values[(s,a)]], linestyle = '-', color = colors[i], label = "VPI("+s+","+a+")", linewidth = 2)
plot(meandata['mod']['r'], color = 'black', label = "R*tau", linestyle = '--', linewidth = 2)
axvline(deval_mod_time, color='black', linewidth = 2)
legend()
grid()
ylim(0,0.1)
xlabel("Trial")

subplot(223)
for s in ['s0']:
    for a,i in zip(actions, range(len(actions))):
        plot(meandata['mod']['p'][:,selection.values[(s,a)]], linestyle = '-', color = colors[i], label = "P("+s+","+a+")", linewidth = 1.5)
axvline(deval_mod_time, color='black', linewidth = 2)
ylim(0.1,0.9)
xlabel("Trial")
ylabel("P(s,a)")
yticks(np.arange(0.2, 0.9, 0.1))
grid()
legend()

subplot(222)
for s in ['s0']:
    for a,i in zip(actions, range(len(actions))):
        plot(meandata['ext']['vpi'][:,selection.values[(s,a)]], linestyle = '-', color = colors[i], label = "VPI("+s+","+a+")", linewidth = 2)
plot(meandata['ext']['r'], linestyle = '--', color = 'black', label = "R*tau", linewidth = 2)
axvline(deval_ext_time, color='black', linewidth = 2)
legend()
xlabel("Trial")
grid()
ylim(0,0.1)


subplot(224)
for s in ['s0']:
    for a,i in zip(actions, range(len(actions))):
        plot(meandata['ext']['p'][:,selection.values[(s,a)]], linestyle = '-', color = colors[i], label = "P("+s+","+a+")", linewidth = 1.5)
axvline(deval_ext_time, color='black', linewidth = 2)
ylim(0.1,0.9)
grid()
xlabel("Trial")
ylabel("P(s,a)")
yticks(np.arange(0.2, 0.9, 0.1))
legend()

subplots_adjust(left = 0.08, wspace = 0.3, right = 0.86, hspace = 0.35)
figtext(0.12, 0.93, "Moderate pre-devaluation training", fontsize = 18)
figtext(0.55, 0.93, "Extensive pre-devaluation training", fontsize = 18)
figtext(0.06, 0.92, 'A', fontsize = 20)
figtext(0.06, 0.45, 'C', fontsize = 20)
figtext(0.49, 0.92, 'B', fontsize = 20)
figtext(0.49, 0.45, 'D', fontsize = 20)

fig1.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig5.pdf', bbox_inches='tight')

show()
# -----------------------------------








