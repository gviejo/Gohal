#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

do the analysis for the leave-one-out verification for frontiers.


Copyright (c) 2015 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *

from Models import *

from matplotlib import *
from pylab import *

from Sferes import pareto
from itertools import *

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
	sys.stdout.write("Sorry: you must specify at least 1 argument")
	sys.stdout.write("More help avalaible with -h or --help option")
	sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test \n If none is provided, all files are loaded", default=False)
parser.add_option("-o", "--output", action="store", help="The output file of best parameters to test", default=False)
(options, args) = parser.parse_args()
# -----------------------------------

def pickling(direc):
	with open(direc, "rb") as f:
		return pickle.load(f)

human = dict({s_dir.split(".")[0]:pickling("../Sferes/fmri/"+s_dir) for s_dir in os.listdir("../Sferes/fmri/")})        

# -----------------------------------
# LOADING DATA
# -----------------------------------
directory = options.input

data = dict()

# LOAD DATA
model_in_folders = os.listdir(directory)
for m in model_in_folders:
	data[m] = dict()
	lrun = os.listdir(directory+"/"+m)

	for r in lrun:
		s = r.split("_")[3]
		n = int(r.split("_")[4].split(".")[0])
		if s in data[m].keys():
			data[m][s][n] = np.genfromtxt(directory+"/"+m+"/"+r)
		else:
			data[m][s] = dict()
			data[m][s][n] = np.genfromtxt(directory+"/"+m+"/"+r)


# CONSTRUCT PARETO FRONTIER
N = 117
pareto = dict()
best_log = -3*(3*np.log(5)+2*np.log(4)+2*np.log(3)+np.log(2))
worst_log = N*np.log(0.2)
for m in data.iterkeys():
	pareto[m] = dict()
	for s in data[m].iterkeys():
		pareto[m][s] = dict()
		for n in xrange(1, 5):
			tmp = data[m][s][n]
			ind = tmp[:,2] != 0
			tmp = tmp[ind]
			tmp = tmp[tmp[:,2].argsort()][::-1]
			pareto_frontier = [tmp[0]]
			for pair in tmp[1:]:
				if pair[3] >= pareto_frontier[-1][3]:
					pareto_frontier.append(pair)
			pareto[m][s][n] = np.array(pareto_frontier)
			pareto[m][s][n][:,2] = pareto[m][s][n][:,2] - 2000.0
			pareto[m][s][n][:,3] = pareto[m][s][n][:,3] - 500.0
			pareto[m][s][n][:,2] = 1.0 - (pareto[m][s][n][:,2]/(N*np.log(0.2)))
			
			pareto[m][s][n][:,3] = 1.0 - (-pareto[m][s][n][:,3])/(np.power(2*human[s]['mean'][0], 2).sum())
			# on enleve les points negatifs
			pareto[m][s][n] = pareto[m][s][n][(pareto[m][s][n][:,3:5]>0).prod(1)==1]


# CONSTRUCT MIXED 
m_order = ['qlearning', 'bayesian', 'selection', 'fusion', 'mixture']
mixed = dict()
subjects = pareto['fusion'].keys()
for s in subjects:
	mixed[s] = {}
	for n in np.arange(1, 5):		
		tmp = []
		for m in pareto.iterkeys():
			if s in pareto[m].iterkeys():
				tmp.append(np.hstack((np.ones((len(pareto[m][s][n]),1))*m_order.index(m), pareto[m][s][n][:,0:4])))            
		tmp = np.vstack(tmp)
		tmp = tmp[tmp[:,3].argsort()][::-1]
		if len(tmp):
			mixed[s][n] = []
			mixed[s][n] = [tmp[0]]
			for pair in tmp[1:]:
				if pair[4] >= mixed[s][n][-1][4]:
					mixed[s][n].append(pair)
		mixed[s][n] = np.array(mixed[s][n])

# TCHEBYTCHEV
lambdaa = 0.5
epsilon = 0.001
best = {}
for s in mixed.iterkeys():
	best[s] = dict()
	for n in xrange(1,5):		
		tmp = mixed[s][n][:,3:5]
		ideal = np.max(tmp, 0)
		nadir = np.min(tmp, 0)
		value = lambdaa*((ideal-tmp)/(ideal-nadir))
		value = np.max(value, 1) + epsilon*np.sum(value, 1)
		ind_best_point = np.argmin(value)
		best_ind = mixed[s][n][ind_best_point]
		m = m_order[int(best_ind[0])]
		best[s][n] = m


# WRITE 

f = open("best.txt", "w")

f.write("-Bloc "+" ".join(best.keys())+"\n")

for i in xrange(1, 5):
	line = "-"+str(i)+" "

	for s in best.keys():
		line += best[s][i] + " "


	f.write(line+"\n")

f.close()



figure()

for i in xrange(1, 5):
	subplot(2,2,i)
	for s in subjects:
		plot(mixed[s][i][:,3], mixed[s][i][:,4], 'o-')

show()