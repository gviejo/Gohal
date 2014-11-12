#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explore parameters space


Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
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

# -----------------------------------
# LOADING DATA
# -----------------------------------
best = np.array([-4*(3*np.log(5)+2*np.log(4)+2*np.log(3)+np.log(2)), 0.0])
front = pareto(options.input, best, N = 156)

front.constructParetoFrontier()

front.removeIndivDoublons()
# model = front.reTest(20)
# front.constructParetoFrontier()
front.constructMixedParetoFrontier()
front.rankDistance()
front.rankOWA()
front.rankTchebytchev()

position = {2:(1,2),3:(2,2),4:(2,2),5:(2,3),6:(2,3),8:(4,4),9:(3,3)}

parameters = {'bayesian':['length','noise','sigma'],
				'qlearning':['beta','sigma'],
				'fusion':['beta','length','threshold','noise','gain','sigma'],
				'mixture':['length','noise','beta','gain','sigma'],
				'selection':['beta','length','noise','sigma_rt']}

color_list = plt.cm.Dark2(np.linspace(0, 1, 14))

for m in front.pareto.iterkeys():
	fig = figure()
	fig.canvas.set_window_title(m)
	n = len(parameters[m])
	for i in xrange(n):
		ax = fig.add_subplot(position[n][0],position[n][1],i+1)
		for s in front.pareto[m].iterkeys():
			data = front.pareto[m][s][:,5:]
			ax.plot(range(len(data)), data[:,front.p_order[m].index(parameters[m][i])], '.-', color = color_list[front.pareto[m].keys().index(s)], alpha = 0.5)
			ax.set_title(parameters[m][i])
	fig.savefig("space_parameters_"+m+".pdf")

os.system("evince space_parameters_*.pdf")
