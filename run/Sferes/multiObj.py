#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and and plot multi objective results from Sferes 2 optimisation 


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
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
best = np.array([-257.0, 0.0])
front = pareto(options.input, best, N = 156)

front.constructParetoFrontier()

front.removeIndivDoublons()
# model = front.reTest(20)
##front.constructParetoFrontier()
front.constructMixedParetoFrontier()
front.rankDistance()
front.rankOWA()
front.rankTchebytchev()
front.zoomBox(0.0, 0.0)
# front.preview()
# data_single, p_test_single = front.rankIndividualStrategy()
# show()


# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/pareto_front.pickle") , 'wb') as handle:    
#     pickle.dump(front.pareto, handle)

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/mixed_pareto_front.pickle"), 'wb') as handle:    
#     pickle.dump(front.mixed, handle)

# # useless
# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rank_distance.pickle"), 'wb') as handle:
# 	pickle.dump(front.distance, handle)

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rank_all_operators.pickle"), 'wb') as handle:
# 	pickle.dump(front.zoom, handle)

# with open("parameters.pickle", 'wb') as f:
# 	pickle.dump(front.p_test, f)

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rank_single.pickle"), 'wb') as handle:
# 	pickle.dump(data_single, handle)

# with open("parameters_single.pickle", 'wb') as f:
# 	pickle.dump(p_test_single, f)


figure()
# s_to_plot = ['S2','S9','S19','S13','S8','S5','S14']
s_to_plot = front.pareto['qlearning'].keys()

for s in s_to_plot:	
	for m in front.pareto.iterkeys():
	# for s in front.pareto[m].iterkeys():
	
		obj = front.pareto[m][s][:,3]
		ind = np.ones(len(obj))*(s_to_plot.index(s)+0.1*float(front.pareto.keys().index(m)))
		plot(ind, obj, 'o', color = front.colors_m[m], markersize = 10, alpha = 0.8)
ylim(0.55, 0.95)
xticks(range(0,len(s_to_plot)), s_to_plot)
show()





show()