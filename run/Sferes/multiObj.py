#!/usr/bin/python
# encoding: utf-8
"""
subjectTest.py

load and and plot multi objective results from Sferes 2 optimisation 


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np

sys.path.append("../../src")
from fonctions import *

from Models import *
from matplotlib import *
from pylab import *

from Sferes import pareto
from itertools import *

def plotParetoFront():
    
    for m in front.pareto.iterkeys():
        for i in xrange(len(front.data[m].keys())):
            s = front.data[m].keys()[i]
            ax = subplot(4,4,i+1)
            ax.plot(front.pareto[m][s][:,3], front.pareto[m][s][:,4], "-o")
            #ax.scatter(front.pareto[m][s][:,3], front.pareto[m][s][:,4], c=front.pareto[m][s][:,0])
            ax.scatter(front.pareto[m][s][:,3], front.pareto[m][s][:,4], c=front.rank[m][s])
            ax.plot(front.opt[m][s][3], front.opt[m][s][4], 'o', markersize = 15, label = m, alpha = 0.8)
            ax.grid()
    rcParams['xtick.labelsize'] = 6
    rcParams['ytick.labelsize'] = 6                
    ax.legend(loc='lower left', bbox_to_anchor=(1.15, 0.2), fancybox=True, shadow=True)
    subplots_adjust(left = 0.08, wspace = 0.26, hspace = 0.26, right = 0.92, top = 0.96)
    show()


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
front = pareto(options.input, threshold = [-1000, -1500], N = 156)

front.rankFront([0.5,0.5])

# plotParetoFront()
# sys.exit()
# front.plotFrontEvolution()
# front.plotSolutions()

#front.quickTest('fusion')
#front.quickTest('qlearning')


with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/pareto_front.pickle") , 'wb') as handle:    
 pickle.dump(front.pareto, handle)

with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/mixed_pareto_front.pickle"), 'wb') as handle:    
	pickle.dump(front.mixed, handle)



