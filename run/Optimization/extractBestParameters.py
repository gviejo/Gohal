#!/usr/bin/python
# encoding: utf-8
"""
parametersOptimization.py

scripts to load and plot parameters

run parameterTest.py -i data_model_date -m 'model'

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys

from optparse import OptionParser
import numpy as np
import cPickle as pickle
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS
from HumanLearning import HLearning
from Models import *
from matplotlib import *
from pylab import *

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test", default=False)
parser.add_option("-o", "--output", action="store", help="The name of the output files", default=False)

(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------

# -----------------------------------

# -----------------------------------
# PARAMETERS Loading
# -----------------------------------
f = open(options.input, 'rb')
p = pickle.load(f)
f.close()
# -----------------------------------

# -----------------------------------
# Order data
# -----------------------------------

n_search = p['search']
subject = p['subject']
n_parameters = len(p['p_order'])
fname = p['fname']
X = p['opt']
if fname == 'minimize':
    tmp = np.zeros((len(subject), n_search, n_parameters))
    fun = np.zeros((len(subject), n_search))
    X = np.reshape(X, (len(subject), n_search))
    for i in xrange(len(X)):
        for j in xrange(len(X[i])):
            if X[i][j].success == True:
                tmp[i][j] = X[i][j].x
                fun[i][j] = -X[i][j].fun        
    X = tmp
elif fname == 'fmin':
    X = np.reshape(X, (len(subject), n_search, n_parameters))
else:
    print "scipy function not specified\n"
    sys.exit()


# -----------------------------------

# -----------------------------------
# Saving 
# -----------------------------------
output = open(options.output, 'w')
output.write("# Optimization function : "+fname+"\n")
output.write("# Nb search : "+str(n_search)+"\n")

for i in xrange(len(p['subject'])):
    best = X[i][fun[i] == np.max(fun[i])]
    l = round(np.max(fun[i]), 2)
     
    line = p['subject'][i]+":(likelihood:"+str(l)+","
    for j in xrange(len(p['p_order'])):
        line = line+p['p_order'][j]+":"+str(round(best[0][j], 2))+","    
    line = line[0:-1] + ")\n"
    output.write(line)

output.close()    