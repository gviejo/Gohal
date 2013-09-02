#!/usr/bin/python
# encoding: utf-8
"""
scripts to plot figure pour le rapport IAD
figure4 : optimization de parameters

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from matplotlib import *
from pylab import *
from sklearn.cluster import KMeans

import cPickle as pickle

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the pickle to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------

# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
correlation = 'Z'

p = ['beta', 'gamma', 'lenght', 'noise']

# -----------------------------------
# DATA loading
# -----------------------------------
pkfile = open(options.input,'rb')
data = pickle.load(pkfile)
values = data['values']
tmp = data.keys()
tmp.remove('values')
X = data[np.max(tmp)]
X = np.transpose(X)

est = KMeans(500)
est.fit(X)
markersize = np.array([np.sum(est.labels_ == i) for i in xrange(est.k)])
markersize = markersize/float(np.max(markersize))
markersize = markersize*20
# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
figure()
subplot(2,1,1)
plot(X[:,0], X[:,1], 'o')
subplot(2,1,2)
plot(X[:,2], X[:,3], 'o')

fig = figure(figsize=(9,6))

subplot(3,2,1)
for i in xrange(est.k):
    plot(est.cluster_centers_[i,0],est.cluster_centers_[i,1],'o', markersize=markersize[i])
xlabel(p[0])
ylabel(p[1])
grid()
#xticks(values['beta'])
#yticks(values['gamma'])
ylim(0, 1)
xlim(0, 5)

subplot(3,2,2)
for i in xrange(est.k):
    plot(est.cluster_centers_[i,2],est.cluster_centers_[i,3],'o', markersize=markersize[i])
xlabel(p[2])
ylabel(p[3])
grid()
#xticks(values['lenght'])
#yticks(values['noise'])
ylim(0, 0.5)
xlim(3, 15)

subplot(3,2,3)
for i in xrange(est.k):
    plot(est.cluster_centers_[i,0],est.cluster_centers_[i,2],'o', markersize=markersize[i])
xlabel(p[0])
ylabel(p[2])
title("B-WM")
grid()
#xticks(values[p[0]])
#yticks(values[p[2]])
xlim(np.min(values[p[0]]),np.max(values[p[0]]))
ylim(np.min(values[p[2]]),np.max(values[p[2]]))

subplot(3,2,4)
for i in xrange(est.k):
    plot(est.cluster_centers_[i,2],est.cluster_centers_[i,1],'o', markersize=markersize[i])
xlabel(p[2])
ylabel(p[1])
title("B-WM")
grid()
#xticks(values[p[1]])
#yticks(values[p[3]])
xlim(np.min(values[p[2]]),np.max(values[p[2]]))
ylim(np.min(values[p[1]]),np.max(values[p[1]]))

subplot(3,2,5)
for i in xrange(est.k):
    plot(est.cluster_centers_[i,0],est.cluster_centers_[i,3],'o', markersize=markersize[i])
xlabel(p[0])
ylabel(p[3])
title("B-WM")
grid()
#xticks(values[p[0]])
#yticks(values[p[2]])
xlim(np.min(values[p[0]]),np.max(values[p[0]]))
ylim(np.min(values[p[3]]),np.max(values[p[3]]))

subplot(3,2,6)
for i in xrange(est.k):
    plot(est.cluster_centers_[i,2],est.cluster_centers_[i,0],'o', markersize=markersize[i])
xlabel(p[2])
ylabel(p[0])
title("B-WM")
grid()
#xticks(values[p[1]])
#yticks(values[p[3]])
xlim(np.min(values[p[2]]),np.max(values[p[2]]))
ylim(np.min(values[p[0]]),np.max(values[p[0]]))


subplots_adjust(wspace = 0.45, hspace = 0.4, right = 0.75)
#fig.savefig('../../../Dropbox/ISIR/Rapport/Rapport_AIAD/Images/fig4.pdf', bbox_inches='tight')
show()


