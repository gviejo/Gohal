#!/usr/bin/python
# encoding: utf-8
"""
HumanLearning.py

scripts to load and analyze data from Brovelli
python HumanLearning.py -i ../../Data/PEPS_GoHaL/Beh_Model/
or
python HumanLearning.py -i ../../Data/fMRI

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
sys.path.append("../../src")
import os
from optparse import OptionParser
import numpy as np
from fonctions import *
from LearningAnalysis import SSLearning
import scipy.io
from scipy.stats import binom, sem
from pylab import *
from ColorAssociationTasks import CATS

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# FONCTIONS
# -----------------------------------    
def loadDirectoryMEG(direct):
    data = dict()
    line = "ls "+direct
    p = os.popen(line, "r").read()
    files = p.split('\n')[:-1]    
    for i in files:
        data[i] = dict()
        tmp = scipy.io.loadmat(direct+i+'/beh.mat')['beh']
        for j in range(1, len(tmp[0])-1):
            data[i][j] = {}
            for k in range(len(tmp.dtype.names)):
                data[i][j][tmp.dtype.names[k]] = tmp[0][j][k]   
        tmp = scipy.io.loadmat(direct+i+'/NC.mat')['NC']
        for j in range(1, len(tmp[0])-1):
            for k in range(len(tmp.dtype.names)):
                data[i][j][tmp.dtype.names[k]] = tmp[0][j][k]
    return data

def loadDirectoryfMRI(direct):
    data = dict()
    tmp = scipy.io.loadmat(direct+'/beh_allSubj.mat')['data']
    m, n = tmp.shape
    for i in xrange(m):
        sujet = str(tmp[i][0][-1][0]).split("\\")[-2]
        data[sujet] = dict()
        for j in xrange(n):
            num = int(list(str(tmp[i][j][-1][0]).split("\\")[-1])[-1])
            data[sujet][num] = dict()
            for k in range(len(tmp[i][j].dtype.names)):
                if tmp[i][j].dtype.names[k] == 'sar_time':
                    data[sujet][num]['time'] = tmp[i][j][k]
                else:
                    data[sujet][num][tmp[i][j].dtype.names[k]] = tmp[i][j][k]                                   
    return data


# -----------------------------------

# -----------------------------------
# PARAMETERS + INITIALIZATION
# -----------------------------------
nStepEm = 200
pOutset = 0.2
size = 42 if 'PEPS_GoHaL' in options.input else 39
# -----------------------------------

# -----------------------------------
# Loading human data
# -----------------------------------
responses = []
stimulus = []
action = []
reaction = []
reaction2 = []
indice2 = []
if 'PEPS_GoHaL' in options.input:
    data = loadDirectoryMEG(options.input)
elif 'fMRI' in options.input:
    data = loadDirectoryfMRI(options.input)
else:
    print "No directory provided... can't load the data"
    sys.exit(0)

for i in data.iterkeys():
    for j in data[i].iterkeys():
        responses.append(data[i][j]['sar'][0:size,2])
        stimulus.append(data[i][j]['sar'][0:size,0])
        action.append(data[i][j]['sar'][0:size,1])
        reaction2.append(data[i][j]['time'][0:size,1]-data[i][j]['time'][0:size,0])
responses = np.array(responses)
stimulus = np.array(stimulus)
action = np.array(action)
reaction2 = np.array(reaction2)
# -----------------------------------

# -----------------------------------
# Time Analysis
# -----------------------------------
reaction = []
for i in data.iterkeys():
    for j in data[i].iterkeys():
        if 'PEPS_GoHaL' in options.input:
            #reaction.append(data[i][j]['time'][0:size,1]-data[i][j]['time'][0:size,0])
            reaction.append(data[i][j]['RT'].flatten()[0:size])
        elif 'fMRI' in options.input:
            reaction.append(data[i][j]['RT'].flatten()[0:size])
reaction = np.array(reaction)
# -----------------------------------

# -----------------------------------
#fitting state space model
# -----------------------------------
ss = SSLearning(len(responses[0]), pOutset)
p = []
for r in responses:
    ss.runAnalysis(r, nStepEm)
    p.append(ss.pmode)
p = np.array(p)

# index of performance
Ipm = np.log2(p/pOutset)

# index of cognitive control
Icc = np.zeros(responses.shape)
for i in xrange(Icc.shape[0]):
    j = np.where(responses[i] == 1)[0][2]
    Icc[i,0:j+1] = -np.log2(1-p[i, 0:j+1])
    Icc[i,j+1:-1] = -np.log2(p[i, j+1:-1])

# stimulus driven index
Psd = np.ones(responses.shape)
for s in xrange(1,4):
    for i in xrange(Psd.shape[0]):
        for j in xrange(Psd.shape[1]):
            n = np.sum(stimulus[i, 0:j]==s)
            k = np.sum(responses[i, 0:j][stimulus[i,0:j]==s])
            Psd[i, j] = Psd[i, j] * binom.cdf(k, n, pOutset)            
Isd = np.log2(Psd/(0.5**3))
# -----------------------------------

# -----------------------------------
# Representative Steps
# -----------------------------------
steps, indice = getRepresentativeSteps(reaction, stimulus, action, responses)
steps_reaction = steps
steps_Ipm,ind = getRepresentativeSteps(Ipm, stimulus, action, responses)
steps_Icc,ind = getRepresentativeSteps(Ipm+Icc, stimulus, action, responses)
steps_Isd,ind = getRepresentativeSteps(Isd, stimulus, action, responses)
steps_responses,ind = getRepresentativeSteps(responses, stimulus, action, responses)
# -----------------------------------

# -----------------------------------
# Plot
# -----------------------------------
figure()
m, s = computeMeanRepresentativeSteps(steps_Ipm)
plot(m,'o-', color='red', label = 'Ipm')
m, s = computeMeanRepresentativeSteps(steps_Icc)
plot(m,'o-',color='green', label = 'Icc')
m, s = computeMeanRepresentativeSteps(steps_Isd)
plot(m,'o-', color='blue', label = 'Isd')
legend()
grid()

figure()
subplot(211)
m,s = computeMeanRepresentativeSteps(steps_reaction)
errorbar(range(len(m)), m, s, linewidth = 2)
xlabel('representative steps')
ylabel('reaction time (s)')
grid()

subplot(212)
m, s = computeMeanRepresentativeSteps(steps_responses)
errorbar(range(len(m)), m, s, linewidth = 2)
xlabel('trials')
ylabel('accuracy')
grid()


show()

# -----------------------------------









