#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_exp1.py

Plot the exp 1 from Keramati & al, 2011
Take as input a folder containing the data
Each run is in the form of exp1run_.pickle
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np
sys.path.append("../../src")
from fonctions import *
import subprocess
from pylab import plot, figure, show, subplot, legend, ylim, axvline
# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
#if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the input folder to load the data", default=False)
#parser.add_option("-o", "--output", action="store", help="The name of the output file to store the data", default=False)
(options, args) = parser.parse_args() 
# -----------------------------------

# -----------------------------------
# PARAMETERS
# -----------------------------------
eta = 0.0001     # variance of evolution noise v
var_obs = 0.05   # variance of observation noise n
beta = 1.0       # rate of exploration
gamma = 0.95     # discount factor
sigma = 0.02     # updating rate of the average reward
rau = 0.1        # update rate of the reward function
tau = 0.08       # time step for graph exploration

phi = 0.1        # update rate of the transition function
depth = 3        # depth of search when computing the goal value
init_cov = 1.1     # initialisation of covariance matrice
kappa = 0.1      # unscentered transform parameters

nb_iter_test = 500

nb_iter_mod = 100
deval_mod_time = 40
nb_iter_ext = 350
deval_ext_time = 240

states = ['s0', 's1']
actions = ['pl', 'em']

values = createQValuesDict(states, actions)
# -----------------------------------
# Loading data
# -----------------------------------

data = dict({'data':dict({'values':dict({'mean':[],
                                         'var':[]}),
                          'h':dict({'mean':[],
                                    'var':[]}),
                          'r':dict({'mean':[],
                                    'var':[]}),
                          'vpi':dict({'mean':[],
                                      'var':[]}),
                          'p':dict({'mean':[],
                                    'var':[]})}),
             'data2':dict({'values':dict({'mean':[],
                            'var':[]}),
                          'h':dict({'mean':[],
                                    'var':[]}),
                          'r':dict({'mean':[],
                                    'var':[]}),
                          'vpi':dict({'mean':[],
                                      'var':[]}),
                          'p':dict({'mean':[],
                                    'var':[]})})})

tmp = dict({'data': dict({'values':dict({0:[],1:[],2:[],3:[]}), 
                          'h':dict({0:[],1:[],2:[],3:[]}), 
                          'vpi':dict({0:[], 1:[]}), 
                          'r':[],
                          'p':dict({0:[],1:[],2:[],3:[]})}),
            'data2': dict({'values':dict({0:[],1:[],2:[],3:[]}), 
                           'h':dict({0:[],1:[],2:[],3:[]}), 
                           'vpi':dict({0:[], 1:[]}), 
                           'r':[],
                           'p':dict({0:[],1:[],2:[],3:[]})})})

process = subprocess.Popen("ls "+options.input+" | grep exp1", shell = True, stdout=subprocess.PIPE)
list_data = process.communicate()[0].split("\n")

for i in list_data[0:-1]:
    d = loadData(options.input+"/"+i)
    for j in d.iterkeys():
        for k in d[j].iterkeys():
            if len(d[j][k].shape) == 2:
                for c in range(d[j][k].shape[1]):
                    tmp[j][k][c].append(d[j][k][:,c].copy())
            else :
                tmp[j][k].append(d[j][k].copy())

for i in tmp.iterkeys():
    for j in tmp[i].iterkeys():
        if type(tmp[i][j]) == list:
            data[i][j]['mean'] = np.mean(tmp[i][j], 0)
            data[i][j]['var'] = np.var(tmp[i][j], 0)
        else:
            for k in tmp[i][j]:
                data[i][j]['mean'].append(np.mean(tmp[i][j][k], 0))
                data[i][j]['var'].append(np.var(tmp[i][j][k], 0))
            data[i][j]['mean'] = np.array(data[i][j]['mean'])
            data[i][j]['var'] = np.array(data[i][j]['var'])

delib = dict({'data':dict({'mean':[],
                           'var':[]}),
              'data2':dict({'mean':[],
                            'var':[]})})
                         
n = len(tmp['data']['vpi'][0])
for i in delib.iterkeys():
    diff = []
    for k in range(n):
        a = np.mean([tmp[i]['vpi'][0][k], tmp[i]['vpi'][1][k]], 0)
        diff.append(a-tmp[i]['r'][k])
    delib[i]['mean'] = np.mean(diff, 0)
    delib[i]['var'] = np.var(diff, 0)
        
# -----------------------------------
# Plot
# -----------------------------------\

colors = {('s0','pl'):'green',('s0','em'):'red',('s1','pl'):'cyan',('s1','em'):'purple'}
figure()
subplot(521)
#for s in states:
for s in ['s0']:
    for a in actions:
        plot(data['data']['vpi']['mean'][values[(s,a)]], 'o-', color = colors[(s,a)], label = "VPI("+s+","+a+")")
plot(data['data']['r']['mean'], 'o-', color = 'blue', label = "R*tau")
axvline(deval_mod_time-1, color='black')
legend()
ylim(0,0.1)
subplot(522)
#for s in states:
for s in ['s0']:
    for a in actions:
        plot(data['data2']['vpi']['mean'][values[(s,a)]], 'o-', color = colors[(s,a)], label = "VPI("+s+","+a+")")
plot(data['data2']['r']['mean'], 'o-', color = 'blue', label = "R*tau")
axvline(deval_ext_time-1, color='black')
legend()
ylim(0,0.1)
subplot(523)
for s in ['s0']:
    for a in actions:
        plot(data['data']['p']['mean'][values[(s,a)]], 'o-', color = colors[(s,a)], label = "p("+s+","+a)
axvline(deval_mod_time-1, color='black')
ylim(0.3,0.7)
legend()
subplot(524)
for s in ['s0']:
    for a in actions:
        plot(data['data2']['p']['mean'][values[(s,a)]], 'o-', color = colors[(s,a)], label = "p("+s+","+a)
axvline(deval_ext_time-1, color='black')
ylim(0.3,0.7)
legend()
subplot(525)
plot(delib['data']['mean']>0, color = 'blue', label = 'deliberation time')
axvline(deval_mod_time-1, color='black')
ylim(0, 1.5)
legend()
subplot(526)
plot(delib['data2']['mean']>0, color = 'blue', label = 'deliberation time')
axvline(deval_ext_time-1, color='black')
ylim(0, 1.5)
legend()
subplot(527)
plot(data['data']['h']['mean'][0]-data['data']['h']['mean'][1], label = "Qh(s0, pl)-Qh(s0, em)")
axvline(deval_mod_time-1, color='black')
legend()
ylim(0,0.5)
subplot(528)
plot(data['data2']['h']['mean'][0]-data['data2']['h']['mean'][1], label = "Qh(s0, pl)-Qh(s0, em)")
axvline(deval_ext_time-1, color='black')
ylim(0,0.5)
legend()

show()
# -----------------------------------









