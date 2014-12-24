#!/usr/bin/env python
# -*- coding: utf-8 -*-
                                                               
import sys                                                                          
sys.path.append("../../src")                                            
from Sferes import EA
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *
import cPickle as pickle



model1 = CSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
tmp = [1, 0.0353706, 0.866322, 1, 0.0773437, 0.999994, 0.00193269]	
order = ['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma']
parameters = {}
for p,i in zip(order, tmp):
    parameters[p] = model1.bounds[p][0]+i*(model1.bounds[p][1]-model1.bounds[p][0])


# parameters = {'alpha':0.0,
# 			'beta':94.953,
# 			'noise':0.116,
# 			'length':10.0,
# 			'weight':0.821,
# 			# 'weight':1.0,
# 			'threshold':0.045,
# 			'sigma':100.0,
# 			'gain':3.758}

# parameters = {'alpha': 0.98999999999999999,
#  'beta': 4.0883544999999994,
#  'length': 10.0,
#  'noise': 0.42176178000000003,
#  'sigma': 0.0,
#  'threshold': 2.3163563481786835,
#  'weight': 0.10000000000000001}

with open("fmri/S14.pickle", "rb") as f:
   data = pickle.load(f)


model1 = CSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)
opt = EA(data, 'S14', model1)
llh, lrs = opt.getFitness()                                                                                      
print llh, lrs

# print float(llh)-2000.0


