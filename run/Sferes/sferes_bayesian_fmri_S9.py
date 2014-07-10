#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/python                                                       
# encoding: utf-8                                                                   
import sys                                                                          

sys.path.append("../../src")
from HumanLearning import HLearning                                                 
from Sferes import EA
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *

p_order = ['length', 'noise', 'threshold', 'sigma']

parameters = {'length':9,
			'noise':0.001,
			'threshold':1.0, 
			'sigma':0.1}

with open("fmri/S9.pickle", "rb") as f:
   data = pickle.load(f)
model = BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])


model = BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes=True)

opt = EA(data, 'S9', model)

llh, lrs = opt.getFitness()

print llh, lrs




# figure(2)
# plot(opt.mean[0], 'o-')
# plot(opt.mean[1], 'o--')



# show()
