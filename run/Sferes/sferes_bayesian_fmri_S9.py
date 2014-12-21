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

# parameters = {'length':10.0,
# 			'noise':0.0,
# 			'threshold':0.01, 
# 			'sigma':0.282}

parameters = {'noise':0.116,
			'length':10.0,			
			'threshold':0.045,
			'sigma':100.0}
			

with open("fmri/S15.pickle", "rb") as f:

   data = pickle.load(f)

model = BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes=True)

opt = EA(data, 'S15', model)

llh, lrs = opt.getFitness()

# print llh, lrs
print float(llh)-2000.0

llh = float(llh)-2000.0
lrs = float(lrs)-500.0
llh = 2*llh-4.0*np.log(156)
print llh, lrs


# figure(2)
# plot(opt.mean[0], 'o-')
# plot(opt.mean[1], 'o--')



# show()
