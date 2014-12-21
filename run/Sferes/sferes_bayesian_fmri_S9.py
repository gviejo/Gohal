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

parameters = {'length': 10.0,
 'noise': 0.0006302870000000001,
 'sigma': 0.37079983,
 'threshold': 0.79976619685399741}

with open("fmri/S2.pickle", "rb") as f:
   data = pickle.load(f)
model = BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])


model = BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes=True)

opt = EA(data, 'S2', model)

llh, lrs = opt.getFitness()


llh = float(llh)-2000.0
lrs = float(lrs)-500.0
llh = 2*llh-4.0*np.log(156)
print llh, lrs


# figure(2)
# plot(opt.mean[0], 'o-')
# plot(opt.mean[1], 'o--')



# show()
