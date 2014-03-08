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

p_order = ['alpha','beta', 'gamma', 'noise','length','gain','sigma_bwm', 'sigma_ql']

p = map(float, "0.988166 0.35222 0.285128 0.770395 0.478959 0.0322654 0.957454 0.727134".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = tmp
for p in parameters.iterkeys():
	if parameters[p]:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model.setAllParameters(parameters)

opt = EA(human.subject['fmri']['S2'], 'S2', model)

llh, lrs = opt.getFitness()

print llh, lrs

figure()
subplot(211)
plot(opt.rt_model)
plot(opt.rt)

subplot(212)
m = opt.rt_model[15]
h = opt.rt[15]
x = np.arange(-10, 10, 0.01)
def f(x, u, v):
    return (1/np.sqrt(2*pi*v))*np.exp(-0.5*np.power((x-u)/v, 2))
[plot(x, f(x, i, model.parameters['sigma_ql'])) for i in [m[0]]]
[plot(x, f(x, i, model.parameters['sigma_bwm'])) for i in m[1:]]

axvline(h)
show()

