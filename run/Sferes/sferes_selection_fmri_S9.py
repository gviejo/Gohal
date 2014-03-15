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

p_order = ["gamma", "beta", "eta", "length", "threshold", "noise", "sigma"]

p = map(float, "1 0.867443 0.2 0.506596 0.172783 0.0321344 0.2 1 0.2".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = KSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = tmp
for p in parameters.iterkeys():
	if parameters[p]:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model.setAllParameters(parameters)

opt = EA(human.subject['fmri']['S2'], 'S2', model)

llh, lrs = opt.getFitness()

print llh, lrs

c, n = np.histogram(opt.rt, opt.edges)
c = c.astype('float')
c = c/c.sum()
n = n[1:]-((n[1]-n[0])/2)

figure()
subplot(311)

plot(opt.rt)

xx = opt.rbm.xx
xx = (xx.T/xx.sum(axis=1)).T

rtm = np.array([n[(xx[i].cumsum()<np.random.rand()).sum()] for i in xrange(len(xx))])
plot(rtm)


subplot(312)
m = opt.rt_model[0]



plot(n, c, 'o-')
tmp = opt.rbm.xx.sum(0)
tmp = tmp/tmp.sum()
plot(n, tmp, 'o--')

subplot(313)
plot(opt.rbm.Error)


show()