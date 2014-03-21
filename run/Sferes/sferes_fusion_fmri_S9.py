#!/usr/bin/python                                                       
# encoding: utf-8                                                                   
import sys                                                                          

sys.path.append("../../src")                                            
from HumanLearning import HLearning                                                 
from Sferes import EA, RBM
from Models import *                                                                
from Selection import *
from matplotlib import *
from pylab import *
from scipy.stats import norm

p_order = ['alpha','beta', 'gamma', 'noise','length','gain','threshold']

#p = map(float, "0.784762 0.245554 0.815565 0 1 1 0.610446 0.170918".split(" "))
# p = map(float, "0.118246 0.897181 0.946751 0.595232 0.092146 0.922554 0.126166 0.341448".split(" "))
# tmp = dict()
# for i in p_order:
# 	tmp[i] = p[p_order.index(i)]

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])


# parameters = tmp
# for p in parameters.iterkeys():
# 	if parameters[p]:
# 		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])

parameters = dict({'noise':0.2,
                    'length':7,
                    'alpha':0.9,
                    'beta':3.5,
                    'gamma':0.5,
                    'gain':0.6,
                    'threshold':1.5})


model.parameters = parameters

opt = EA(human.subject['fmri']['S9'], 'S9', model)

llh, lrs = opt.getFitness()
print llh, lrs

sys.exit()
tirage = np.argmax(opt.rbm.xx[:,0:opt.rbm.nx], 1)
center = opt.edges[1:]-(opt.bin_size/2.)
rtm = center[tirage]

figure()
subplot(311)

plot(opt.rt,'o-')
plot(rtm,'o-')
rt = opt.rt
ind = np.argsort(rt)
subplot(312)
plot(rt[ind],'o-')
plot(rtm[ind],'o-')

subplot(313)
rtmean = []
for e in xrange(len(opt.rbm.m)):
	tirage = np.argmax(opt.rbm.m[e][:,0:opt.rbm.nx], 1)
	rtmean.append(center[tirage])
rtmean = np.array(rtmean)
errorbar(range(156),np.mean(rtmean, 0)[ind], yerr = np.std(rtmean, 0)[ind])

show()
