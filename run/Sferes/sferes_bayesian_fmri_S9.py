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

p_order = ['length', 'noise', 'threshold']

p = map(float, "0.5 0.5 0.5 0.5".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

parameters = {'length':9,
			'noise':0.001,
			'threshold':0.5}

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters)
# parameters = tmp
# for p in parameters.iterkeys():
# 	if parameters[p]:
# 		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
# model.setAllParameters(parameters)

opt = EA(human.subject['fmri']['S9'], 'S9', model)

llh, lrs = opt.getFitness()

print llh, lrs

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left+width+0.02
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]
plt.figure(1, figsize=(12,9))
axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)
axScatter.imshow(opt.p, interpolation = 'nearest', origin = 'lower')
axHistx.plot(opt.p_rtm)
#axHistx.set_title("dh/n")
axHistx.set_title("dh")
#axHistx.set_title("n")
axHisty.plot(opt.mass, np.arange(len(opt.mass)))
plt.text(s = "MI "+str(lrs), x = 0.0, y = 16.0)

#savefig("/home/viejo/Desktop/meeting_11_04_14/mutual_bayesian_h_n.pdf")
#savefig("/home/viejo/Desktop/meeting_11_04_14/mutual_bayesian_h.pdf")
#savefig("/home/viejo/Desktop/meeting_11_04_14/mutual_bayesian_n.pdf")


#plt.figure(2, figsize=(8,8))
rt = np.tile(opt.rt, opt.n_repets)
ind = np.argsort(rt)
# subplot(411)
# plot(rt[ind])
# subplot(212)
# plot(opt.rtm[ind])

#np.save("rtm_h_n", opt.rtm[ind])
#np.save("rtm_h", opt.rtm[ind])
np.save("rtm_n", opt.rtm[ind])
sys.exit()

hn = np.load("rtm_h_n.npy")
h = np.load("rtm_h.npy")
n = np.load("rtm_n.npy")

figure(figsize = (9, 14))
subplot(411)
plot(rt[ind])
subplot(412)
plot(hn)
title("h/n")
subplot(413)
plot(h)
title("h")
subplot(414)
plot(n)
title("n")
savefig("/home/viejo/Desktop/meeting_11_04_14/cumul_bayesian.pdf")
show()


show()

sys.exit()
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