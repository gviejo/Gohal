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
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

parameters = dict({'noise':0.2,
                    'length':7,
                    'alpha':0.9,
                    'beta':3.5,
                    'gamma':0.5,
                    'gain':0.6,
                    'threshold':1.5})

human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
model = FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])

model.parameters = parameters

opt = EA(human.subject['fmri']['S9'], 'S9', model)

llh, lrs = opt.getFitness()
print llh, lrs

rt = opt.rt
rtm = (model.pdf.cumsum(1)<np.random.rand(156,1)).sum(1)

xmin = rt.min()
xmax = rt.max()

ymin = rtm.min()
ymax = rtm.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([rt, rtm])

kernel = stats.gaussian_kde(values)
kernel_rt = stats.gaussian_kde(rt)
kernel_rtm = stats.gaussian_kde(rtm)

Z = np.reshape(kernel(positions).T, X.shape)

fig = figure()
ax = fig.add_subplot(111,projection='3d')

#imshow(np.rot90(Z), cmap = cm.gist_earth_r, extent = [xmin, xmax, ymin, ymax])
ax.plot_surface(X, Y, Z)


#plot(rt, rtm, 'k.', markersize = 2)


figure()
x = np.arange(xmin, xmax, 0.01)
plot(x, kernel_rt(x))

figure()
x = np.arange(ymin, ymax, 0.01)
plot(x, kernel_rtm(x))

show()





