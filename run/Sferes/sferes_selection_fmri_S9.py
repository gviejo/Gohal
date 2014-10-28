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

p_order = ["gamma", "beta", "eta", "length", "threshold", "noise", "sigma", "sigma_rt"]

p = map(float, "1 0.867443 0.2 0.506596 0.172783 0.0321344 0.2 1 0.2 0.1".split(" "))
tmp = dict()
for i in p_order:
	tmp[i] = p[p_order.index(i)]

#human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))
with open("fmri/S9.pickle", "rb") as f:
   data = pickle.load(f)
model = KSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'])
parameters = tmp
for p in parameters.iterkeys():
	if parameters[p] is not None:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model = KSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes=True)

opt = EA(data, 'S9', model)

llh, lrs = opt.getFitness()

print llh, lrs

# figure(2)
# plot(opt.mean[0], 'o-')
# plot(opt.mean[1], 'o--')







# # left, width = 0.1, 0.65
# # bottom, height = 0.1, 0.65
# # bottom_h = left_h = left+width+0.02
# # rect_scatter = [left, bottom, width, height]
# # rect_histx = [left, bottom_h, width, 0.2]
# # rect_histy = [left_h, bottom, 0.2, height]
# # plt.figure(1, figsize=(12,9))
# # axScatter = plt.axes(rect_scatter)
# # axHistx = plt.axes(rect_histx)
# # axHisty = plt.axes(rect_histy)
# # axScatter.imshow(opt.p, interpolation = 'nearest', origin = 'lower')
# # axHistx.plot(opt.p_rtm)
# # axHisty.plot(opt.mass, np.arange(len(opt.mass)))
# # #axHistx.set_title("dh/n")
# # #axHistx.set_title("dh")
# # axHistx.set_title("n")
# # plt.text(s = "MI "+str(lrs), x = 0.0, y = 16.0)



# show()