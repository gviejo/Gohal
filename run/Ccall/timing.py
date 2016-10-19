#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import cPickle as pickle

data = {}

with open("../Sferes/parameters_single.pickle", 'rb') as f:
	p_test = pickle.load(f)
m = 'bayesian'
data[m] = {}

for s in p_test[m]['tche'].iterkeys():
	value = os.popen("./timing_"+m+" data_fmri/"+s+"/ "+str(p_test[m]['tche'][s][m]['length'])+" "+str(p_test[m]['tche'][s][m]['noise'])+" "+str(p_test[m]['tche'][s][m]['threshold'])+" "+str(p_test[m]['tche'][s][m]['sigma'])).read()		
	value = value[:-1].split(" ")	
	data[m][s] = np.array([float(value[0]), float(value[1])])

m = 'qlearning'
data[m] = {}

for s in p_test[m]['tche'].iterkeys():
	value = os.popen("./timing_"+m+" data_fmri/"+s+"/ "+str(p_test[m]['tche'][s][m]['alpha'])+" "+str(p_test[m]['tche'][s][m]['beta'])+" "+str(p_test[m]['tche'][s][m]['sigma'])).read()		
	value = value[:-1].split(" ")	
	data[m][s] = np.array([float(value[0]), float(value[1])])	


with open("../Sferes/timing_single.pickle", 'wb') as f:
    pickle.dump(data, f)





