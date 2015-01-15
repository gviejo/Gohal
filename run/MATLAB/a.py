


import sys
import os

import numpy as np

sys.path.append("../../src")
from HumanLearning import HLearning


human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',48), 'fmri':('../../fMRI',39)}))

ss = human.subject['meg'].keys()

for s in ss:
	os.system("mkdir "+s)
	
