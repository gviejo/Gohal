#!/usr/bin/python                                                                     
# encoding: utf-8                                                                     
"""                                                                                   
submit_job.py                                                                         
                                                                                      
to launch sferes job on cluster                                                       
                                                                                      
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.                              
"""

import sys,os
from optparse import OptionParser
sys.path.append("Gohal/src")
from HumanLearning import HLearning

# -----------------------------------                                                 
# ARGUMENT MANAGER                                                                    
# -----------------------------------                                                 
if not sys.argv[1:]:
   sys.stdout.write("Sorry: you must specify at least 1 argument")
   sys.stdout.write("More help avalaible with -h or --help option")
   sys.exit(0)
parser = OptionParser()
parser.add_option("-m", "--model", action="store", help="The name of the model to optimize", default=False)
parser.add_option("-t", "--time", action="store", help="Time of execution", default=False)
parser.add_option("-d", "--data", action="store", help="The data to fit", default=False)
parser.add_option("-o", "--output", action="store", help="Output directory", default=False)
(options, args) = parser.parse_args()
# -----------------------------------                                                 
# -----------------------------------                                                 
# HUMAN LEARNING                                                                      
# -----------------------------------                                                 
human = HLearning(dict({'meg':('Beh/MEG/Beh_Model/',48), 'fmri':('Beh/fMRI',39)}))
# -----------------------------------                                                 

# -----------------------------------                                                 
# POSSIBLE MODELS                                                                     
# -----------------------------------                                                 
models = dict({'bayesian':"BayesianWorkingMemory(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], vars(options))",
               'qlearning':"QLearning(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], vars(options))",
               'fusion':"FSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], vars(options))",
               'kalman':"KalmanQLearning(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], vars(options))",
               'keramati':"KSelection(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], vars(options))",
              })
# -----------------------------------
# -----------------------------------                                                 
# GENERATE BASH SCRIPTS                                                               
# -----------------------------------                                                 
list_subject = human.subject[options.data].keys()
n_params = len(models[options.model].split("options"))-1

s = list_subject[0]
filename = "submit_"+options.model+"_"+options.data+"_"+s+".sh"
f = open(filename, "w")
f.writelines("#!/bin/sh\n")
f.writelines("#PBS -N sferes_"+options.model+"_"+options.data+"_"+s+"\n")
f.writelines("#PBS -o /home/viejo/log/sferes_"+options.model+"_"+options.data+"_"+options.time+"_"+s
f.writelines("#PBS -b /home/viejo/log/sferes_"+options.model+"_"+options.data+"_"+options.time+"_"+s
f.writelines("#PBS -m abe\n")
f.writelines("#PBS -M guillaume.viejo@gmail.com\n")
f.writelines("#PBS -l walltime="+options.time+"\n")
#f.writelines("#PBS -l nodes=1:ppn=8"+"\n")                                           
f.writelines(("#PBS -l ncpus=1\n"))
f.writelines("#PBS -d /home/viejo\n")
f.writelines("#PBS -v PYTHONPATH=/home/viejo/lib/python/lib/python\n")
f.writelines("sferes2/trunk/build/debug/exp/"+options.model+"/"+options.model+" --subject="+s+" --data="+options.data+"\n")
f.close()
#-----------------------------------

# ----------------------------------                                                  
# GENERATE PYTHON SCRIPTS                                                             
# ----------------------------------                                                  
pythonfile = "/home/viejo/sferes2/trunk/exp/"+options.model+"/sferes_"+options.model+"_"+options.data+"_"+s+".py"
pf = open(pythonfile, 'w')                                                          
pf.write("""#!/usr/bin/python                                                       
# encoding: utf-8                                                                   
import sys                                                                          
import numpy as np                                                                  
from optparse import OptionParser                                                   
sys.path.append("/home/viejo/Gohal/src")                                            
from HumanLearning import HLearning                                                 
from Sweep import Sferes
from Models import *                                                                
from Selection import *
parser = OptionParser()                                                             
parser.add_option("-t", "--threshold", action="store", type = 'float')
parser.add_option("-l", "--length", action="store", type = 'float')
parser.add_option("-n", "--noise", action="store", type = 'float')
parser.add_option("-a", "--alpha", action="store", type = 'float')
parser.add_option("-b", "--beta", action="store", type = 'float')
parser.add_option("-g", "--gamma", action="store", type = 'float')
parser.add_option("-z", "--gain", action="store", type = 'float')
parser.add_option("-s", "--sigma", action = "store", type = 'float')                                
(options, args) = parser.parse_args()                                               
human = HLearning(dict({'meg':('/home/viejo/Beh/MEG/Beh_Model/',48), 'fmri':('/home/viejo/Beh/fMRI',39)}))

""")                                                                                
pf.writelines("model = "+models[options.model]+"\n")                                
pf.writelines("opt = Sferes(human.subject['"+options.data+"']['"+s+"'], '"+s+"', model)")
pf.writelines("""
llh, lrs = opt.getFitness()
print llh, lrs
""")                                                                          
pf.close()                                                                          

# ------------------------------------
# CREATE DIRECTORY RESULTS
# ------------------------------------
os.system("rm -r "+options.model)
os.system("mkdir "+options.model)
# ------------------------------------                                                
# SUBMIT                                                                              
# ------------------------------------                                                
os.system("chmod +x "+filename)
os.system("qsub "+filename)
os.system("rm "+filename)




