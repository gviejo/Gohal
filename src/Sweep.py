#!/usr/bin/python
# encoding: utf-8
"""
Sweep.py

Class to explore parameters

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import numpy as np
from fonctions import *





class Sweep():
    """
    fonctions to explore parameters
    """
    
    def __init__(self):



    def chi2(obs1, obs2):
        assert list(np.unique(obs1)) == [0,1]
        assert list(np.unique(obs2)) == [0,1]

        
