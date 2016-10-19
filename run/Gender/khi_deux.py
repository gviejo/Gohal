#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import chi2_contingency


f = open("Subjects.csv", 'r')

data = []

for l in f:
	data.append(l.split("\n")[0].split(" ")[1:])

table = np.zeros((2, 3))
m_order = ['fusion', 'mixture', 'selection']
g_order = ['F', 'M']

for g, m in data:
	table[g_order.index(g),m_order.index(m)] += 1.0

chi2, p, dof, expected = chi2_contingency(table)






