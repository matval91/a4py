#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:14:21 2020

@author: vallar
"""
import a4py.postprocessing.COM_ascot4_class as c4
import sys

if len(sys.argv) == 3:
    fname_a4=sys.argv[1]
    E=float(sys.argv[2])
else:
    fname_a4='../examples/ascot_TCV.h5'
    E=25
print('Read input', fname_a4, str(E))

eq=c4.COM_a4(fname_a4, E, debug=True, plot=True)