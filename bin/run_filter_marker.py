#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:30:43 2020

@author: vallar
"""

import a4py.preprocessing.filter_marker as fm
import sys

if len(sys.argv) == 3:
    input_fname=sys.argv[1]
    max_markers=int(sys.argv[2])
else:
    fname_a4='../examples/ascot_TCV.h5'
    E=25
print('Read input', input_fname, str(max_markers))

fm.filter_marker(input_fname, fname_out='input.particles_npart'+str(max_markers), minrho=0., minxi=-1., maxxi=1., sign=1, max_markers=max_markers, maxrho=5.)
