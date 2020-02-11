#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:15:54 2020

@author: vallar
"""
import sys
import a4py.postprocessing.COM_eqdsk_class as ce

if len(sys.argv) == 4:
    fname_particles=sys.argv[1]
    fname_eqdsk=sys.argv[2]
    E=float(sys.argv[3])
else:
    fname_particles='../examples/input.particles_TCV_10'
    fname_eqdsk = '../examples/EQDSK_58934t0.8000_COCOS03_COCOS03'
    E=25
print('Read input', fname_particles, fname_eqdsk, str(E))
ce.COM_eqdsk_markers(fname_particles, fname_eqdsk, E)