#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:15:54 2020

@author: vallar
"""
import sys
import a4py.postprocessing.COM_hdrbkg_class as chb

if len(sys.argv) == 5:
    fname_particles=sys.argv[1]
    fname_hdr=sys.argv[2]
    fname_bkg=sys.argv[3]
    E=float(sys.argv[4])
else:
    fname_particles='../examples/input.particles_TCV_10'
    fname_hdr='../examples/input.magn_header_'
    fname_bkg='../examples/input.magn_bkg_'
    E=25
    
print('Read input', fname_particles, fname_hdr, fname_bkg, str(E))
chb.COM_hdrbkg_markers(fname_particles, fname_hdr, fname_bkg, E)