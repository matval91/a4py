#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:14:21 2020

@author: vallar
"""
import a4py.postprocessing.COM_hdrbkg_class as chb
import sys

if len(sys.argv) == 4:
    fname_hdr=sys.argv[1]
    fname_bkg=sys.argv[2]
    E=float(sys.argv[3])
else:
    fname_hdr='../examples/input.magn_header_TCV'
    fname_bkg='../examples/input.magn_bkg_TCV'
    E=25
print('Read input', fname_hdr, '  ', fname_bkg, str(E))

data, B0, R0=chb.COM_hdrbkg(fname_hdr, fname_bkg, E,debug=True, plot=True)