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
    fname_hdr='../examples/input.magn_header_'
    fname_bkg='../examples/input.magn_bkg_'
    E=25
print('Read input', fname_hdr, '  ', fname_bkg, str(E))

bkg, hdr=chb.COM_hdrbkg(fname_hdr, fname_bkg, E)