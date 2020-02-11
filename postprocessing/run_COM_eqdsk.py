#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:14:21 2020

@author: vallar
"""
import a4py.postprocessing.COM_eqdsk_class as ce
import sys

if len(sys.argv) == 3:
    fname_eqdsk=sys.argv[1]
    E=float(sys.argv[2])
else:
    fname_eqdsk='../examples/EQDSK_58934t0.8000_COCOS03_COCOS03'
    E=25
print('Read input', fname_eqdsk, str(E))

eq=ce.COM_eqdsk(fname_eqdsk, E)