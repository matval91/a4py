#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:54:54 2020

@author: matval
"""
import a4py.classes.ReadEQDSK as re
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
fname_eqdsk = '../examples/EQDSK_58934t0.8000_COCOS03_COCOS03'
eq=re.ReadEQDSK(fname_eqdsk)
Ekev=12
E=Ekev*1000*1.602e-19

# Getting values
# Values for normalization
B0 = np.abs(eq.B0EXP)
R0 = eq.R0EXP

#Other values needed
psi = np.linspace(0,1,eq.nrbox)
#get parameters for psi2D
_R = np.linspace(eq.rboxleft, eq.rboxleft+eq.rboxlength, eq.nrbox)
_z = np.linspace(-1.*eq.zboxlength/2., eq.zboxlength/2., eq.nzbox)
psi2d_param = interp.interp2d(_R, _z, (eq.psi-eq.psiaxis)/(eq.psiedge-eq.psiaxis))
psiw = eq.psiedge; psia=eq.psiaxis;

#T is as function of psi (equidistant)
T_param = interp.interp1d(psi, eq.T)
#Find psi at the R of the Rbox on midplane
#psi_at_outer_midplane = psi2d_param(np.linspace(R0, np.max(_R), np.size(eq.p)),0)
R_grid = np.linspace(np.min(eq.R), np.max(eq.R), np.size(eq.T))

psi_at_midplane = psi2d_param(R_grid, 0)
ind=psi_at_midplane<1.
R_grid=R_grid[ind]
R=R_grid
psi_at_midplane=psi_at_midplane[psi_at_midplane<1.]
T_on_midplane = T_param(psi_at_midplane)
B = np.abs(T_on_midplane/R_grid); Bmin = np.min(B); Bmax=np.max(B)
Rmin = min(R_grid); Rmax=max(R_grid)
B_paramR = interp.interp1d(R_grid, B)
B0 = B_paramR(R0)
gedge = np.abs(eq.T[-1])
g_param=T_param
g0 = g_param(R0)

# This is so to have psi=[0, psiw(>0)]
if psiw<psia or psiw==0:
    psiw-=psia; psia-=psia; # now stuff set from 0 to something.
    if psiw<0: psiw=psiw*-1.


# get normalized units
mp=1.66e-27; q=1.602e-19;
A=2; Z=1;
R_notnorm=np.copy(R); B_notnorm=np.copy(B);
B/=B0; Bmin/=B0; Bmax/=B0; #normalizing B
R/=R0; Rmin/=R0; Rmax/=R0; #Normalizing R
gedge = gedge/(R0*B0)
g0 = g0/(R0*B0)
psiw = psiw/(R0*R0*B0)
psia = psia/(R0*R0*B0)
psi = np.abs(eq.psi_grid)/(R0*R0*B0)
E = E*mp*A/(Z*Z*q*q*R0**2*B0**2)



# Defining p_xi/psi_W
x = np.linspace(-2., 1.5, 100)
#Right hand edge of midplane
#These functions should plot mu/E. You must retrieve mu/E from equations at page
# 85 of RW book
copss_lost = 1/Bmin-(Bmin/2.)*(psiw**2*(1+x)**2)/(2*gedge**2*E)
cntrpss_lost = 1/Bmax-(Bmax/2.)*(psiw**2*(1+x)**2)/(2*gedge**2*E)
magaxis = 1-(x*psiw)**2*B0/(2*g0**2*E)

#Normalization
#Trapped/passing boundary - UPPER
trpp_up={}
#step 1: find R(z=0, theta=0) at the psi wanted
psi_z0 = psi2d_param(R_notnorm[R_notnorm>=R0], 0)
#Normalization
psi_z0 = np.abs(psi_z0/R0*R0*B0)
psi_z0 = psi_z0[psi_z0<1.]
trpp_up['x'] = -1.*psi_z0;
#trpp_up['x']=np.flipud(trpp_up['x'])
# step 2 : find B at the R>R0, with normalizations
B_theta0 = B_paramR(np.linspace(R0, max(R_notnorm), np.size(psi_z0))); B_theta0/=B0;
trpp_up['y'] = (1./B_theta0);

#Trapped/passing boundary - LOWER
trpp_down={}
#step 1: find R(z=0, theta=pi) at the psi wanted
psi_zpi = psi2d_param(R_notnorm[R_notnorm<=R0], 0)
#Normalization
psi_zpi = np.abs(psi_zpi/R0*R0*B0)
psi_zpi = psi_zpi[psi_zpi<1.]
trpp_down['x'] = -1.*psi_zpi;
#trpp_down['x']=np.flipud(trpp_down['x'])
# step 2 : find B at the R>R0, with normalizations
B_thetapi = B_paramR(np.linspace(min(R_notnorm), R0, np.size(psi_zpi))); B_thetapi/=B0;
trpp_down['y'] = (1/B_thetapi);


f=plt.figure()
ax=f.add_subplot(111)
ax.plot(x, copss_lost, 'k')
ax.plot(x, cntrpss_lost,'r')
ax.plot(x, magaxis,'b')
ax.plot(trpp_up['x'], trpp_up['y'],'b')
ax.plot(trpp_down['x'], trpp_down['y'],'b')
ax.plot([-1,-1], [max(copss_lost), max(cntrpss_lost)], 'k--')
ax.set_title(r'E={:.2f} keV'.format(Ekev))
ax.set_ylim([0, 1.5])
ax.grid('on')
f.tight_layout()
plt.show()