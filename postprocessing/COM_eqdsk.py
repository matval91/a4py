#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:54:54 2020

@author: matval
"""
import ReadEQDSK as re
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import sys
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=24, titlesize=24)
plt.rc('figure', facecolor='white')
plt.rc('legend', fontsize=15)
plt.rc('lines', linewidth= 2.3)

def run(fname_eqdsk, Ekev, debug=1, plot=1):
    """ COM boundaries
    Plot COM boundary spaces given eqdsk and Ekev
    """
    eq=re.ReadEQDSK(fname_eqdsk)
    #[_,eq]=ct.cocos_transform(eq, 3, 2)
    E=Ekev*1000*1.602e-19
    
    # Getting psi_2d (Normalized to edge and axis value) and interpolate it
    psiw = eq.psiedge; psia=eq.psiaxis;
    _R = np.linspace(eq.rboxleft, eq.rboxleft+eq.rboxlength, eq.nrbox)
    _z = np.linspace(-1.*eq.zboxlength/2., eq.zboxlength/2., eq.nzbox)
    psi2d_param = interp.interp2d(_R, _z, (eq.psi-psia)/(psiw-psia))
    #psi2d_param_notnorm = interp.interp2d(_R, _z, eq.psi)
    # Finding the axis R0 of the device, which is the one to use to normalize
    R0 = _R[np.argmin(psi2d_param(_R,0))]
    if debug:
        print('R0={:.2f} vs old R0={:.2f} \n'.format(R0, eq.R0EXP))
    #T is as function of psi (equidistant)
    psi = np.linspace(0,1,eq.nrbox)
    T_param = interp.interp1d(psi, eq.T)
    #Find psi at the R of the Rbox on midplane, Z=0
    psi_at_midplane = psi2d_param(_R, 0)
    #Limiting only inside the plasma
    R=_R[psi_at_midplane<1.]
    psi_at_midplane=psi_at_midplane[psi_at_midplane<1.]
    #Finding values of T and then B at the midplane, inside the plasma
    T_on_midplane = T_param(psi_at_midplane)
    B = T_on_midplane/R;
    #Forcing B to be positive and decreasing in R
    B = np.abs(B)
    Bmin = np.min(B); Bmax=np.max(B)
    
    Rmin = min(R); Rmax=max(R)
    B_paramR = interp.interp1d(R, B)
    B0 = B_paramR(R0)
    #finding also extrema of g
    g_param=T_param
    gedge = np.abs(g_param(1.))
    g0 = np.abs(g_param(0.))
    # We want psi increasing from 0 to psi_wall
    psi=eq.psi_grid
    if psiw<psia or psiw==0:
        psiw-=psia; psi-=psia; psia-=psia; # now stuff set from 0 to something.
        if psiw<0: 
            psiw=psiw*-1.; psi*=-1;
    ####################################################################
    #print values for debugging
    if debug=='a':
        print('Bmin={:.2f}; Bmax={:.2f}; B={:.2f}'.format(Bmin, Bmax, B0))
        print('gax={:.2f}; gedge={:.2f}; B0R0={:.2f}'.format(g0, gedge, R0*B0))
        print('psiw={:.2f}; psiax={:.2f}'.format(psiw, psia))
        
    # get normalized units
    mp=1.67e-27; q=1.602e-19;
    A=2; Z=1;
    R_notnorm=np.copy(R); 
    B/=B0; Bmin/=B0; Bmax/=B0; #normalizing B
    R/=R0; Rmin/=R0; Rmax/=R0; #Normalizing R
    gedge = gedge/(R0*B0)
    g0 = g0/(R0*B0)
    psiw = psiw/(R0*R0*B0)
    psia = psia/(R0*R0*B0)
    psi = psi/(R0*R0*B0)
    E = E*mp*A/(Z*Z*q*q*R0**2*B0**2)

    #print values for debugging
    if debug:
        print('After normalization')
        print('Bmin={:.2f}; Bmax={:.2f}; B={:.2f}'.format(Bmin, Bmax, B0))
        print('gax={:.2f}; gedge={:.2f}; B0R0={:.2f}'.format(g0, gedge, R0*B0))
        print('psiw={:.2f}; psiax={:.2f}; psi={:.2f}'.format(psiw, psia, np.mean(psi)))
        print('E={:.2e}'.format(E)) #E Looks ok.
        print()
        print('zero with bmin {:.3f}'.format(-1-np.sqrt(2*E)*gedge/(psiw*Bmin)))
        print('zero with bmin {:.3f}'.format(-1+np.sqrt(2*E)*gedge/(psiw*Bmin)))
        print('max with Bmin {:.3f}'.format(1/Bmin))
        print('zero with bmax {:.3f}'.format(-1-np.sqrt(2*E)*gedge/(psiw*Bmax)))
        print('zero with bmax {:.3f}'.format(-1+np.sqrt(2*E)*gedge/(psiw*Bmax)))
        print('max with Bmax {:.3f}'.format(1/Bmax))

    # Defining p_xi/psi_W
    x = np.linspace(-2., 1, 500)
    #Right hand edge of midplane
    #These functions should plot mu/E. You must retrieve mu/E from equations at page
    # 85 of RW book
    copss_lost = 1./Bmin-(1.+x)**2*(Bmin*psiw*psiw)/(2*gedge*gedge*E)
    # copss_lost*B0=1/(Bmin)-(Bmin/2.)*(psi**2*(1+x)**2)/(2*(R0*B0*gedge)**2*E)
    cntrpss_lost = 1/Bmax-(Bmax)*(psiw**2*(1+x)**2)/(2*gedge**2*E)
    magaxis = 1-(x*psiw)**2/(2*E*g0**2)
    
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

    if plot:
        f=plt.figure(figsize=(8,6))
        ax=f.add_subplot(111)
        ax.plot(x, copss_lost, 'k')
        ax.plot(x, cntrpss_lost,'k')
        ax.plot(x, magaxis,'b')
        ax.plot(trpp_up['x'], trpp_up['y'],'r')
        ax.plot(trpp_down['x'], trpp_down['y'],'r')
        ax.plot([-1,-1], [max(copss_lost), max(cntrpss_lost)], 'k--')
        ax.set_title(r'E={:.2f} keV'.format(Ekev))
        ax.set_xlabel(r'P$_\phi$/$\psi_w$')
        ax.set_ylabel(r'$\mu\frac{B_0}{E}$')
        ax.set_ylim([0, 1.5]); ax.set_xlim([-2, 1.])
        ax.grid('on')
        f.tight_layout()
        #f.savefig('COM_{:s}_E{:.2f}.png'.format(fname_eqdsk, Ekev), dpi=800)
    plt.show()
    
    return eq
    
# if len(sys.argv) == 3:
#     fname_eqdsk=sys.argv[1]
#     E=float(sys.argv[2])
# else:
#     fname_eqdsk='EQDSK_58934t0.8000_COCOS03_COCOS03'
#     E=25
# print('Read input', fname_eqdsk, str(E))
# main(fname_eqdsk, E)
