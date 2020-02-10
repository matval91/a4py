#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute the angular momentum and mangetic moment of the particles

Angular momentum
P_ki = m x R x V_ki - Z x e x psi
ki=toroidal direction
psi=poloidal flux

Magnetic moment
mu = E_perp/B = m x V_perp**2 / (2 x B)
"""
import numpy as np
import matplotlib.pyplot as plt 
from utils.plot_utils import common_style
import a4py.preprocessing.filter_marker as a4fm 
import a4py.postprocessing.COM_eqdsk as COM_eqdsk
import sys

def main(fname_particles='../examples/input.particles_pt_1e4', 
         fname_eqdsk = '../examples/EQDSK_58934t0.8000_COCOS03_COCOS03', E=85):
    """
    """
    try:
        print('Read markers')
        mm,_,_,_=a4fm.filter_marker(fname_particles, fname_out='')
        #ind = mm[:,9]<-1.5e6 #filter on rho
        #v=np.sqrt(mm[:,9]**2+mm[:,10]**2+mm[:,11]**2)
        #pitch=mm[:,9]/v
#        ind=np.abs(pitch)<0.2
#        mm = mm[ind,:]
    except:
        print('No markers detected')
        mm = None

    eq=COM_eqdsk.run(fname_eqdsk, E, 0,1)

    if mm is not None:
        pdict=convert_arrpart_to_dict(mm)
        angmom=calculate_angmom(pdict, eq)
        mu=calculate_mu(pdict)
    else:
        angmom=[0]; mu=[0]

    mom_unit, energy_unit = _momentum_unit(eq)
    R0=eq.R0EXP
    B0=np.abs(eq.B0EXP)
    print(R0, B0)
    ax=plt.gca();
    x=angmom/(mom_unit*np.max(np.abs([eq.psiaxis, eq.psiedge]))/(R0*R0*B0))
    y=mu*np.abs(eq.B0EXP)/energy_unit
    ax.scatter(x, y)
    print(x,y)
    #plt.show()

def _momentum_unit(eq):
    """
    Calculation of fkng units!
    E_unit = m*omega_0**2*R**2 = (m*v**2/2)*(2*R**2/rho**2)
    rho = mv/qB
    """
    mp=1.66e-27;
    q=1.602e-19;
    B = eq.B0EXP
    R = eq.R0EXP

    omega_0 = q*B/mp
    mom_unit=omega_0*mp*R**2
    energy_unit = mp*omega_0**2*R**2
    return mom_unit, energy_unit

def convert_arrpart_to_dict(particles):
    """	
    Converts the array of particles to a dict

    partdict = convert_arrpart_to_dict(particles)

    Arguments:
        particles (arr) : particles object read from input.particles
    Parameters:
        partdict (dict): dict with the variables
    """
    partdict = {}
    partdict['m'] = particles[:,0]
    partdict['Z'] = particles[:,2]
    partdict['rho'] = particles[:,5]
    partdict['R'] = particles[:,7]
    partdict['vphi'] = particles[:,9]
    partdict['vR'] = particles[:,10]
    partdict['vz'] = particles[:,11]
    partdict['Bphi'] = particles[:,16]
    partdict['BR'] = particles[:,17]
    partdict['Bz'] = particles[:,18]

    return partdict

def calculate_angmom(partdict, eq):
    """calc pphi
    Script to calculate canonical angular momentum, defined as
    P_ki = m x R x V_ki - Z x e x psi
    ki=toroidal direction
    psi=poloidal flux
    
    pphi = calculate_angmom(particles)
    
    The canonical angular momentum dimensionally is [kg*m2*s-1]=[E][dt]
    The poloidal flux dimensionally is [Vs]
    pol.flux x charge x R = [V dt][q][dx] = [F][dt][dx] = [E][dt]

    Arguments
        partdict (dict): dict with the variables
        hdr (dict) : magnetic field with psiaxis and psiedge (poloidal fluxes) 
    Parameters


    """
    rho = partdict['rho']
    polflux_norm = rho**2

    psia = eq.psiaxis; psiw=eq.psiedge
    if psiw<psia or psiw==0:
        psiw-=psia; psia-=psia; # now stuff set from 0 to something.
        if psiw<0: 
            psiw=psiw*-1.;

    polflux = polflux_norm*(psiw-psia)+psia

    m = partdict['m']
    R = partdict['R']
    vphi = np.copy(partdict['vphi'])
    Z = partdict['Z']
    #vphi *= m*1.6e-27/(Z*1.602e-19*b_param(R)*R)
    canangmom = m*1.66e-27*R*vphi-Z*1.602e-19*polflux
    canangmom = np.array(canangmom)
    return canangmom

def calculate_mu(partdict):
    """calc mu
    Calculates the magnetic moment of the particles

    mu = E_perp/B = m x V_perp**2 / (2 x B)

    mu=calculate_mu(partdict)

    Arguments:
        partdict (dict): dict with the variables
    Parameters:
        mu
    """
    m = 1.67262e-27 * partdict['m']
    v_perp = np.sqrt(partdict['vR']**2+partdict['vz']**2)
    B = np.sqrt(partdict['Bphi']**2+partdict['BR']**2+partdict['Bz']**2)
    mu = m*v_perp**2/(2.*B)
    return mu

if len(sys.argv) == 4:
    fname_particles=sys.argv[1]
    fname_eqdsk=sys.argv[2]
    E=float(sys.argv[3])
else:
    fname_particles='../examples/input.particles_pt_1e4'
    fname_eqdsk = '../examples/EQDSK_58934t0.8000_COCOS03_COCOS03'
    E=25
print('Read input', fname_particles, fname_eqdsk, str(E))
main(fname_particles, fname_eqdsk, E)