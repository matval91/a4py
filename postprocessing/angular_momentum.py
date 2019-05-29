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
from scipy.interpolate import interp1d
from utils.plot_utils import common_style

def _find_parabola(x1, y1, x2, y2, x3, y3):
	'''
	Adapted and modifed to get the unknowns for defining a parabola:
	http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
	'''
	denom = (x1-x2) * (x1-x3) * (x2-x3);
	A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
	B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
	C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;
	return A,B,C

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

def calculate_angmom(partdict, hdr, bkg):
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
	b_param, bphi, axis, edge = _param_B(hdr,bkg)
	rho = partdict['rho']
	PFxx = hdr['PFxx'] #fluxes at AXIS and X POINT
	polflux_norm = rho**2
	polflux = polflux_norm*(edge['flux']-axis['flux'])+axis['flux']

	m = partdict['m']
	R = partdict['R']
	vphi = np.copy(partdict['vphi'])
	Z = partdict['Z']
	#vphi *= m*1.6e-27/(Z*1.602e-19*b_param(R)*R)
	#canangmom = m*R*vphi-Ze*polflux
	mom_unit = 1.602e-19*b_param(axis['R'])*axis['R']**2
	fac1=m*1.66e-27*(R-axis['R'])*vphi/mom_unit
	fac2=Z*1.602e-19*polflux/mom_unit
	canangmom = (fac1-fac2)/edge['flux']
	return canangmom


def calculate_muE(partdict, hdr, bkg):
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
	E = m*(v_perp**2+partdict['vphi']**2)/2.
	mu = m*v_perp**2/(2.*B)
	b_param, bphi, axis, edge = _param_B(hdr,bkg)
	mu *= b_param(axis['R'])/E
	return mu

def _param_B(hdr, bkg):
	"""dict with B
	Create dictionary with B values

	Arguments:
		hdr
		bkg
	Parameters:
	"""
	bphi = np.abs(bkg['Bphi'][0,:]) #B must be decreasing with R, as defined in White book
	ind=np.where(bphi!=0)[0]
	bphi = bphi[ind]
	b_param = interp1d(bkg['R'][ind], bphi)	
	# poloidal flux must be from 0 to psi_edge, having maximum at psi_edge
	# since the hdr and bkg should be in COCOS5, psi should be increasing and thus I set to 0 the axis
	# remember the *2pi, done in building because of the ASCOT implementation
	#axis={'R':hdr['RPFx'][0], 'z':hdr['zPFx'][0], 'flux':hdr['PFxx'][0]-hdr['PFxx'][0]}
	#edge={'R':hdr['RPFx'][1], 'z':hdr['zPFx'][1], 'flux':(hdr['PFxx'][1]-hdr['PFxx'][0])/(2*np.pi)}
	edge={'R':hdr['RPFx'][0], 'z':hdr['zPFx'][0], 'flux':hdr['PFxx'][0]+hdr['PFxx'][1]}
	axis={'R':hdr['RPFx'][1], 'z':hdr['zPFx'][1], 'flux':0}
	
	return b_param, bphi, axis, edge

def _lost_HFS(E, hdr, bkg):
	"""
	see R. White, page 76

	Parabola with:
	maximum at mu=E/Bmin
	intercepts with (mu=0) at P=-psiedge+-sqrt(2E)g(edge)/Bmin
	where g=R*Btor

	Arguments:
		hdr
		bkg
	Parameters:
	"""
	b_param, bphi, axis, edge = _param_B(hdr,bkg)

	x1 = -edge['flux']
	y1 = E/np.min(bphi)/(E/b_param(axis['R']))
	# vacuum approximation
	x2 = -edge['flux']-np.sqrt(2*E)*(axis['R'])*b_param(axis['R'])/np.min(bphi)
	y2 = 0
	x3 = -edge['flux']+np.sqrt(2*E)*(axis['R'])*b_param(axis['R'])/np.min(bphi)
	y3 = 0
	A,B,C = np.polyfit([x1/edge['flux'],x2/edge['flux'],x3/edge['flux']],[y1,y2,y3],2)
	return A,B,C, x2/edge['flux'], x3/edge['flux']

def _lost_LFS(E, hdr, bkg):
	"""
	see R. White, page 76

	Parabola with:
	maximum at mu=E/Bmin
	intercepts with (mu=0) at P=-psiedge+-sqrt(2E)g(edge)/Bmin
	where g=R*Btor

	Arguments:
		hdr
		bkg
	Parameters:
	"""
	b_param, bphi, axis, edge = _param_B(hdr,bkg)
	mom_unit = 1.602e-19*b_param(axis['R'])*axis['R']**2
	#norm_v=2*1.66e-27*(R-axis['R'])/mom_unit
	norm_flux=1*1.602e-19/mom_unit
	x1 = -edge['flux']
	y1 = E/np.max(bphi)/(E/b_param(axis['R']))
	# vacuum approximation
	x2 = -edge['flux']-np.sqrt(2*E)*axis['R']*b_param(axis['R'])/np.max(bphi)
	y2 = 0
	x3 = -edge['flux']+np.sqrt(2*E)*axis['R']*b_param(axis['R'])/np.max(bphi)
	y3 = 0

	A,B,C = np.polyfit([x1/edge['flux'],x2/edge['flux'],x3/edge['flux']],[y1,y2,y3],2)
	#A,B,C = _find_parabola(x1/edge['flux'],y1,x2/edge['flux'],y2,x3/edge['flux'],y3)
	return A,B,C, x2/edge['flux'], x3/edge['flux']


def _magaxis(E, hdr, bkg):
	"""
	see R. White, page 76

	Parabola with:
	maximum at mu=E/Bmin
	intercepts with (mu=0) at P=-psiedge+-sqrt(2E)g(edge)/Bmin
	where g=R*Btor

	Arguments:
		hdr
		bkg
	Parameters:
	"""
	b_param, bphi, axis, edge = _param_B(hdr,bkg)

	x1 = 0
	y1 = 1
	# vacuum approximation
	x2 = -np.sqrt(2*E)*axis['R']
	y2 = 0
	x3 = np.sqrt(2*E)*axis['R']
	y3 = 0

	A,B,C = np.polyfit([x1/edge['flux'],x2/edge['flux'],x3/edge['flux']],[y1,y2,y3],2)
	#A,B,C = _find_parabola(x1/edge['flux'],y1,x2/edge['flux'],y2,x3/edge['flux'],y3)
	return A,B,C, x2/edge['flux'], x3/edge['flux']

def _trapp_passing(E, hdr, bkg):
	"""
	see R. White, page 76

	Parabola with:
	maximum at mu=E/Bmin
	intercepts with (mu=0) at P=-psiedge+-sqrt(2E)g(edge)/Bmin
	where g=R*Btor

	Arguments:
		hdr
		bkg
	Parameters:
	"""
	b_param, bphi, axis, edge = _param_B(hdr,bkg)

	x = -1*np.linspace(-1,0,num=100)**2
	#lower curve
	y_low = np.linspace(b_param(axis['R'])/np.max(bphi), 1, num=100)
	#upper curve
	y_up = np.linspace(b_param(axis['R'])/np.min(bphi), 1, num=100)

	return x, y_up, y_low

def plot_lossregion(E,hdr,bkg):
	"""plot the loss regions

	Parameters

	Returns

	"""
	common_style()
	A,B,C,x2,x3 = _lost_HFS(E,hdr,bkg)
	x_lost_HFS = np.linspace(x2,x3,1000)
	y_lost_HFS = np.polyval([A,B,C], x_lost_HFS)
	xmaxhfs=x3

	A,B,C,x2,x3 = _lost_LFS(E,hdr,bkg)
	x_lost_LFS = np.linspace(x2,x3,1000)
	y_lost_LFS = np.polyval([A,B,C], x_lost_LFS)

	A,B,C,x2,x3 = _magaxis(E,hdr,bkg)
	x_magaxis = np.linspace(x2,x3,1000)
	y_magaxis = np.polyval([A,B,C], x_magaxis)
	xminmagaxis=x2
	xtr, y_up, y_low = _trapp_passing(E,hdr,bkg)

	f=plt.figure(figsize=(8,8));
	ax=f.add_subplot(111)
	ax.plot(x_lost_HFS, y_lost_HFS, label='lost HFS')
	ax.plot(x_lost_LFS, y_lost_LFS, label='lost LFS')
	plt.fill_between(x_lost_HFS[x_lost_HFS<-1.],y_lost_HFS[x_lost_HFS<-1.], color='k', alpha=1)
	ax.plot(x_magaxis, y_magaxis, label='Mag. Axis')
	ax.plot(xtr, y_up, 'r', label='trapped')
	ax.plot(xtr, y_low, 'r')
	plt.fill_between(x_lost_HFS[x_lost_HFS>-1.],y_lost_HFS[x_lost_HFS>-1.], color='k', alpha=1)
	plt.fill_between(xtr[xtr>-1.], y_low[xtr>-1.], color='w')
	plt.fill_between(x_lost_LFS,y_lost_LFS, color='white')
	ax.vlines(x=-1, ymin=np.max(y_lost_LFS), ymax=np.max(y_lost_HFS), color='r')
	if xmaxhfs>xminmagaxis:
		print('There is overlap...need to plot')
	ax.set_xlabel(r'$P_\zeta$/$\psi_w$')
	ax.set_ylabel(r'$\frac{\mu B_0}{E}$')
	ax.set_title(r'E='+str(E))
	ax.legend(loc='best'); ax.grid('on')
	f.tight_layout()
