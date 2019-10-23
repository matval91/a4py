import a4py.postprocessing.angular_momentum as am
import a4py.postprocessing.read_magnbkg as read_magnbkg 
import a4py.preprocessing.filter_marker as a4fm 
import matplotlib.pyplot as plt

print('Read markers')
m,_,_,_=a4fm.filter_marker('input.particles_pt_1e4', fname_out='')
print('Read B field')
hdr=read_magnbkg.read_header('input.magn_header') 
bkg=read_magnbkg.read_bkg('input.magn_bkg')

print('Calculate angular moment and mu')
pdict=am.convert_arrpart_to_dict(m) 
angmom=am.calculate_angmom(pdict,hdr, bkg) 
mue=am.calculate_muE(pdict, hdr, bkg) 

b_param, bphi, axis, edge = am._param_B(hdr,bkg)
E=85e3*1.602e-19
m=1.6e-27; q=1.602e-19
eu=m*(q*b_param(axis['R'])/m)**2*axis['R']**2  
am.plot_lossregion(E, 2,hdr, bkg)
#f=plt.figure()
ax=plt.gca(); ax.scatter(angmom, mue)
plt.show()
