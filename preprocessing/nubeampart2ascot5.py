import sys
import pytransp.classes.transp_deposition as td
import numpy as np
import a5py.ascot5io.mrk_gc as mrk_gc

def main(fname= '../examples/65052V01_fi_1.cdf', h5fn='ascot.h5'):
    """
    """
    dep = td.absorption(fname, '')
    data = dep.data_i
    #keys = data.keys()
    
    # We have guiding centers (theta is random)
    print("Warning! Forcing time to zero and "
          "randomizing zeta for all markers.")
    zeta = 2*np.pi*np.random.rand(data["id"].size)
    mrk_gc.write_hdf5(
        fn=h5fn, n=data["id"].size, ids=data["id"],
        mass=data["mass"], charge=data["charge"],
        r=data["Rgc"], phi=data["phi"],
        z=data["zgc"],
        energy=data["E"], pitch=data["pitch"],
        zeta=zeta,
        anum=data['Anum'], znum=data['Znum'],
        weight=data["weight"],
        time=data["weight"]*0 )
    return

h5fn='ascot.h5'
if len(sys.argv)==2:
    fname = sys.argv[1]
elif len(sys.argv) ==3:
    fname = sys.argv[1]
    h5fn = sys.argv[2]
else:
    print("Please give as input a birth profile and an optional h5 filename")
    print('\n e.g. \n nubeampart2ascot5.py ../examples/65052V02_birth.cdf1 \n')
    sys.exit()
    
main(fname, h5fn)
