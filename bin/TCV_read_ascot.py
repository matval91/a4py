import a4py.classes.prof as a4p
import a4py.classes.Bfield as a4b
import os
import re
# deal with specific folder
fullpathname=os.getcwd()
pathname=fullpathname.replace("/home/stipani/Documents/FILD/simulations/ASCOT/","")
eqid=int(pathname[2:3])
shot=int(pathname[5:10])
time=float(pathname[11:15])
for file in os.listdir(fullpathname):
    if file.startswith('EQDSK'):
        mm=[int(m) for m in re.findall(r'\d+',file)]
        cocos=mm[3]
        eqdskpath=os.path.join(fullpathname,file)
# query MDS for reading profiles and writing as ascot input file
dictIn={'shot':shot,'t':time,'nrho':50,'zeff':2}
profiles=a4p.TCV_mds(dictIn)
profiles.read_2d()
profiles.write_input()
profiles.plot_profiles()
# build the magnetic equilibrium file as ascot input file
magnetic=a4b.Bfield_eqdsk(eqdskpath,259,259,COCOS=cocos,devnam='TCV')
magnetic.plot_B()
whicheq=input('Is it a LIM or SN plasma?')
if whicheq=='LIM':
    magnetic.build_lim()
elif whicheq=='SN':
    magnetic.build_SN()
else:
    raise ValueError("input must be either LIM or SN")
magnetic.write(suff=str(eqid))
