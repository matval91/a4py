import a4py.classes.prof as a4p
import a4py.classes.Bfield as a4b
import os, re, getpass
import glob, sys

if len(sys.argv) >= 3:
    shot = sys.argv[1]
    time = float(sys.argv[2])
    if len(sys.argv) == 4:
        pathname = sys.argv[3]
else:
    print("Please give as input shot number, time, (folder)")
    print("Files (i.e. EQDSK) will be looked for in folder, if given ")
    print('\n e.g. \n TCV_read_ascot.py 62124 1.3 /tmp/vallar \n')
    sys.exit()
eqid='{}_t{:1.4f}'.format(shot, time)
# deal with specific folder
try:
    for file in os.listdir(pathname):
        if file.startswith('EQDSK_'):
            mm=[int(m) for m in re.findall(r'\d+',file)]
            if mm[0]==int(shot):
                cocos=mm[3]
                eqdskpath=os.path.join(pathname,file)
except NameError:
    print('Generating EQDSK file, hope you are in lac \n')
    user=getpass.getuser()
    cocos=17
    ## Produce eqdsk file
    os.system("matlab -nodisplay -r \"eq=gdat({}, 'eqdsk', 'time', {});exit\"".format(shot, time))
    eqdskpath = '/tmp/{}/EQDSK_{}t{:1.4f}_COCOS17'.format(user, shot, time)
    print('eqdskpath = '+eqdskpath)

# query MDS for reading profiles and writing as ascot input file
dictIn={'shot':int(shot),'t':time,'nrho':50,'zeff':2}
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
