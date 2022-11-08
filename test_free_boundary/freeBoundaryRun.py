#!/usr/bin/env python3
import os
import shutil
import numpy as np
import booz_xform as bx
from simsopt import load
from pathlib import Path
from subprocess import run
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from simsopt.util import MpiPartition
from simsopt.field.coil import coils_to_makegrid
from simsopt.field.mgrid import MGrid
from simsopt.mhd import Vmec, Boozer, QuasisymmetryRatioResidual, VirtualCasing
from simsopt.geo import curves_to_vtk
this_path = str(Path(__file__).parent.resolve())
mpi = MpiPartition()

create_mgrid_files = True

# folder = 'optimization_CNT'
# QA_or_QH = 'QA'
# full_torus = True

folder = 'optimization_QH'
QA_or_QH = 'QH'
full_torus = False

finite_beta = False

ncoils = 4
nphi = 256
ntheta = 128

dir = os.path.join(this_path,folder)

mgrid_executable = '/Users/rogeriojorge/bin/xgrid'
vmec_executable = '/Users/rogeriojorge/bin/xvmec2000'

if finite_beta: dir += '_finitebeta'
filename_final = os.path.join(dir,'input.final')
outdir =os.path.join(dir,'output')
outdir_coils = os.path.join(dir,'coils')

#####################
# Create mgrid files
#####################
if create_mgrid_files:
    vmec_final = Vmec(filename_final, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta)#, range_surface='half period')
    nfp = vmec_final.indata.nfp
    s_final = vmec_final.boundary
    bs_final = load(os.path.join(outdir_coils,"biot_savart_opt.json"))
    ncoils_total = len(bs_final.coils)

    r0 = np.sqrt(s_final.gamma()[:, :, 0] ** 2 + s_final.gamma()[:, :, 1] ** 2)
    z0 = s_final.gamma()[:, :, 2]
    nzeta = 45
    nr = 47
    nz = 49

    rmin=0.9*np.min(r0)
    rmax=1.1*np.max(r0)
    zmin=1.1*np.min(z0)
    zmax=1.1*np.max(z0)

    print('Creating to_mgrid file')
    bs_final.to_mgrid(os.path.join(outdir_coils,'tomgrid_opt_coils.nc'), nr=nr, nphi=nzeta, nz=nz, rmin=rmin, rmax=rmax, zmin=zmin, zmax=zmax, nfp=nfp)
    print('Done')

    if full_torus:
        curves = [c.curve for c in bs_final.coils]
        currents = [c.current for c in bs_final.coils]
    else:
        curves = [c.curve for c in bs_final.coils[0:ncoils]]
        currents = [c.current for c in bs_final.coils[0:ncoils]]

    if full_torus:
        coils_to_makegrid(os.path.join(outdir_coils,'coils.opt_coils'), curves, currents, true_nfp=nfp)
    else:
        coils_to_makegrid(os.path.join(outdir_coils,'coils.opt_coils'), curves, currents, nfp=nfp, stellsym=True)

    with open(os.path.join(outdir_coils,'input_xgrid.dat'), 'w') as f:
        f.write('opt_coils\n')
        f.write('S\n')
        f.write('y\n')
        f.write(f'{rmin}\n')
        f.write(f'{rmax}\n')
        f.write(f'{zmin}\n')
        f.write(f'{zmax}\n')
        f.write(f'{nzeta}\n')
        f.write(f'{nr}\n')
        f.write(f'{nz}\n')

    print("Running makegrid")
    os.chdir(outdir_coils)
    run_string = f"{mgrid_executable} < {os.path.join(outdir_coils,'input_xgrid.dat')} > {os.path.join(outdir_coils,'log_xgrid.opt_coils')}"
    run(run_string, shell=True, check=True)
    os.chdir(this_path)
    print(" done")

#####################
#####################

os.chdir(outdir_coils)

mgrid1_file = os.path.join(outdir_coils,'tomgrid_opt_coils.nc')
mgrid2_file = os.path.join(outdir_coils,'mgrid_opt_coils.nc')

mgrid1 = MGrid().from_file(mgrid1_file)
mgrid2 = MGrid().from_file(mgrid2_file)

assert mgrid1.bp.shape == mgrid2.bp.shape
assert mgrid1.br.shape == mgrid2.br.shape
assert mgrid1.bz.shape == mgrid2.bz.shape

diff_arrays = mgrid1.bp-mgrid2.bp
ind_max_diff = np.unravel_index(diff_arrays.argmax(),diff_arrays.shape)
print('ind_max_diff bp =',ind_max_diff,'with diff mgrid1-mgrid2:',diff_arrays[ind_max_diff])
diff_arrays = mgrid1.br-mgrid2.br
ind_max_diff = np.unravel_index(diff_arrays.argmax(),diff_arrays.shape)
print('ind_max_diff br =',ind_max_diff,'with diff mgrid1-mgrid2:',diff_arrays[ind_max_diff])
diff_arrays = mgrid1.bz-mgrid2.bz
ind_max_diff = np.unravel_index(diff_arrays.argmax(),diff_arrays.shape)
print('ind_max_diff bz =',ind_max_diff,'with diff mgrid1-mgrid2:',diff_arrays[ind_max_diff])

mgrid1.plot(show=False)
mgrid2.plot(show=False)
plt.show()

os.chdir(this_path)

#########

