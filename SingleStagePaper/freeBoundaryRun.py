#!/usr/bin/env python3
import os
import shutil
import numpy as np
import booz_xform as bx
from simsopt import load
from pathlib import Path
from subprocess import run
import matplotlib.pyplot as plt
from simsopt.util import MpiPartition
from simsopt.field.coil import coils_to_makegrid
from simsopt.mhd import Vmec, Boozer, QuasisymmetryRatioResidual, VirtualCasing
from simsopt.geo import curves_to_vtk
this_path = str(Path(__file__).parent.resolve())
mpi = MpiPartition()

# folder = 'optimization_CNT'
folder = 'optimization_CNT_circular'
QA_or_QH = 'QA'
full_torus = True

# folder = 'optimization_QH'
# QA_or_QH = 'QH'
# full_torus = False

ncoils = 4
nphi = 256
ntheta = 128
finite_beta = True

dir = os.path.join(this_path,folder)

mgrid_executable = '/Users/rogeriojorge/bin/xgrid'
vmec_executable = '/Users/rogeriojorge/bin/xvmec2000'

if finite_beta: dir += '_finitebeta'
filename_final = os.path.join(dir,'input.final')
outdir =os.path.join(dir,'output')
outdir_coils = os.path.join(dir,'coils')

vmec_final = Vmec(filename_final, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta)#, range_surface='half period')
nfp = vmec_final.indata.nfp
s_final = vmec_final.boundary
bs_final = load(os.path.join(outdir_coils,"biot_savart_opt.json"))
ncoils_total = len(bs_final.coils)

r0 = np.sqrt(s_final.gamma()[:, :, 0] ** 2 + s_final.gamma()[:, :, 1] ** 2)
z0 = s_final.gamma()[:, :, 2]
nzeta = 101
nr = 101
nz = 101
# print('Creating to_mgrid file')
# bs_final.to_mgrid(os.path.join(outdir_coils,'tomgrid_opt_coils.nc'), nr=nr, nphi=nzeta, nz=nr, rmin=0.9*np.min(r0), rmax=1.1*np.max(r0), zmin=1.1*np.min(z0), zmax=1.1*np.max(z0), nfp=nfp)
# print('Done')

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

r0 = np.sqrt(s_final.gamma()[:, :, 0] ** 2 + s_final.gamma()[:, :, 1] ** 2)
z0 = s_final.gamma()[:, :, 2]
with open(os.path.join(outdir_coils,'input_xgrid.dat'), 'w') as f:
    f.write('opt_coils\n')
    f.write('S\n')
    f.write('y\n')
    f.write(f'{0.9*np.min(r0)}\n')
    f.write(f'{1.1*np.max(r0)}\n')
    f.write(f'{1.1*np.min(z0)}\n')
    f.write(f'{1.1*np.max(z0)}\n')
    f.write(f'{nzeta}\n')
    f.write(f'{nr}\n')
    f.write(f'{nz}\n')

print("Running makegrid")
os.chdir(outdir_coils)
run_string = f"{mgrid_executable} < {os.path.join(outdir_coils,'input_xgrid.dat')} > {os.path.join(outdir_coils,'log_xgrid.opt_coils')}"
run(run_string, shell=True, check=True)
os.chdir(this_path)
print(" done")

vmec_final.indata.lfreeb = True
vmec_final.indata.mgrid_file = os.path.join(outdir_coils,'mgrid_opt_coils.nc')
# vmec_final.indata.mgrid_file = os.path.join(outdir_coils,'tomgrid_opt_coils.nc')
vmec_final.indata.extcur[0:ncoils_total] = [c.current.get_value() for c in bs_final.coils]
vmec_final.indata.nvacskip = 6
vmec_final.indata.nzeta = nzeta
vmec_final.indata.phiedge = -np.abs(vmec_final.indata.phiedge)

vmec_final.indata.ns_array[:4]    = [   9,    29,    49,   101]
vmec_final.indata.niter_array[:4] = [4000,  6000,  6000,  8000]
vmec_final.indata.ftol_array[:4]  = [1e-5,  1e-6, 1e-12, 1e-15]

vmec_final.write_input(os.path.join(dir,'input.final_freeb'))

print("Running VMEC")
os.chdir(os.path.join(dir,'vmec'))
run_string = f"{vmec_executable} {os.path.join(dir,'input.final_freeb')}"
# run(run_string, shell=True, check=True)
try: shutil.move(os.path.join(dir, 'vmec', f"wout_final_freeb.nc"), os.path.join(dir, f"wout_final_freeb.nc"))
except Exception as e: print(e)
os.chdir(this_path)

boozxform_nsurfaces=10
helical_detail=True
print("Plotting VMEC result")
if os.path.isfile(os.path.join(dir, f"wout_final_freeb.nc")):
    print('Found final vmec file')
    print("Plot VMEC result")
    import vmecPlot2
    vmecPlot2.main(file=os.path.join(dir, f"wout_final_freeb.nc"), name='free_b', figures_folder=outdir, coils_curves=[c.curve for c in bs_final.coils])
    vmec_freeb = Vmec(os.path.join(dir, f"wout_final_freeb.nc"), nphi=nphi, ntheta=ntheta)
    quasisymmetry_target_surfaces = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    if QA_or_QH == 'QA':
        qs = QuasisymmetryRatioResidual(vmec_freeb, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=0)
    else:
        qs = QuasisymmetryRatioResidual(vmec_freeb, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
    print('####################################################')
    print('####################################################')
    print('Quasisymmetry objective free boundary =',qs.total())
    print('Mean iota free boundary =',vmec_freeb.mean_iota())
    print('####################################################')
    print('####################################################')
    print('Creating Boozer class for vmec_final')
    b1 = Boozer(vmec_freeb, mpol=64, ntor=64)
    print('Defining surfaces where to compute Boozer coordinates')
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    print(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    print('Running BOOZ_XFORM')
    os.chdir(os.path.join(dir,'vmec'))
    try:
        b1.run()
        b1.bx.write_boozmn(os.path.join(dir,'vmec',"boozmn_free_b.nc"))
        print("Plot BOOZ_XFORM")
        fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
        plt.savefig(os.path.join(outdir, "Boozxform_surfplot_1_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
        plt.savefig(os.path.join(outdir, "Boozxform_surfplot_2_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
        plt.savefig(os.path.join(outdir, "Boozxform_surfplot_3_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
        plt.savefig(os.path.join(outdir, "Boozxform_symplot_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
        plt.savefig(os.path.join(outdir, "Boozxform_modeplot_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    except Exception as e: print(e)

    bs_final = load(os.path.join(outdir_coils,"biot_savart_opt.json"))
    s_final = vmec_freeb.boundary
    B_on_surface_final = bs_final.set_points(s_final.gamma().reshape((-1, 3))).AbsB()
    # norm_final = np.linalg.norm(s_final.normal().reshape((-1, 3)), axis=1)
    # meanb_final = np.mean(B_on_surface_final * norm_final)/np.mean(norm_final)
    # absb_final = bs_final.AbsB().reshape(s_final.gamma().shape[:2] + (1,))
    Bbs = bs_final.B().reshape((nphi, ntheta, 3))
    vc_src_nphi = int(nphi/2/vmec_freeb.wout.nfp)
    vc_final = VirtualCasing.from_vmec(vmec_freeb, src_nphi=vc_src_nphi, src_ntheta=ntheta)
    BdotN_surf = np.sum(Bbs * s_final.unitnormal(), axis=2) - vc_final.B_external_normal_extended
    # pointData_final = {"B·n/|B|": BdotN_surf[:, :, None]/absb_final,
    #             "|B|": bs_final.AbsB().reshape(s_final.gamma().shape[:2] + (1,))/meanb_final}
    curves = [c.curve for c in bs_final.coils]
    curves_to_vtk(curves, os.path.join(outdir_coils, "curves_freeb"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    s_final.to_vtk(os.path.join(outdir_coils, "surf_freeb"), extra_data=pointData)

    os.chdir(this_path)

files_to_remove = [os.path.join(this_path,'input.final_000_000000'),
                   os.path.join(this_path,'threed1.final'),
                   os.path.join(outdir_coils,'mgrid_opt_coils.nc'),
                   os.path.join(outdir_coils,'tomgrid_opt_coils.nc')]
for file in files_to_remove:
    try: os.remove(file)
    except Exception as e: print(e)