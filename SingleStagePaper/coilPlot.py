#!/usr/bin/env python3
import os
import numpy as np
from coilpy import Coil
from simsopt import load
from simsopt.mhd import Vmec, VirtualCasing
from simsopt.util import MpiPartition
mpi = MpiPartition()

# # dir = 'optimization_QH'
# dir = 'optimization_CNT'
# # dir = 'optimization_CNT_circular'
# whole_torus = True
# stage1 = False

dir = 'optimization_QH'
whole_torus = False
stage1 = True

nphi = 256
ntheta = 128
ncoils = 3

finite_beta = True

if finite_beta: dir += '_finitebeta'
filename_final = dir+'/input.final'
filename_stage1 = dir+'/input.stage1'
outdir = dir+'/coils/'

def coilpy_plot(curves, filename, height=0.1, width=0.1):
    def wrap(data):
        return np.concatenate([data, [data[0]]])
    xx = [wrap(c.gamma()[:, 0]) for c in curves]
    yy = [wrap(c.gamma()[:, 1]) for c in curves]
    zz = [wrap(c.gamma()[:, 2]) for c in curves]
    II = [1. for _ in curves]
    names = [i for i in range(len(curves))]
    coils = Coil(xx, yy, zz, II, names, names)
    coils.toVTK(filename, line=False, height=height, width=width)

if whole_torus: vmec_final = Vmec(filename_final, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta)
else: vmec_final = Vmec(filename_final, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta, range_surface='half period')
vmec_final.indata.ns_array[:3]    = [  16,     51,   101]
vmec_final.indata.niter_array[:3] = [ 4000,  6000, 10000]
vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-16]
s_final = vmec_final.boundary
vc_src_nphi = int(nphi/2/vmec_final.indata.nfp) if whole_torus else nphi
if finite_beta: vc_final = VirtualCasing.from_vmec(vmec_final, src_nphi=vc_src_nphi, src_ntheta=ntheta)

bs_final = load(outdir + "biot_savart_opt.json")
B_on_surface_final = bs_final.set_points(s_final.gamma().reshape((-1, 3))).AbsB()
norm_final = np.linalg.norm(s_final.normal().reshape((-1, 3)), axis=1)
meanb_final = np.mean(B_on_surface_final * norm_final)/np.mean(norm_final)
absb_final = bs_final.AbsB().reshape(s_final.gamma().shape[:2] + (1,))
Bbs = bs_final.B().reshape((nphi, ntheta, 3))
if finite_beta:
    if whole_torus: BdotN_surf = np.sum(Bbs * s_final.unitnormal(), axis=2) - vc_final.B_external_normal_extended
    else: BdotN_surf = np.sum(Bbs * s_final.unitnormal(), axis=2) - vc_final.B_external_normal
else:
    BdotN_surf = np.sum(Bbs * s_final.unitnormal(), axis=2)
pointData_final = {"B·n/|B|": BdotN_surf[:, :, None]/absb_final,
             "|B|": bs_final.AbsB().reshape(s_final.gamma().shape[:2] + (1,))/meanb_final}
if whole_torus: coilpy_plot([c.curve for c in bs_final.coils], outdir + "coils_optPlot.vtu", height=0.05, width=0.05)
else: coilpy_plot([c.curve for c in bs_final.coils[0:ncoils]], outdir + "coils_optPlot.vtu", height=0.05, width=0.05)
s_final.to_vtk(outdir + "surf_optPlot", extra_data=pointData_final)

if stage1:
    if whole_torus: vmec_stage1 = Vmec(filename_stage1, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta)
    else: vmec_stage1 = Vmec(filename_stage1, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta, range_surface='half period')
    vmec_stage1.indata.ns_array[:3]    = [  16,     51,   101]
    vmec_stage1.indata.niter_array[:3] = [ 4000,  6000, 10000]
    vmec_stage1.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-16]
    s_stage1 = vmec_stage1.boundary
    if finite_beta: vc_stage1 = VirtualCasing.from_vmec(vmec_stage1, src_nphi=vc_src_nphi, src_ntheta=ntheta)

    bs_stage1 = load(outdir + "biot_savart_inner_loop_max_mode_1.json")
    B_on_surface_stage1 = bs_stage1.set_points(s_stage1.gamma().reshape((-1, 3))).AbsB()
    norm_stage1 = np.linalg.norm(s_stage1.normal().reshape((-1, 3)), axis=1)
    meanb_stage1 = np.mean(B_on_surface_stage1 * norm_stage1)/np.mean(norm_stage1)
    absb_stage1 = bs_stage1.AbsB().reshape(s_stage1.gamma().shape[:2] + (1,))
    Bbs = bs_stage1.B().reshape((nphi, ntheta, 3))
    if finite_beta:
        if whole_torus: BdotN_surf = np.sum(Bbs * s_stage1.unitnormal(), axis=2) - vc_stage1.B_external_normal_extended
        else: BdotN_surf = np.sum(Bbs * s_stage1.unitnormal(), axis=2) - vc_stage1.B_external_normal
    else:
        BdotN_surf = np.sum(Bbs * s_stage1.unitnormal(), axis=2)
    pointData_stage1 = {"B·n/|B|": BdotN_surf[:, :, None]/absb_stage1,
                "|B|": bs_stage1.AbsB().reshape(s_stage1.gamma().shape[:2] + (1,))/meanb_stage1}
    if whole_torus: coilpy_plot([c.curve for c in bs_stage1.coils], outdir + "coils_stage1Plot.vtu", height=0.05, width=0.05)
    else: coilpy_plot([c.curve for c in bs_stage1.coils[0:ncoils]], outdir + "coils_stage1Plot.vtu", height=0.05, width=0.05)
    s_stage1.to_vtk(outdir + "surf_stage1Plot", extra_data=pointData_stage1)

files_to_remove = ['input.final_000_000000','input.stage1_000_000000','parvmecinfo.txt','threed1.final','threed1.stage1',
                   'vcasing_final_000_000000.nc','vcasing_stage1_000_000000.nc','wout_final_000_000000.nc','wout_stage1_000_000000.nc']
for file in files_to_remove:
    try: os.remove(file)
    except Exception as e: print(e)

print(f"Created coils_optPlot.vtu, surf_optPlot.vts, coils_stage1Plot.vtu and surf_stage1Plot.vts in directory {outdir}")