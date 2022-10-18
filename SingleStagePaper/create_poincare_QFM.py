#!/usr/bin/env python
import os
import sys
import time
import shutil
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
import booz_xform as bx
from pathlib import Path
parent_path = str(Path(__file__).parent.resolve())
from math import ceil, sqrt
import matplotlib.pyplot as plt
from simsopt import load
from simsopt.mhd import Vmec, Boozer
from simsopt.geo import QfmSurface, SurfaceRZFourier
from simsopt.geo import QfmResidual, Volume, curves_to_vtk
from simsopt.field import particles_to_vtk, compute_fieldlines
import logging
logging.basicConfig()
logger = logging.getLogger('SingleStage')
logger.setLevel(1)
def pprint(*args, **kwargs):
    if comm.rank == 0: print(*args, **kwargs)

main_directory = 'optimization_CNT'
volume_scale = 1.0
nfieldlines = 4
tmax_fl = 1500
tol_qfm = 1e-14
tol_poincare = 1e-13
nphi_QFM = 25
ntheta_QFM = 35
mpol = 6
ntor = 6
maxiter_qfm = 700
constraint_weight=1e-0
ntheta_VMEC = 300
create_QFM = False
create_Poincare = True
nfp=2
bs_file = 'biot_savart_opt.json'

boozxform_nsurfaces = 10
helical_detail = False
this_path = os.path.join(parent_path, main_directory)
OUT_DIR = os.path.join(this_path, "output")
os.chdir(this_path)
bs = load(os.path.join(this_path, f"coils/{bs_file}"))

vmec_ran_QFM = False
if create_QFM:
    vmec = Vmec(os.path.join(this_path,f'wout_final.nc'))
    s = SurfaceRZFourier.from_wout(os.path.join(this_path,f'wout_final.nc'), nphi=nphi_QFM, ntheta=ntheta_QFM, range="half period")
    s.change_resolution(mpol, ntor)
    s_original_VMEC = SurfaceRZFourier.from_wout(os.path.join(this_path,f'wout_final.nc'), nphi=nphi_QFM, ntheta=ntheta_QFM, range="half period")
    nfp = vmec.wout.nfp
    s.to_vtk(os.path.join(OUT_DIR, 'QFM_original_VMEC'))
    pprint('Obtaining QFM surface')
    bs.set_points(s.gamma().reshape((-1, 3)))
    curves = [coil.curve for coil in bs.coils]
    curves_to_vtk(curves, os.path.join(OUT_DIR, "curves_QFM_test"))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi_QFM, ntheta_QFM, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(os.path.join(OUT_DIR, "surf_QFM_test"), extra_data=pointData)
    # Optimize at fixed volume
    qfm = QfmResidual(s, bs)
    pprint(f"Initial qfm.J()={qfm.J()}")
    vol = Volume(s)
    vol_target = Volume(s).J()*volume_scale
    qfm_surface = QfmSurface(bs, s, vol, vol_target)
    t1=time.time()
    pprint(f"Initial ||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=tol_qfm, maxiter=maxiter_qfm, constraint_weight=constraint_weight)
    pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=tol_qfm, maxiter=maxiter_qfm)
    pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    pprint(f"Found QFM surface in {time.time()-t1}s.")

    s.to_vtk(os.path.join(OUT_DIR, 'QFM_found'))
    s_gamma = s.gamma()
    s_R = np.sqrt(s_gamma[:, :, 0]**2 + s_gamma[:, :, 1]**2)
    s_Z = s_gamma[:, :, 2]
    s_gamma_original = s_original_VMEC.gamma()
    s_R_original = np.sqrt(s_gamma_original[:, :, 0]**2 + s_gamma_original[:, :, 1]**2)
    s_Z_original = s_gamma_original[:, :, 2]

    # Plot QFM surface
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    plt.plot(s_R[0,:],s_Z[0,:], label = 'QFM')
    plt.plot(s_R_original[0,:],s_Z_original[0,:], label = 'VMEC')
    plt.xlabel('R')
    plt.ylabel('Z')
    ax.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'QFM_surface.pdf'), bbox_inches = 'tight', pad_inches = 0)

    # Create QFM VMEC equilibrium
    os.chdir(OUT_DIR)
    vmec_QFM = Vmec(os.path.join(this_path,f'input.final'))
    vmec_QFM.indata.mpol = mpol
    vmec_QFM.indata.ntor = ntor
    vmec_QFM.boundary = s
    vmec_QFM.indata.ns_array[:2]    = [   16,     51]
    vmec_QFM.indata.niter_array[:2] = [ 5000,  10000]
    vmec_QFM.indata.ftol_array[:2]  = [1e-14,  1e-14]
    vmec_QFM.indata.am[0:10] = [0]*10
    vmec_QFM.write_input(os.path.join(this_path,f'input.qfm'))
    vmec_QFM = Vmec(os.path.join(this_path,f'input.qfm'))
    try:
        vmec_QFM.run()
        vmec_ran_QFM = True
    except Exception as e:
        pprint('VMEC QFM did not converge')
        pprint(e)
    try:
        shutil.move(os.path.join(OUT_DIR, f"wout_qfm_000_000000.nc"), os.path.join(this_path, f"wout_qfm.nc"))
        os.remove(os.path.join(OUT_DIR, f'input.qfm_000_000000'))
    except Exception as e:
        print(e)

if vmec_ran_QFM or os.path.isfile(os.path.join(this_path, f"wout_QFM.nc")):
    vmec_QFM = Vmec(os.path.join(this_path,f'wout_QFM.nc'))
    nfp = vmec_QFM.wout.nfp
    sys.path.insert(1, os.path.join(parent_path, '../single_stage/plotting'))
    if vmec_ran_QFM or not os.path.isfile(os.path.join(OUT_DIR, "QFM_VMECparams.pdf")):
        import vmecPlot2
        vmecPlot2.main(file=os.path.join(this_path, f"wout_QFM.nc"), name='QFM', figures_folder=OUT_DIR)
    nzeta=4
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    theta = np.linspace(0,2*np.pi,num=ntheta_VMEC)
    iradii = np.linspace(0,vmec_QFM.wout.ns-1,num=nfieldlines).round()
    iradii = [int(i) for i in iradii]
    R = np.zeros((nzeta,nfieldlines,ntheta_VMEC))
    Z = np.zeros((nzeta,nfieldlines,ntheta_VMEC))
    Raxis = np.zeros(nzeta)
    Zaxis = np.zeros(nzeta)
    phis = zeta

    pprint("Obtain VMEC QFM surfaces")
    for itheta in range(ntheta_VMEC):
        for izeta in range(nzeta):
            for iradius in range(nfieldlines):
                for imode, xnn in enumerate(vmec_QFM.wout.xn):
                    angle = vmec_QFM.wout.xm[imode]*theta[itheta] - xnn*zeta[izeta]
                    R[izeta,iradius,itheta] += vmec_QFM.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
                    Z[izeta,iradius,itheta] += vmec_QFM.wout.zmns[imode, iradii[iradius]]*np.sin(angle)
    for izeta in range(nzeta):
        for n in range(vmec_QFM.wout.ntor+1):
            angle = -n*nfp*zeta[izeta]
            Raxis[izeta] += vmec_QFM.wout.raxis_cc[n]*np.cos(angle)
            Zaxis[izeta] += vmec_QFM.wout.zaxis_cs[n]*np.sin(angle)

    if vmec_ran_QFM or not os.path.isfile(os.path.join(OUT_DIR,"boozmn_QFM.nc")):
        pprint('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec_QFM, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn_QFM.nc"))
        pprint("Plot BOOZ_XFORM")
        fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()

if create_Poincare:
    def trace_fieldlines(bfield, R0, Z0):
        t1 = time.time()
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=tol_poincare, comm=comm,
            phis=phis, stopping_criteria=[])
        t2 = time.time()
        pprint(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        # if comm is None or comm.rank == 0:
        #     particles_to_vtk(fieldlines_tys, os.path.join(OUT_DIR,f'fieldlines_optimized_coils'))
            # plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(OUT_DIR, f'poincare_fieldline_optimized_coils.png'), dpi=150)
        return fieldlines_tys, fieldlines_phi_hits, phis

    if vmec_ran_QFM or os.path.isfile(os.path.join(this_path, f"wout_QFM.nc")):
        R0 = R[0,:,0]
        Z0 = Z[0,:,0]
    else:
        pprint('R0 and Z0 not found.')
        exit()
    pprint('Beginning field line tracing')
    fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs, R0, Z0)
    pprint('Creating Poincare plot R, Z')
    r = []
    z = []
    for izeta in range(len(phis)):
        r_2D = []
        z_2D = []
        for iradius in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[iradius][-1, 1] < 0
            data_this_phi = fieldlines_phi_hits[iradius][np.where(fieldlines_phi_hits[iradius][:, 1] == izeta)[0], :]
            if data_this_phi.size == 0:
                pprint(f'No Poincare data for iradius={iradius} and izeta={izeta}')
                continue
            r_2D.append(np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2))
            z_2D.append(data_this_phi[:, 4])
        r.append(r_2D)
        z.append(z_2D)
    r = np.array(r, dtype=object)
    z = np.array(z, dtype=object)
    pprint('Plotting Poincare plot')
    nrowcol = ceil(sqrt(len(phis)))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(12, 8))
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
        axs[row, col].set_xlabel("$R$")
        axs[row, col].set_ylabel("$Z$")
        axs[row, col].set_aspect('equal')
        axs[row, col].tick_params(direction="in")
        for j in range(nfieldlines):
            if j== 0 and i == 0:
                legend1 = 'Poincare plot'
                legend2 = 'VMEC QFM'
            else:
                legend1 = legend2 = '_nolegend_'
            try: axs[row, col].scatter(r[i][j], z[i][j], marker='o', s=0.7, linewidths=0, c='b', label = legend1)
            except Exception as e: pprint(e, i, j)
            if vmec_ran_QFM or os.path.isfile(os.path.join(this_path, f"wout_QFM.nc")):
                axs[row, col].scatter(R[i,j], Z[i,j], marker='o', s=0.7, linewidths=0, c='r', label = legend2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'poincare_QFM_fieldline_all.pdf'), bbox_inches = 'tight', pad_inches = 0)
