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
this_path = str(Path(__file__).parent.resolve())
from math import ceil, sqrt
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, Boozer
from simsopt.geo import QfmSurface, SurfaceRZFourier
from simsopt.geo import QfmResidual, Volume, curves_to_vtk
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.field import BiotSavart, Coil, Current
from simsopt.field import particles_to_vtk, compute_fieldlines, plot_poincare_data
import logging
logging.basicConfig()
logger = logging.getLogger('CNTqs')
logger.setLevel(1)
from find_magnetic_axis import find_magnetic_axis
def pprint(*args, **kwargs):
    if comm.rank == 0:  # only print on rank 0
        print(*args, **kwargs)

# Input parameters
gamma=0.57555024
alpha=0.23910724
center1 = 0.405
center2 = 0.315
radius1 = 1.08
radius2 = 0.405
current2 = 170e3
nfp = 2
delta_R = 0.05
nfieldlines = 12
initial_QFM_radius=0.15
tmax_fl = 2500
mpol = 7
ntor = 7
tol_qfm = 1e-13
maxiter_qfm = 4000
constraint_weight=1e-3
nphi_QFM = 35
ntheta_QFM = 55
stellsym = True
ntheta_VMEC = 300
boozxform_nsurfaces = 10

create_QFM = True
create_Poincare = True

# Create output directory
OUT_DIR = os.path.join(this_path, "output")
os.makedirs(OUT_DIR, exist_ok=True)

pprint('Creating coils')
current1 = alpha*current2
currents = [Current(current1),Current(current1),Current(current2),Current(-current2)]
curves = [CurveXYZFourier(128, 1) for i in range(4)]
curves[0].set_dofs([0, 0, radius1, 0, radius1, 0, -center1, 0., 0])
curves[1].set_dofs([0, 0, radius1, 0, radius1, 0,  center1, 0., 0])
curves[2].set_dofs([ center2, 0, radius2, 0,-radius2*np.sin(gamma), 0, 0, radius2*np.cos(gamma), 0])
curves[3].set_dofs([-center2, 0, radius2, 0, radius2*np.sin(gamma), 0, 0, radius2*np.cos(gamma), 0])
coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
bs = BiotSavart(coils)
curves_to_vtk(curves, os.path.join(OUT_DIR, 'coils'))

pprint('Finding magnetic axis')
R_axis, _, Z_axis = find_magnetic_axis(bs, 32, 0.27)[0]
pprint(f'  Found magnetic axis at R={R_axis}')

vmec_ran_QFM = False
if create_QFM:
    pprint('Obtaining QFM surface')
    phis_QFM = np.linspace(0, 1/nfp/2, nphi_QFM, endpoint=False)
    thetas_QFM = np.linspace(0, 1, ntheta_QFM, endpoint=False)
    s = SurfaceRZFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis_QFM,
        quadpoints_theta=thetas_QFM)
    ma = CurveRZFourier(nphi_QFM, 1, 1, True)
    ma.rc[:] = [R_axis,0]
    ma.zs[:] = [0]
    ma.x = ma.get_dofs()
    s.fit_to_curve(ma, initial_QFM_radius, flip_theta=True)
    curves_to_vtk([ma], os.path.join(OUT_DIR, 'QFM_ma'))
    s.to_vtk(os.path.join(OUT_DIR, 'QFM_original'))
    # Optimize at fixed volume
    qfm = QfmResidual(s, bs)
    pprint(f"Initial qfm.J()={qfm.J()}")
    vol = Volume(s)
    vol_target = vol.J()
    qfm_surface = QfmSurface(bs, s, vol, vol_target)
    pprint(f"Initial ||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=tol_qfm, maxiter=maxiter_qfm, constraint_weight=constraint_weight)
    pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=tol_qfm, maxiter=maxiter_qfm)
    pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    s.to_vtk(os.path.join(OUT_DIR, 'QFM_found'))
    s_gamma = s.gamma()
    s_R = np.sqrt(s_gamma[:, :, 0]**2 + s_gamma[:, :, 1]**2)
    s_Z = s_gamma[:, :, 2]

    # Plot QFM surface
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    plt.plot(s_R[0,:],s_Z[0,:], label = 'QFM')
    plt.xlabel('R')
    plt.ylabel('Z')
    ax.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'QFM_surface.pdf'), bbox_inches = 'tight', pad_inches = 0)

    # Create QFM VMEC equilibrium
    os.chdir(OUT_DIR)
    vmec_QFM = Vmec(os.path.join(this_path,f'input.CNT_qfm'))
    vmec_QFM.indata.mpol = mpol
    vmec_QFM.indata.ntor = ntor
    vmec_QFM.boundary = s
    vmec_QFM.indata.ns_array[:2]    = [  16,     51]
    vmec_QFM.indata.niter_array[:2] = [ 3000,  5000]
    vmec_QFM.indata.ftol_array[:2]  = [1e-13, 1e-14]
    vmec_QFM.write_input(os.path.join(this_path,f'input.CNT_qfm'))
    try:
        vmec_QFM.run()
        vmec_ran_QFM = True
        sys.path.insert(1, os.path.join(this_path, '../single_stage/plotting'))
        import vmecPlot2
        vmecPlot2.main(file=os.path.join(OUT_DIR, f"wout_CNT_qfm.nc"), name='CNT', figures_folder=OUT_DIR, coils_curves=curves)
    except Exception as e:
        pprint('VMEC QFM did not converge')
        pprint(e)
    try:
        shutil.move(f"wout_CNT_qfm_000_000000.nc", f"wout_CNT_qfm.nc")
        os.remove(f'input.CNT_qfm_000_000000')
    except Exception as e:
        print(e)
    if vmec_ran_QFM:
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

        ## Obtain VMEC QFM surfaces
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

        pprint('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec_QFM, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn_CNT_QFM.nc"))
        pprint("Plot BOOZ_XFORM")
        helical_detail = True
        fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_CNT_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_CNT_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_CNT_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_CNT_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_CNT_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()

if create_Poincare:
    if vmec_ran_QFM:
        R0 = R[0,:,0]
        Z0 = Z[0,:,0]
    else:
        pprint('Finding magnetic axis')
        R_axis, _, Z_axis = find_magnetic_axis(bs, 64, 0.32)[0]
        pprint(f'  Found magnetic axis at R={R_axis}')
        R0 = np.linspace(R_axis, R_axis + delta_R, nfieldlines)
        Z0 = [Z_axis for i in range(nfieldlines)]

    def trace_fieldlines(bfield, label):
        t1 = time.time()
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-11, comm=comm,
            phis=phis, stopping_criteria=[])
        t2 = time.time()
        pprint(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm is None or comm.rank == 0:
            particles_to_vtk(fieldlines_tys, os.path.join(OUT_DIR,f'fieldlines_{label}'))
            # plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(OUT_DIR, f'poincare_fieldline_{label}.png'), dpi=150)
        return fieldlines_tys, fieldlines_phi_hits, phis
        
    pprint('Beginning field line tracing')
    fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs, 'bs')
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
            if vmec_ran_QFM:
                axs[row, col].scatter(R[i,j], Z[i,j], marker='o', s=0.7, linewidths=0, c='r', label = legend2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'CNT_poincare_QFM_fieldline_all.pdf'), bbox_inches = 'tight', pad_inches = 0)