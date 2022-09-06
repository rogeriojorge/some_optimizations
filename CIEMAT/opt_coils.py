#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.mhd import Vmec
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import CurveLength, CurveCurveDistance, CurveSurfaceDistance, \
    MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation
from simsopt.field import BiotSavart, InterpolatedField
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength
from simsopt.mhd import VirtualCasing
from glob import glob
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
import sys
import time
from simsopt import load
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    def pprint(*args, **kwargs):
        if comm.rank == 0:  # only print on rank 0
            print(*args, **kwargs)
except ImportError:
    comm = None
    pprint = print

# Number of unique coil shapes, i.e. the number of coils per half field period:
ncoils = 6
# Major radius for the initial circular coils:
R0 = 2
# Minor radius for the initial circular coils:
R1 = 0.9
# Number of Fourier modes describing each Cartesian component of each coil:
order = 18
# Weight on the curve length penalties in the objective function:
LENGTH_PENALTY = 2e-4
iter_2_length_scaling = 0.5
# Number of iterations to perform:
MAXITER_1 = 250
# File for the desired boundary magnetic surface:
filename = os.path.join(os.path.dirname(__file__), 'input.LNF4p1714_2537b1.5.vmec')
vmec_file = os.path.join(os.path.dirname(__file__), 'wout_LNF4p1714_2537b1.5.vmec.nc')
# Resolution on the plasma boundary surface:
# nphi is the number of grid points in 1/2 a field period.
nphi = 32
ntheta = 32
# Resolution for the virtual casing calculation:
vc_src_nphi = 80
####
MAXITER_2 = 250
CC_THRESHOLD = 0.1
CC_WEIGHT = 1e+1
CS_THRESHOLD = 0.21
CS_WEIGHT = 1e+1
CURVATURE_THRESHOLD = 10
CURVATURE_WEIGHT = 1e-7
MSC_THRESHOLD = 20
MSC_WEIGHT = 1e-5
ARCLENGTH_WEIGHT = 2e-3
####
run_optimization = False
remove_output = False
find_QFM_surface = True
create_Poincare_plot = True
nzeta_Poincare = 4
nradius_Poincare=10
tol_qfm = 1e-14
maxiter_qfm = 6000
#######################################################
# End of input parameters.
#######################################################

# Directory for output
OUT_DIR = os.path.join(os.path.dirname(__file__), "output/")
if comm.rank == 0:
    if remove_output:
        filelist = glob(f"{OUT_DIR}/*")
        for f in filelist:
            os.remove(f)
    os.makedirs(OUT_DIR, exist_ok=True)

# Once the virtual casing calculation has been run once, the results
# can be used for many coil optimizations. Therefore here we check to
# see if the virtual casing output file alreadys exists. If so, load
# the results, otherwise run the virtual casing calculation and save
# the results.
head, tail = os.path.split(vmec_file)
vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
print('virtual casing data file:', vc_filename)
if os.path.isfile(vc_filename):
    print('Loading saved virtual casing result')
    vc = VirtualCasing.load(vc_filename)
else:
    # Virtual casing must not have been run yet.
    print('Running the virtual casing calculation')
    vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)

# Initialize the boundary magnetic surface:
s = SurfaceRZFourier.from_wout(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
s_total = SurfaceRZFourier.from_wout(vmec_file, range="full torus", nphi=nphi, ntheta=ntheta)
total_current = Vmec(vmec_file).external_current() / (2 * s.nfp)

# Create the initial coils
# Check if there are already optimized coils we can use
bs_json_files = [file for file in os.listdir(OUT_DIR) if '.json' in file]
# if Path(os.path.join(coils_results_path,"biot_savart_opt.json")).is_file():
if len(bs_json_files)==0:
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
    base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
else:
    bs_temporary = load(os.path.join(OUT_DIR, bs_json_files[-1]))
    base_curves = [bs_temporary.coils[i]._curve for i in range(ncoils)]
    base_currents = [bs_temporary.coils[i]._current for i in range(ncoils-1)]

total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]

curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s_total.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the objective function:
Jf = SquaredFlux(s, bs, target=vc.B_external_normal)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
J_LENGTH = LENGTH_PENALTY * sum(Jls)
# J_LENGTH = LENGTH_PENALTY * sum(QuadraticPenalty(Jls[i], Jls[i].J()) for i in range(len(base_curves)))
J_CC = CC_WEIGHT * Jccdist
J_CS = CS_WEIGHT * Jcsdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)

JF_1 = Jf + J_LENGTH
JF_2 = Jf + iter_2_length_scaling*J_LENGTH + J_ALS + J_CC + J_CURVATURE + J_MSC# + J_CS

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs, opt_stage=1, info={'Nfeval':0}):
    if opt_stage==1:
        JF_1.x = dofs
        J = JF_1.J()
        grad = JF_1.dJ()
        it_2_length_scaling = 1
        pprint(f"\nIteration {info['Nfeval']} of {MAXITER_1+3}")
    elif opt_stage==2:
        JF_2.x = dofs
        J = JF_2.J()
        grad = JF_2.dJ()
        it_2_length_scaling = iter_2_length_scaling
        pprint(f"\nIteration {info['Nfeval']} of {MAXITER_2+3}")
    else:
        pprint(f"\n{opt_stage} is not a valid optimization stage number. Use opt_stage=1 or 2.")
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    BdotN_mean = np.mean(BdotN)
    BdotN_max = np.max(BdotN)
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨|B·n|⟩={BdotN_mean:.1e}, max(|B·n|)={BdotN_max:.1e}, ║∇J coils║={np.linalg.norm(grad):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f"\n Jf={jf:.1e}, J_length={it_2_length_scaling*J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CS={J_CS.J():.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}"
    cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
    outstr += f"\n Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
    pprint(outstr, flush=True)
    info['Nfeval']+=1
    return J, grad

# pprint("""
# ################################################################################
# ### Perform a Taylor test ######################################################
# ################################################################################
# """)
# f = fun
# dofs = JF_1.x
# np.random.seed(1)
# h = np.random.uniform(size=dofs.shape)
# J0, dJ0 = f(dofs)
# dJh = sum(dJ0 * h)
# for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
#     J1, _ = f(dofs + eps*h)
#     J2, _ = f(dofs - eps*h)
#     pprint("err", (J1-J2)/(2*eps) - dJh)

if run_optimization:
    pprint("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    pprint("Stage 1")
    dofs = JF_1.x
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER_1, 'maxcor': 300, 'ftol': 1e-20, 'gtol': 1e-20}, tol=1e-20, args=(1, {'Nfeval':0}))
    dofs = res.x
    curves_to_vtk(curves, OUT_DIR + "curves_opt_1")
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    pointData = {"B_N": BdotN[:, :, None]}
    s_total.to_vtk(OUT_DIR + "surf_opt_1", extra_data=pointData)
    bs.save(os.path.join(OUT_DIR,'biot_savart_opt_1.json'))

    pprint("\nStage 2")
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER_2, 'maxcor': 300, 'ftol': 1e-20, 'gtol': 1e-20}, tol=1e-20, args=(2, {'Nfeval':0}))
    dofs = res.x
    curves_to_vtk(curves, OUT_DIR + "curves_opt")
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    pointData = {"B_N": BdotN[:, :, None]}
    s_total.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
    bs.save(os.path.join(OUT_DIR,'biot_savart_opt.json'))

pprint("""
################################################################################
### Creating figures diagnostics ###############################################
################################################################################
""")
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../single_stage'))
if find_QFM_surface:
    pprint("Obtain QFM surface")
    import field_from_coils
    R_qfm, Z_qfm, Raxis_qfm, Zaxis_qfm = field_from_coils.main(folder=os.path.dirname(__file__), OUT_DIR=OUT_DIR, coils_folder=OUT_DIR, vmec_folder=os.path.dirname(__file__), mpi=None, nzeta=nzeta_Poincare, nradius=nradius_Poincare, tol_qfm = tol_qfm, maxiter_qfm = maxiter_qfm, vmec_input_start=os.path.join(os.path.dirname(__file__), 'input.LNF4p1714_2537b1.5.vmec'), name_manual='CIEMAT')
if create_Poincare_plot:
    pprint("Creating Poincare plot")
    nfp = Vmec(filename).indata.nfp
    R0_tracing = R_qfm[0,:,0]
    Z0_tracing = Z_qfm[0,:,0]
    nzeta = nzeta_Poincare
    degree = 4
    tmax_fl = 1000
    tol_tracing = 1e-12
    nfieldlines = len(R0_tracing)
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    sc_fieldline = SurfaceClassifier(s_total, h=0.03, p=2)
    def trace_fieldlines(bfield, label):
        t1 = time.time()
        pprint(f'Initial radii for field line tracer: {R0_tracing}')
        pprint(f'Starting particle tracer')
        phis = zeta
        if comm.rank == 0:
            fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
                bfield, R0_tracing, Z0_tracing, tmax=tmax_fl, tol=tol_tracing,#, comm=mpi.comm_world,
                phis=phis, stopping_criteria=[
                    # LevelsetStoppingCriterion(sc_fieldline.dist)
                    ])
        t2 = time.time()
        pprint(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm.rank == 0:
            # particles_to_vtk(fieldlines_tys, os.path.join(OUT_DIR, f'{name}_fieldlines_{label}'))
            plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(OUT_DIR, f'CIEMAT_poincare_fieldline_{label}.png'), dpi=150)
        return fieldlines_tys, fieldlines_phi_hits, phis
    n = 20
    rs = np.linalg.norm(s_total.gamma()[:, :, 0:2], axis=2)
    zs = s_total.gamma()[:, :, 2]
    rrange = (0.95*np.min(rs), 1.05*np.max(rs), n)
    phirange = (0, 2*np.pi/nfp, n*2)
    zrange = (0, 1.05*np.max(zs), n//2)
    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        pprint("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip
    pprint('Initializing InterpolatedField')
    bsh = InterpolatedField(bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip)
    bsh.set_points(s_total.gamma().reshape((-1, 3)))
    pprint('Done initializing InterpolatedField.')
    pprint('Beginning field line tracing')
    fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bsh, 'bsh')
    pprint('Finished field line tracing')
if create_Poincare_plot and find_QFM_surface:
    pprint('Plotting magnetic surfaces and Poincare plots')
    from opt_funcs import plot_qfm_poincare
    plot_qfm_poincare(phis=phis, fieldlines_phi_hits=fieldlines_phi_hits, R=R_qfm, Z=Z_qfm, OUT_DIR=OUT_DIR, name='CIEMAT')
