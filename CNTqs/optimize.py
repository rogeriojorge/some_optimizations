#!/usr/bin/env python
import os
import sys
import time
start = time.time()
import shutil
import numpy as np
import pandas as pd
from mpi4py import MPI
comm = MPI.COMM_WORLD
import booz_xform as bx
from pathlib import Path
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
from math import isnan
import matplotlib.pyplot as plt
from simsopt import load
from simsopt.mhd import Vmec, Boozer
from simsopt.geo import curves_to_vtk
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt._core.optimizable import make_optimizable
from simsopt.field import BiotSavart, Coil, Current
import logging
logging.basicConfig()
logger = logging.getLogger('CNTqs')
logger.setLevel(1)
from simsopt.util import MpiPartition
def pprint(*args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                        LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation,
                        SurfaceRZFourier)
from simsopt.solve import least_squares_mpi_solve
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt._core.finite_difference import MPIFiniteDifference, FiniteDifference, finite_difference_steps
from simsopt._core.derivative import Derivative
from scipy.optimize import minimize
from simsopt.mhd import VirtualCasing

mpi = MpiPartition()
max_modes = [1, 1, 1, 2, 2]#np.concatenate(([1] * 5, [2]*4, [3]*2))
MAXITER_single_stage = 30
MAXITER_stage_2 = 500
coils_objective_weight = 1e+2
nmodes_coils = 6
circularTopBottom = False
aspect_ratio_target = 3.0
iota_target = -0.22
single_stage=True
magnetic_well=False
vacuum_well_target=0.05
vacuum_well_weight=1
finite_beta=True
# Stage 1 optimization only
MAXITER_stage_1 = 20
stage_1=False

CURVATURE_THRESHOLD = 4.0
MSC_THRESHOLD = 7
LENGTH_THRESHOLD = [7.1,7.1,3.5,3.5]

directory = f'optimization_{nmodes_coils}modes'
if circularTopBottom: directory += '_circular'
if stage_1: directory += '_stage1'
if magnetic_well: directory +='_well'
if finite_beta: directory +='_finitebeta'
use_previous_results_if_available = True
nsurfaces_stage2 = 1

quasisymmetry_helicity_m = 1
aspect_ratio_weight = 1
quasisymmetry_helicity_n = 0
iota_weight = 50
initial_irad = 3

nphi_VMEC=110
ntheta_VMEC=40
vmec_verbose=False
if finite_beta:
    vmec_input_filename='input.CNT_qfm'
else:
    vmec_input_filename='input.CNT_finiteBeta'
quasisymmetry_target_surfaces = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

gamma=0.57555024
alpha=0.23910724
center1 = 0.405
center2 = 0.315
radius1 = 1.08
radius2 = 0.405
current2 = 1.7
nfp = 2

boozxform_nsurfaces = 10
helical_detail = False
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 1e-5
JACOBIAN_THRESHOLD = 550
CC_THRESHOLD = 0.15 # Threshold for the coil-to-coil distance penalty in the objective function
CS_THRESHOLD = 0.1 # Threshold for the curvature penalty in the objective function
LENGTHBOUND = 20 # Threshold for the sum of coil lengths
LENGTH_CON_WEIGHT = 0.1 # Weight on the quadratic penalty for the curve length
LENGTH_WEIGHT = 1e-8 # Weight on the curve lengths in the objective function
CC_WEIGHT = 1e+1 # Weight for the coil-to-coil distance penalty in the objective function
CS_THRESHOLD = 0.3 # Threshold for the coil-to-surface distance penalty in the objective function
CS_WEIGHT = 3e-1 # Weight for the coil-to-surface distance penalty in the objective function
CURVATURE_WEIGHT = 1e-1 # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 1e-1 # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 1e-9 # Weight for the arclength variation penalty in the objective function
vc_src_nphi = 45 # Resolution for the virtual casing calculation

debug_coils_outputtxt = True
coil_gradients_analytical = True
debug_output_file = 'output.txt'

# Create output directories
if not use_previous_results_if_available:
    if comm.rank == 0:
        if os.path.isdir(directory):
            shutil.copytree(directory, f"{directory}_backup", dirs_exist_ok=True)
            shutil.rmtree(directory)
this_path = os.path.join(parent_path, directory)
if comm.rank == 0:
    os.makedirs(this_path, exist_ok=True)
    os.chdir(this_path)
OUT_DIR = os.path.join(this_path, "output")
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")

if comm.rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)

# Stage 1
if use_previous_results_if_available and os.path.isfile(os.path.join(this_path, "input.CNT_final")):
    pprint(f' Using vmec input file {os.path.join(this_path,"input.CNT_final")}')
    vmec = Vmec(os.path.join(this_path,'input.CNT_final'), mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC)
else:
    pprint(f' Using vmec input file {os.path.join(parent_path,vmec_input_filename)}')
    vmec = Vmec(os.path.join(parent_path,vmec_input_filename), mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC)

surf = vmec.boundary

# Finite Beta Virtual Casing Principle
if finite_beta:
    vmec_file = os.path.join(parent_path, 'wout_CNT_finiteBeta.nc')
    head, tail = os.path.split(vmec_file)
    vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
    pprint(' Running the initial virtual casing calculation')
    vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
    total_current = Vmec(vmec_file).external_current()

#Stage 2
if use_previous_results_if_available and os.path.isfile(os.path.join(coils_results_path, "biot_savart_opt.json")):
    bs = load(os.path.join(coils_results_path, "biot_savart_opt.json"))
    curves = [coil._curve for coil in bs.coils]
    currents = [Current(coil._current.x[0])*1e5 for coil in bs.coils]
else:
    if finite_beta:
        currents = [Current(total_current/4*1e-5)*1e5, Current(total_current/4*1e-5)*1e5, Current(total_current/4*1e-5)*1e5, Current(-total_current/4*1e-5)*1e5]
        # total_current = Current(total_current)
        # total_current.fix_all()
        # currents += [total_current - sum(currents)]
        logger.info(f'Coil currents = {[current.x[0] for current in currents]}')
    else:
        current1 = alpha*current2
        currents = [Current(current1)*1e5,Current(current1)*1e5,Current(current2)*1e5,Current(-current2)*1e5]
    curves = [CurveXYZFourier(128, 1) for i in range(4)]
    # Only 1 Fourier mode in coils
    # # curves[0].local_dof_names -> ['xc(0)', 'xs(1)', 'xc(1)', 'yc(0)', 'ys(1)', 'yc(1)', 'zc(0)', 'zs(1)', 'zc(1)']
    # # curves[0].fix_all();curves[0].unfix('xc(1)');curves[0].unfix('ys(1)');curves[0].unfix('zc(0)')
    # curves[0].set_dofs([0, 0, radius1, 0, radius1, 0, -center1, 0., 0])
    # curves[1].set_dofs([0, 0, radius1, 0, radius1, 0,  center1, 0., 0])
    # curves[2].set_dofs([ center2, 0, radius2, 0,-radius2*np.sin(gamma), 0, 0, radius2*np.cos(gamma), 0])
    # curves[3].set_dofs([-center2, 0, radius2, 0, radius2*np.sin(gamma), 0, 0, radius2*np.cos(gamma), 0])
    curves = [CurveXYZFourier(128, nmodes_coils) for i in range(4)]
    curves[0].set_dofs(np.concatenate(([       0, 0, radius1],np.zeros(2*(nmodes_coils-1)),[0,                radius1, 0],np.zeros(2*(nmodes_coils-1)),[-center1,                     0, 0],np.zeros(2*(nmodes_coils-1)))))
    curves[1].set_dofs(np.concatenate(([       0, 0, radius1],np.zeros(2*(nmodes_coils-1)),[0,                radius1, 0],np.zeros(2*(nmodes_coils-1)),[ center1,                     0, 0],np.zeros(2*(nmodes_coils-1)))))
    curves[2].set_dofs(np.concatenate(([ center2, 0, radius2],np.zeros(2*(nmodes_coils-1)),[0, -radius2*np.sin(gamma), 0],np.zeros(2*(nmodes_coils-1)),[       0, radius2*np.cos(gamma), 0],np.zeros(2*(nmodes_coils-1)))))
    curves[3].set_dofs(np.concatenate(([-center2, 0, radius2],np.zeros(2*(nmodes_coils-1)),[0,  radius2*np.sin(gamma), 0],np.zeros(2*(nmodes_coils-1)),[       0, radius2*np.cos(gamma), 0],np.zeros(2*(nmodes_coils-1)))))

# Fix the currents, otherwise they will just reduce to zero to minimize the Squared Flux
# currents[0].fix_all()
# currents[1].fix_all()
# currents[2].fix_all()
# currents[3].fix_all()

if circularTopBottom:
    curves[0].fix_all()
    curves[1].fix_all()

# Save initial surface and coil data
coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
if finite_beta:
    BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
else:
    BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)

# Define the individual terms in the objective function
if finite_beta:
    Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
else:
    Jf = SquaredFlux(surf, bs, local=True)
Jls = [CurveLength(c) for c in curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
# Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(curves)]
Jmscs = [MeanSquaredCurvature(c) for c in curves]
Jals = [ArclengthVariation(c) for c in curves]

J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
# J_CS = CS_WEIGHT * Jcsdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for i, J in enumerate(Jmscs))
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD[i]) for i in range(len(curves))])

JF = Jf + J_CC + J_LENGTH + J_LENGTH_PENALTY + J_CURVATURE# + J_MSC + J_ALS + J_CS

# Initial stage 2 optimization
def fun_coils(dofss, info, oustr_dict=[]):
    info['Nfeval'] += 1
    if info['Nfeval'] == 2: pprint('Iteration #: ', end='', flush=True)
    pprint(info['Nfeval'], ' ', end='', flush=True)
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)))
        outstr = f"\nfun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        dict1 = {}
        dict1.update({
            'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf), 'J_length':float(J_LENGTH.J()),
            'Lengths':float(sum(j.J() for j in Jls)), 'J_CC':float(J_CC.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),
            # 'J_CS':float(J_CS.J()),
            'J_CURVATURE':float(J_CURVATURE.J()), 'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()),
            'curvatures':float(np.sum([np.max(c.kappa()) for c in curves])), 'msc':float(np.sum([j.J() for j in Jmscs])),
            'B.n':float(BdotN), 'gradJcoils':float(np.linalg.norm(JF.dJ())), 'C-C-Sep':float(Jccdist.shortest_distance()),
            # 'C-S-Sep':float(Jcsdist.shortest_distance())
        })
        if debug_coils_outputtxt:
            outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
            outstr += f" J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}"
            cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
            kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in curves)
            msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
            outstr += f" Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
            outstr += f", coils dofs="+", ".join([f"{pr}" for pr in dofss[0:6]])
        with open(debug_output_file, "a") as myfile:
            myfile.write(outstr)
        oustr_dict.append(dict1)
    return J, grad

def plot_df_stage2(df, max_mode):
    df = pd.DataFrame(oustr_dict)
    df.to_csv(f'output_stage2_max_mode_{max_mode}.csv', index_label='index')
    ax=df.plot(kind='line', logy=True, y=['J','Jf','J_length','J_CC','J_CURVATURE','J_MSC','J_ALS','J_LENGTH_PENALTY','C-C-Sep'], linewidth=0.8)#,'J_CENTER_CURVES'], linewidth=0.8)
    ax.set_ylim(bottom=1e-9, top=None)
    ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Objective function')
    plt.axvline(x=info_coils['Nfeval'], linestyle='dashed', color='k', label='simple-loop', linewidth=0.8)
    plt.legend(loc=3, prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'optimization_stage2_max_mode_{max_mode}.pdf'), bbox_inches = 'tight', pad_inches = 0)
    plt.close()

#############################################################
## Define main optimization function with gradients
#############################################################
pprint(f'  Performing Single Stage optimization with {MAXITER_single_stage} iterations')
def fun_J(dofs):
    if np.sum(JF.x!=dofs[:-number_vmec_dofs])>0:
        JF.x = dofs[:-number_vmec_dofs]
    if np.sum(prob.x!=dofs[-number_vmec_dofs:])>0:
        prob.x = dofs[-number_vmec_dofs:]
        if finite_beta:
            try:
                logger.info(f"Running virtual casing")
                vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
                Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
                JF.opts[0].opts[0].opts[0] = Jf
            except Exception as e:
                logger.info(f"Exception caught during VirtualCasing calculation. Returning J={JACOBIAN_THRESHOLD}")
                J = JACOBIAN_THRESHOLD
                Jf = JF.opts[0].opts[0].opts[0]
    bs.set_points(surf.gamma().reshape((-1, 3)))

    J_stage_1 = prob.objective()
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    return J

def fun(dofs, prob_jacobian=None, info={'Nfeval':0}, max_mode=1, oustr_dict=[]):
    logger.info('Entering fun')
    info['Nfeval'] += 1
    os.chdir(vmec_results_path)

    # This should be computed only on the main processor
    J = fun_J(dofs)

    if J > JACOBIAN_THRESHOLD or isnan(J):
        logger.info(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        
    logger.info('Writing result')
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    if finite_beta:
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target
    else:
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
    BdotN = np.mean(np.abs(BdotN_surf))
    outstr = f"\n\nfun#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    dict1 = {}
    dict1.update({
        'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf), 'J_length':float(J_LENGTH.J()),
        'Lengths':float(sum(j.J() for j in Jls)), 'J_CC':float(J_CC.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),
        # 'J_CS':float(J_CS.J()),
        'J_CURVATURE':float(J_CURVATURE.J()), 'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()),
        'curvatures':float(np.sum([np.max(c.kappa()) for c in curves])), 'msc':float(np.sum([j.J() for j in Jmscs])),
        'B.n':float(BdotN), 'gradJcoils':float(np.linalg.norm(JF.dJ())), 'C-C-Sep':float(Jccdist.shortest_distance()),
        # 'C-S-Sep':float(Jcsdist.shortest_distance())
    })

    # Computing gradients
    if J<JACOBIAN_THRESHOLD:
        logger.info(f'Objective function {J} is smaller than the threshold {JACOBIAN_THRESHOLD}')
        logger.info(f'Now calculating the gradient')
        if finite_beta:
            # Finite difference for the coil gradients
            # prob_dJ = prob_jacobian.jac(prob.x)
            # surface = surf
            # bs.set_points(surface.gamma().reshape((-1, 3)))
            # grad_coils = np.zeros(len(dofs),)
            # steps = finite_difference_steps(dofs, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step)
            # f0 = coils_objective_weight * JF.J()
            # for j in range(len(dofs)):
            #     x = np.copy(dofs)
            #     x[j] = dofs[j] + steps[j]
            #     if np.sum(JF.x-x[:-number_vmec_dofs])!=0:
            #         JF.x = x[:-number_vmec_dofs]
            #     if np.sum(prob.x-x[-number_vmec_dofs:])!=0:
            #         prob.x = x[-number_vmec_dofs:]
            #         vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
            #         Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
            #         JF.opts[0].opts[0].opts[0] = Jf
            #     bs.set_points(surf.gamma().reshape((-1, 3)))
            #     fplus = coils_objective_weight * JF.J()
            #     ## This is only doing forward differences, should be centered
            #     ## because the derivative with respect to coils is innacurate
            #     # grad_coils[j] = (fplus - f0) / steps[j]
            #     ## This is doing centered differences
            #     x[j] = dofs[j] - steps[j]
            #     if np.sum(JF.x-x[:-number_vmec_dofs])!=0:
            #         JF.x = x[:-number_vmec_dofs]
            #     if np.sum(prob.x-x[-number_vmec_dofs:])!=0:
            #         prob.x = x[-number_vmec_dofs:]
            #         vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
            #         Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
            #         JF.opts[0].opts[0].opts[0] = Jf
            #     bs.set_points(surf.gamma().reshape((-1, 3)))
            #     fminus = coils_objective_weight * JF.J()
            #     grad_coils[j] = (fplus - fminus) / (2 * steps[j])
            # grad_with_respect_to_coils = grad_coils[:-number_vmec_dofs]
            # grad_with_respect_to_surface = np.ravel(prob_dJ) + grad_coils[-number_vmec_dofs:]
            # grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
            grad = np.empty(len(dofs))
            opt = make_optimizable(fun_J, dofs, dof_indicators=["dof"])
            with MPIFiniteDifference(opt.J, mpi, diff_method="centered", abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as fd:
                if mpi.proc0_world:
                    grad = np.array(fd.jac()[0])
            mpi.comm_world.Bcast(grad, root=0)
        else:
            prob_dJ = prob_jacobian.jac(prob.x)
            surface = surf
            bs.set_points(surface.gamma().reshape((-1, 3)))
            coils_dJ = JF.dJ()
            ## Mixed term - derivative of squared flux with respect to the surface shape
            n = surface.normal()
            absn = np.linalg.norm(n, axis=2)
            B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
            dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
            Bcoil = bs.B().reshape(n.shape)
            unitn = n * (1./absn)[:, :, None]
            Bcoil_n = np.sum(Bcoil*unitn, axis=2)
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            # if finite_beta:
            #     B_n = (Bcoil_n - vc.B_external_normal)
            #     B_diff = Bcoil - B_external
            #     B_N = np.sum(B_diff * n, axis=2)
            # else:
            B_n = Bcoil_n
            B_diff = Bcoil
            B_N = np.sum(Bcoil * n, axis=2)
            assert Jf.local
            dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
            dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
            deriv = surface.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surface.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
            mixed_dJ = Derivative({surface: deriv})(surface)

            ## Put both gradients together
            grad_with_respect_to_coils = coils_objective_weight * coils_dJ
            grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
            grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    else:
        logger.info(f'Objective function {J} is greater than the threshold {JACOBIAN_THRESHOLD}')
        grad = [0] * len(dofs)
        grad_with_respect_to_coils = np.zeros(len(dofs)--number_vmec_dofs,)

    if debug_coils_outputtxt:
        if not finite_beta:
            if nsurfaces_stage2==1: outstr += f", ║∇J coils║={np.linalg.norm(grad_with_respect_to_coils/coils_objective_weight):.1e}"
            else: outstr += f", ║∇J coils║={np.linalg.norm(grad_with_respect_to_coils/coils_objective_weight):.1e}"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"#,, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        # outstr += f"\nJ_CS={J_CS.J():.1e}"
        outstr += f"\n J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}"
        # outstr += f", J_CENTER_CURVES={J_CENTER_CURVES.J():.1e}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f"\n Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}"
        outstr += f", curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
    try:
        outstr += f"\n surface dofs="+", ".join([f"{pr}" for pr in dofs[-number_vmec_dofs:]])
        coilsdofs = dofs[:-number_vmec_dofs]
        outstr += f"\n coils dofs="+", ".join([f"{pr}" for pr in coilsdofs[0:6]])
        if J<JACOBIAN_THRESHOLD:
            outstr += f"\n Quasisymmetry objective={qs.total()}"
            outstr += f"\n aspect={vmec.aspect()}"
            outstr += f"\n mean iota={vmec.mean_iota()}"
            outstr += f"\n magnetic well={vmec.vacuum_well()}"
            dict1.update({'Jquasisymmetry':float(qs.total()), 'Jwell': float((vmec.vacuum_well()-vacuum_well_target)**2), 'Jiota':float((vmec.mean_iota()-iota_target)**2), 'Jaspect':float((vmec.aspect()-aspect_ratio_target)**2)})
        else:
            dict1.update({'Jquasisymmetry':0, 'Jiota':0,'Jaspect':0, 'Jwell': 0})
    except Exception as e:
        pprint(e)

    os.chdir(this_path)
    with open(debug_output_file, "a") as myfile:
        myfile.write(outstr)
        # if J<JACOBIAN_THRESHOLD:
        #     myfile.write(f"\n prob_dJ="+", ".join([f"{p}" for p in np.ravel(prob_dJ)])+"\n coils_dJ[3:10]="+", ".join([f"{p}" for p in coils_dJ[3:10]])+"\n mixed_dJ="+", ".join([f"{p}" for p in mixed_dJ]))
    oustr_dict.append(dict1)
    if np.mod(info['Nfeval'],5)==0:
        pointData = {"B_N": np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path,f"surf_intermediate_max_mode_{max_mode}_{info['Nfeval']}"), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_intermediate_max_mode_{max_mode}_{info['Nfeval']}"))

    return J, grad

#############################################################
## Perform optimization
#############################################################
oustr_dict_outer=[]
for max_mode in max_modes:
    oustr_dict_inner=[]
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))
    qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=quasisymmetry_helicity_m, helicity_n=quasisymmetry_helicity_n)
    objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1), (vmec.mean_iota, iota_target, iota_weight)]
    if magnetic_well: objective_tuple.append((vmec.vacuum_well, vacuum_well_target, vacuum_well_weight))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)

    dofs = np.concatenate((JF.x, vmec.x))
    bs.set_points(surf.gamma().reshape((-1, 3)))

    if stage_1:
        pprint(f'  Performing Stage 1 optimization with {MAXITER_stage_1} iterations')
        os.chdir(vmec_results_path)
        least_squares_mpi_solve(prob, mpi, grad=True, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, max_nfev=MAXITER_stage_1)
        os.chdir(this_path)
        if mpi.proc0_world:
            with open(debug_output_file, "a") as myfile:
                try:
                    myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                    myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                    myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                    myfile.write(f"\nMagnetic well at max_mode {max_mode}: {vmec.vacuum_well()}")
                    myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                except Exception as e:
                    myfile.write(e)

    if mpi.proc0_world:
        info_coils={'Nfeval':0}
        oustr_dict=[]
        pprint(f'  Performing Stage 2 optimization with {MAXITER_stage_2} iterations')
        res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=(info_coils,oustr_dict), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
        dofs[:-number_vmec_dofs] = res.x
        JF.x = dofs[:-number_vmec_dofs]
        curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_after_inner_loop_max_mode_{max_mode}"))
        bs.save(os.path.join(coils_results_path,f"biot_savart_inner_loop_max_mode_{max_mode}.json"))
        df = pd.DataFrame(oustr_dict)
        plot_df_stage2(df, max_mode)
        with open(debug_output_file, "a") as myfile:
            try:
                myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                myfile.write(f"\nMagnetic well at max_mode {max_mode}: {vmec.vacuum_well()}")
                myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
            except Exception as e:
                myfile.write(e)
    mpi.comm_world.Bcast(dofs, root=0)

    if single_stage:
        pprint(f'  Performing single stage optimization with {MAXITER_single_stage} iterations')
        if finite_beta:
            # If in finite beta, MPI is used to compute the gradients of J=J_stage1+J_stage2
            prob_jacobian = FiniteDifference(prob.objective, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method="centered")
            res = minimize(fun, dofs, args=(prob_jacobian,{'Nfeval':0},max_mode,oustr_dict_inner), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)
            oustr_dict_outer.append(oustr_dict_inner)
        else:
            # If in vacuum, MPI is used to compute the gradients of J=J_stage1 only
            with MPIFiniteDifference(prob.objective, mpi, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method="centered") as prob_jacobian:
                if mpi.proc0_world:
                    res = minimize(fun, dofs, args=(prob_jacobian,{'Nfeval':0},max_mode,oustr_dict_inner), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)
                    oustr_dict_outer.append(oustr_dict_inner)

    if mpi.proc0_world:
        pointData = {"B_N": np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path,'surf_opt_max_mode_'+str(max_mode)), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,'curves_opt_max_mode_'+str(max_mode)))
        bs.save(os.path.join(coils_results_path,'biot_savart_opt_max_mode_'+str(max_mode)+'.json'))
        vmec.write_input(os.path.join(vmec_results_path, f'input.CNT_maxmode{max_mode}'))

    os.chdir(vmec_results_path)
    try:
        pprint(f"Aspect ratio at max_mode {max_mode}: {vmec.aspect()}")
        pprint(f"Mean iota at {max_mode}: {vmec.mean_iota()}")
        pprint(f"Quasisymmetry objective at max_mode {max_mode}: {qs.total()}")
        pprint(f"Magnetic well at max_mode {max_mode}: {vmec.vacuum_well()}")
        pprint(f"Squared flux at max_mode {max_mode}: {Jf.J()}")
    except Exception as e:
        pprint(e)
    os.chdir(this_path)
    if mpi.proc0_world:
        try:
            df = pd.DataFrame(oustr_dict_inner)
            df.to_csv(os.path.join(this_path, f'output_max_mode_{max_mode}.csv'), index_label='index')
            ax=df.plot(
                kind='line',
                logy=True,
                y=['J','Jf','B.n','Jquasisymmetry', 'Jwell','Jiota','Jaspect', 'J_length','J_CC','J_LENGTH_PENALTY','J_CURVATURE'],#,'J_CENTER_CURVES'],#,'J_MSC','J_ALS'],#,'C-C-Sep','C-S-Sep'],
                linewidth=0.8)
            ax.set_ylim(bottom=1e-9, top=None)
            ax.set_xlabel('Number of function evaluations')
            ax.set_ylabel('Objective function')
            plt.legend(loc=3, prop={'size': 6})
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'optimization_stage3_max_mode_{max_mode}.pdf'), bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        except Exception as e:
            pprint(e)

#############################################################
## Save figures for coils, surfaces and objective over time
#############################################################
if mpi.proc0_world:
    try:
        pointData = {"B_N": np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path,'surf_opt'), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,'curves_opt'))
        bs.save(os.path.join(coils_results_path,"biot_savart_opt.json"))
        vmec.write_input(os.path.join(this_path, f'input.CNT_final'))
        df = pd.DataFrame(oustr_dict_outer[0])
        df.to_csv(os.path.join(this_path, f'output_CNT_final.csv'), index_label='index')
        ax=df.plot(kind='line',
            logy=True,
            y=['J','Jf','B.n','Jquasisymmetry','Jwell','Jiota','Jaspect','J_length','J_CC','J_LENGTH_PENALTY','J_CURVATURE'],#,'J_CENTER_CURVES'],#,'J_MSC','J_ALS'],#,'C-C-Sep','C-S-Sep'],
            linewidth=0.8)
        ax.set_ylim(bottom=1e-9, top=None)
        plt.legend(loc=3, prop={'size': 6})
        ax.set_xlabel('Number of function evaluations')
        ax.set_ylabel('Objective function')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'optimization_stage3_final.pdf'), bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    except Exception as e:
        pprint(e)

#############################################
## Print main results
#############################################
os.chdir(vmec_results_path)
try:
    pprint(f"Aspect ratio after optimization: {vmec.aspect()}")
    pprint(f"Mean iota after optimization: {vmec.mean_iota()}")
    pprint(f"Quasisymmetry objective after optimization: {qs.total()}")
    pprint(f"Squared flux after optimization: {Jf.J()}")
except Exception as e:
    pprint(e)

#############################################
## Final VMEC equilibrium
#############################################
os.chdir(this_path)
try:
    vmec_final = Vmec(os.path.join(this_path, f'input.CNT_final'))
    vmec_final.indata.ns_array[:2]    = [  16,    51]#,    101,   151,   201]
    vmec_final.indata.niter_array[:2] = [ 4000, 10000]#,  4000,  5000, 10000]
    vmec_final.indata.ftol_array[:2]  = [1e-12, 1e-13]#, 1e-14, 1e-15, 1e-15]
    vmec_final.run()
    if mpi.proc0_world:
        shutil.move(os.path.join(this_path, f"wout_CNT_final_000_000000.nc"), os.path.join(this_path, f"wout_CNT_final.nc"))
        os.remove(os.path.join(this_path, f'input.CNT_final_000_000000'))
except Exception as e:
    pprint('Exception when creating final vmec file:')
    pprint(e)

#############################################
## Create results figures
#############################################
if os.path.isfile(os.path.join(this_path, f"wout_CNT_final.nc")):
    pprint('Found final vmec file')
    sys.path.insert(1, os.path.join(parent_path, '../single_stage/plotting'))
    if mpi.proc0_world:
        pprint("Plot VMEC result")
        import vmecPlot2
        vmecPlot2.main(file=os.path.join(this_path, f"wout_CNT_final.nc"), name='CNT', figures_folder=OUT_DIR, coils_curves=curves)

        pprint('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec_final, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        if mpi.proc0_world:
            b1.bx.write_boozmn(os.path.join(vmec_results_path,"boozmn_CNT.nc"))
            pprint("Plot BOOZ_XFORM")
            fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_CNT.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_CNT.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_CNT.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_CNT.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_CNT.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()

stop = time.time()

pprint("============================================")
pprint("Finished optimization")
pprint(f"Took {stop-start} seconds")
pprint("============================================")
