#!/usr/bin/env python
import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from simsopt.mhd import VirtualCasing
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable, make_optimizable
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import CurveLength, MeanSquaredCurvature, LpCurveCurvature
from simsopt.objectives import SquaredFlux, LeastSquaresProblem, QuadraticPenalty
from simsopt._core.finite_difference import finite_difference_steps, FiniteDifference, MPIFiniteDifference
logger = logging.getLogger(__name__)
mpi = MpiPartition()

# Define absolute steps
abs_step_array = [1e-4,1e-5,1e-6]#,1e-7,1e-8]
rel_step_value = 0

# Input parameters
derivative_algorithm = "centered"
LENGTHBOUND=20
LENGTH_CON_WEIGHT=1e-2
JACOBIAN_THRESHOLD=50
CURVATURE_THRESHOLD=5
CURVATURE_WEIGHT=1e-6
MSC_THRESHOLD=5
MSC_WEIGHT=1e-6
max_mode=1
coils_objective_weight=1
ncoils=3
R0=1
R1=0.5
order=2
nphi=70
ntheta=30
finite_beta = False
vc_src_nphi = 20
OUT_DIR = f"output"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)

# Stage 1
if finite_beta:
    vmec_file = '../input.nfp2_QAS_FiniteBeta'
else:
    vmec_file = '../input.nfp2_QA_lowres'
vmec = Vmec(vmec_file, nphi=nphi, ntheta=ntheta, mpi=mpi, verbose=False)
surf = vmec.boundary
objective_tuple = [(vmec.aspect, 4, 1),(vmec.mean_iota, 0.4, 1)]
prob = LeastSquaresProblem.from_tuples(objective_tuple)
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(surf.x))

# Finite Beta Virtual Casing Principle
if finite_beta:
    print('Running the virtual casing calculation')
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
    total_current = Vmec(vmec_file).external_current() / (2 * surf.nfp)
    initial_current = total_current / ncoils * 1e-5
else:
    initial_current = 1

# Stage 2
base_curves = create_equally_spaced_curves(ncoils, vmec.indata.nfp, stellsym=True, R0=R0, R1=R1, order=order)
if finite_beta:
    base_currents = [Current(initial_current) * 1e5 for i in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
else:
    base_currents = [Current(initial_current) * 1e5 for i in range(ncoils)]
    base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, vmec.indata.nfp, True)
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
if finite_beta:
    Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
else:
    Jf = SquaredFlux(surf, bs, local=True)
Jls = [CurveLength(c) for c in base_curves]
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
JF = Jf \
   + CURVATURE_WEIGHT * sum(Jcs) \
   + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
   + LENGTH_CON_WEIGHT * QuadraticPenalty(sum(Jls[i] for i in range(len(base_curves))), LENGTHBOUND)

def set_dofs(x0):
    if np.sum(JF.x!=x0[:-number_vmec_dofs])>0:
        JF.x = x0[:-number_vmec_dofs]
    if np.sum(prob.x!=x0[-number_vmec_dofs:])>0:
        prob.x = x0[-number_vmec_dofs:]
        if finite_beta:
            vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
            Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
            JF.opts[0].opts[0].opts[0] = Jf
    bs.set_points(surf.gamma().reshape((-1, 3)))

def fun(x0):
    set_dofs(x0)
    J_stage_1 = prob.objective()
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    return J

def grad_fun_analytical(x0, finite_difference_rel_step=0, finite_difference_abs_step=1e-7):
    set_dofs(x0)
    ## Finite differences for the second-stage objective function
    coils_dJ = JF.dJ()
    ## Mixed term - derivative of squared flux with respect to the surface shape
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    B = bs.B().reshape((nphi, ntheta, 3))
    dB_by_dX = bs.dB_by_dX().reshape((nphi,ntheta, 3, 3))
    Bcoil = bs.B().reshape(n.shape)
    unitn = n * (1./absn)[:, :, None]
    Bcoil_n = np.sum(Bcoil*unitn, axis=2)
    assert Jf.local
    if finite_beta:
        B_n = (Bcoil_n - vc.B_external_normal)
        B_diff = Bcoil - vc.B_external
        B_N = np.sum(B_diff * n, axis=2)
    else:
        B_n = Bcoil_n
        B_diff = Bcoil
        B_N = np.sum(Bcoil * n, axis=2)
    mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
    dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
    dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
    deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi*ntheta)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi*ntheta))
    ## Check with the FiniteDifference class if this derivative is being computed correctly
    mixed_dJ = Derivative({surf: deriv})(surf)
    ## Finite differences for the first-stage objective function
    prob_jacobian = FiniteDifference(prob.objective, mpi, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method=derivative_algorithm)
    prob_dJ = prob_jacobian.jac(prob.x)
    ## Put both gradients together
    grad_with_respect_to_coils = coils_objective_weight * coils_dJ
    grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    return grad

def grad_fun_numerical(x0, diff_method: str = derivative_algorithm, abs_step = 1e-7, rel_step = 0):
    set_dofs(x0)
    grad = np.zeros(len(x0),)
    steps = finite_difference_steps(x0, abs_step=abs_step, rel_step=rel_step)
    if diff_method == "centered":
        for j in range(len(x0)):
            print(f'FiniteDifference iteration {j}/{len(x0)}')
            x = np.copy(x0)
            x[j] = x0[j] + steps[j]
            fplus = fun(x)
            x[j] = x0[j] - steps[j]
            fminus = fun(x)
            grad[j] = (fplus - fminus) / (2 * steps[j])
    elif diff_method == "forward":
        f0 = fun(x0)
        for j in range(len(x0)):
            print(f'FiniteDifference iteration {j}/{len(x0)}')
            x = np.copy(x0)
            x[j] = x0[j] + steps[j]
            fplus = fun(x)
            grad[j] = (fplus - f0) / steps[j]
    return grad

# Set degrees of freedom
dofs = np.concatenate((JF.x, vmec.x))

# Perform regression test
sqrt_squared_diff_grad_with_respect_to_coils_array=[]
sqrt_squared_diff_grad_with_respect_to_surface_array=[]
start_outer = time.time()
for abs_step in abs_step_array:
    set_dofs(dofs)
    start_inner = time.time()
    gradAnalytical = grad_fun_analytical(dofs, finite_difference_rel_step=rel_step_value, finite_difference_abs_step=abs_step)
    gradAnalytical_with_respect_to_coils = gradAnalytical[:-number_vmec_dofs]
    gradAnalytical_with_respect_to_surface = gradAnalytical[-number_vmec_dofs:]

    gradNumerical = np.empty(len(dofs))
    opt = make_optimizable(fun, dofs, dof_indicators=["dof"])
    with MPIFiniteDifference(opt.J, mpi, diff_method=derivative_algorithm, abs_step=abs_step, rel_step=rel_step_value) as fd:
        if mpi.proc0_world:
            print(f'abs_step={abs_step}')
            gradNumerical = np.array(fd.jac()[0])
    mpi.comm_world.Bcast(gradNumerical, root=0)

    # gradNumerical = grad_fun_numerical(x0=dofs, abs_step=abs_step, rel_step=rel_step_value)

    gradNumerical_with_respect_to_coils = gradNumerical[:-number_vmec_dofs]
    gradNumerical_with_respect_to_surface = gradNumerical[-number_vmec_dofs:]

    sqrt_squared_diff_grad_with_respect_to_coils = np.sqrt(np.sum((gradAnalytical_with_respect_to_coils - gradNumerical_with_respect_to_coils)**2))
    sqrt_squared_diff_grad_with_respect_to_surface = np.sqrt(np.sum((gradAnalytical_with_respect_to_surface - gradNumerical_with_respect_to_surface)**2))

    sqrt_squared_diff_grad_with_respect_to_coils_array.append(sqrt_squared_diff_grad_with_respect_to_coils)
    sqrt_squared_diff_grad_with_respect_to_surface_array.append(sqrt_squared_diff_grad_with_respect_to_surface)
    if mpi.proc0_world: print(f' took {time.time()-start_inner}s')

if mpi.proc0_world:
    print(f'Outer abs_step loop took {time.time()-start_outer}s')

    # Plot and save results
    fig=plt.figure()
    plt.loglog(abs_step_array, sqrt_squared_diff_grad_with_respect_to_coils_array, label='coils grad')
    plt.loglog(abs_step_array, sqrt_squared_diff_grad_with_respect_to_surface_array, label='surface grad')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.xlabel('Absolute Step (finite difference)')
    plt.ylabel('RMS |Analytical Derivative - Finite Difference|')
    plt.tight_layout()
    plt.savefig("test_result.png", dpi=250)
    plt.show()

    # Print results
    print(f'abs_step_array={abs_step_array}')
    print(f'RMS differences grad coils={sqrt_squared_diff_grad_with_respect_to_coils_array}')
    print(f'RMS differences grad surface={sqrt_squared_diff_grad_with_respect_to_surface_array}')