#!/usr/bin/env python

import os
import glob
import numpy as np
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
from simsopt import make_optimizable
from simsopt.solve import least_squares_mpi_solve
import matplotlib.pyplot as plt
from mpi4py import MPI
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:  # only pprint on rank 0
        print(*args, **kwargs)
mpi = MpiPartition()
############################################################################
#### Input Parameters
############################################################################
MAXITER = 25
max_mode = 2
ntheta_PEST=20
s_EP = 0.1
alphas_EP=0
QA_or_QH = 'QH'
weight_optEP = 0.001
omega_EP_constraint = 0
aspect_ratio_target = 5
############################################################################
############################################################################
if QA_or_QH == 'QA':
    filename = os.path.join(os.path.dirname(__file__), 'input.nfp2_QA')
else:
    filename = os.path.join(os.path.dirname(__file__), 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)
######################################
theta_PEST = np.linspace(-np.pi, np.pi, ntheta_PEST)
def EPcostFunction(v):
    v.run()
    fl = vmec_fieldlines(vmec, s_EP, [alphas_EP], theta1d=theta_PEST)
    return fl.B_cross_grad_B_dot_grad_alpha[0][0] - omega_EP_constraint
optEP = make_optimizable(EPcostFunction, vmec)
######################################
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
if QA_or_QH == 'QH':
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
else:
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)    
opt_tuple = [(vmec.aspect, aspect_ratio_target, 1),
            # (vmec.vacuum_well, 0.1, 1),
            (optEP.J, 0, weight_optEP),
            (qs.residuals, 0, 1)
            ]
if QA_or_QH == 'QA': opt_tuple.append((vmec.mean_iota, 0.43, 1))
######################################
prob = LeastSquaresProblem.from_tuples(opt_tuple)
######################################
pprint("Initial aspect ratio:", vmec.aspect())
pprint("Initial mean iota:", vmec.mean_iota())
pprint("Initial magnetic well:", vmec.vacuum_well())
fl1 = vmec_fieldlines(vmec, s_EP, alphas_EP, theta1d=theta_PEST, plot=True, show=False)
plt.savefig(f'Initial_profiles_s{s_EP}_alpha{alphas_EP}.png');plt.close()
pprint("Initial max B_cross_grad_B_dot_grad_alpha:", np.max(fl1.B_cross_grad_B_dot_grad_alpha[0][0]))
pprint("Quasisymmetry objective before optimization:", qs.total())
pprint("Total objective before optimization:", prob.objective())
######################################
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=MAXITER)
######################################
pprint("Final aspect ratio:", vmec.aspect())
pprint("Final mean iota:", vmec.mean_iota())
pprint("Final magnetic well:", vmec.vacuum_well())
fl2 = vmec_fieldlines(vmec, s_EP, alphas_EP, theta1d=theta_PEST, plot=True, show=False)
plt.savefig(f'Final_profiles_s{s_EP}_alpha{alphas_EP}.png');plt.close()
pprint("Final max B_cross_grad_B_dot_grad_alpha:", np.max(fl2.B_cross_grad_B_dot_grad_alpha[0][0]))
pprint("Quasisymmetry objective after optimization:", qs.total())
pprint("Total objective after optimization:", prob.objective())
######################################
try:
    for objective_file in glob.glob("objective_*"):
        os.remove(objective_file)
    for residuals_file in glob.glob("residuals_*"):
        os.remove(residuals_file)
    for jac_file in glob.glob("jac_log_*"):
        os.remove(jac_file)
    for threed_file in glob.glob("threed1.*"):
        os.remove(threed_file)
    os.remove("parvmecinfo.txt")
#     for input_file in glob.glob("input.nfp4_QH_warm_start_000*"):
#         os.remove(input_file)
#     for wout_file in glob.glob("wout_nfp4_QH_warm_start_000_*"):
#         os.remove(wout_file)
except Exception as e:
    pprint(e)
############################################################################
############################################################################