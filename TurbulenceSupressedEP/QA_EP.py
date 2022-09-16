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
MAXITER = 30
max_modes = [1]
s_EP = 0.1
alphas_EP=0
QA_or_QH = 'QH'
weight_optEP = 0.01
omega_EP_constraint = -1.
aspect_ratio_target = 6
theta_min_max=np.pi/20
ntheta_PEST=20
opt_quasisymmetry = False
opt_EP = True
######################################
######################################
if QA_or_QH == 'QA': filename = os.path.join(os.path.dirname(__file__), 'input.nfp2_QA')
else: filename = os.path.join(os.path.dirname(__file__), 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)
surf = vmec.boundary
######################################
OUT_DIR=f'out_constraint-minus{abs(omega_EP_constraint)}_s{s_EP}_alpha{alphas_EP}_NFP{vmec.indata.nfp}'
if opt_quasisymmetry: OUT_DIR+=f'_{QA_or_QH}'
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
######################################
def EPcostFunction(v):
    v.run()
    theta_PEST = np.linspace(-theta_min_max, theta_min_max, ntheta_PEST)
    fl = vmec_fieldlines(vmec, s_EP, [alphas_EP], theta1d=theta_PEST)
    return fl.gbdrift[0][0] - omega_EP_constraint
optEP = make_optimizable(EPcostFunction, vmec)
######################################
pprint("Initial aspect ratio:", vmec.aspect())
pprint("Initial mean iota:", vmec.mean_iota())
pprint("Initial magnetic well:", vmec.vacuum_well())
def middle(arr): return arr[int((len(arr)-1)/2)]
fl1 = vmec_fieldlines(vmec, s_EP, alphas_EP, theta1d=np.linspace(-2*np.pi, 2*np.pi, 150), plot=True, show=False)
plt.savefig(f'Initial_profiles_s{s_EP}_alpha{alphas_EP}.png');plt.close()
pprint("Initial gbdrift at 0:", middle(fl1.gbdrift[0][0]))
######################################
if QA_or_QH == 'QH': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)    
opt_tuple = [(vmec.aspect, aspect_ratio_target, 1)]# (vmec.vacuum_well, 0.1, 1)]
if QA_or_QH == 'QA': opt_tuple.append((vmec.mean_iota, 0.43, 1))
if opt_EP: opt_tuple.append((optEP.J, 0, weight_optEP))
if opt_quasisymmetry: opt_tuple.append((qs.residuals, 0, 1))
pprint("Quasisymmetry objective before optimization:", qs.total())
######################################
for max_mode in max_modes:
    pprint('-------------------------')
    pprint(f'Optimizing with max_mode = {max_mode}')
    pprint('-------------------------')
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    ######################################
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=MAXITER)
    ######################################
    pprint("Final aspect ratio:", vmec.aspect())
    pprint("Final mean iota:", vmec.mean_iota())
    pprint("Final magnetic well:", vmec.vacuum_well())
    fl2 = vmec_fieldlines(vmec, s_EP, alphas_EP, theta1d=np.linspace(-2*np.pi, 2*np.pi, 150), plot=True, show=False)
    plt.savefig(f'Final_profiles_s{s_EP}_alpha{alphas_EP}.png');plt.close()
    pprint("Final gbdrift at 0:", middle(fl2.gbdrift[0][0]))
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
    # os.remove("parvmecinfo.txt")
    # for input_file in glob.glob("input.nfp4_QH_warm_start_000*"):
    #     os.remove(input_file)
    # for wout_file in glob.glob("wout_nfp4_QH_warm_start_000_*"):
    #     os.remove(wout_file)
except Exception as e:
    pprint(e)
############################################################################
############################################################################