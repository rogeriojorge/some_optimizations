#!/usr/bin/env python
import os
import glob
import shutil
import vmecPlot2
import numpy as np
from mpi4py import MPI
import booz_xform as bx
from pathlib import Path
import matplotlib.pyplot as plt
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:  # only pprint on rank 0
        print(*args, **kwargs)
mpi = MpiPartition()
############################################################################
#### Input Parameters
############################################################################
MAXITER = 80
max_modes = [3]
s_EP = 0.1
alphas_EP=0
QA_or_QH = 'QA'
weight_optEP = 0.03
omega_EP_constraint = 0.4
aspect_ratio_target = 6
theta_min_max=np.pi/20
ntheta_PEST=25
opt_quasisymmetry = True
opt_EP = True
opt_well = False
boozxform_nsurfaces=10
ftol=1e-7
######################################
######################################
if QA_or_QH == 'QA': filename = os.path.join(os.path.dirname(__file__), 'input.nfp2_QA')
else: filename = os.path.join(os.path.dirname(__file__), 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)
surf = vmec.boundary
######################################
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'out_constraint-minus{abs(omega_EP_constraint)}_s{s_EP}_alpha{alphas_EP}_NFP{vmec.indata.nfp}')
if opt_quasisymmetry: OUT_DIR+=f'_{QA_or_QH}'
if opt_well: OUT_DIR+=f'_well'
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
######################################
def EPcostFunction(v):
    v.run()
    theta_PEST = np.linspace(-theta_min_max, theta_min_max, ntheta_PEST)
    fl = vmec_fieldlines(vmec, s_EP, [alphas_EP], theta1d=theta_PEST)
    # print(f'On processor {MPI.COMM_WORLD.Get_rank()} result ={middle(fl.gbdrift[0][0])-omega_EP_constraint}')
    return fl.gbdrift[0][0] - omega_EP_constraint
optEP = make_optimizable(EPcostFunction, vmec)
######################################
pprint("Initial aspect ratio:", vmec.aspect())
pprint("Initial mean iota:", vmec.mean_iota())
pprint("Initial magnetic well:", vmec.vacuum_well())
def middle(arr): return arr[int((len(arr)-1)/2)]
fl1 = vmec_fieldlines(vmec, s_EP, alphas_EP, theta1d=np.linspace(-4*np.pi, 4*np.pi, 250), plot=True, show=False)
plt.savefig(f'Initial_profiles_s{s_EP}_alpha{alphas_EP}.png');plt.close()
pprint("Initial gbdrift at 0:", middle(fl1.gbdrift[0][0]))
######################################
if QA_or_QH == 'QH': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)    
opt_tuple = [(vmec.aspect, aspect_ratio_target, 1)]
if opt_well: opt_tuple.append((vmec.vacuum_well, 0.1, 1))
if QA_or_QH == 'QA': opt_tuple.append((vmec.mean_iota, 0.42, 10))
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
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-4, abs_step=1e-6, max_nfev=MAXITER, ftol=ftol)
    ######################################
    pprint("Final aspect ratio:", vmec.aspect())
    pprint("Final mean iota:", vmec.mean_iota())
    pprint("Final magnetic well:", vmec.vacuum_well())
    fl2 = vmec_fieldlines(vmec, s_EP, alphas_EP, theta1d=np.linspace(-4*np.pi, 4*np.pi, 250), plot=True, show=False)
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
######################################
vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'), mpi=mpi)
vmec_final.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
vmec_final.indata.niter_array[:3] = [ 4000, 10000,  4000]#,  5000, 10000]
vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
vmec_final.run()
if mpi.proc0_world:
    shutil.move(os.path.join(OUT_DIR, f"wout_final_000_000000.nc"), os.path.join(OUT_DIR, f"wout_final.nc"))
    os.remove(os.path.join(OUT_DIR, f'input.final_000_000000'))
    try: vmecPlot2.main(file=os.path.join(OUT_DIR, f"wout_final.nc"), name='EP_opt', figures_folder=OUT_DIR)
    except Exception as e: print(e)
    pprint('Creating Boozer class for vmec_final')
    b1 = Boozer(vmec_final, mpol=64, ntor=64)
    pprint('Defining surfaces where to compute Boozer coordinates')
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    pprint(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    pprint('Running BOOZ_XFORM')
    b1.run()
    b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn_single_stage.nc"))
    pprint("Plot BOOZ_XFORM")
    fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
############################################################################
############################################################################