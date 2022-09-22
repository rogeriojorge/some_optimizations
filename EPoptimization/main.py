#!/usr/bin/env python
import os
import glob
import time
import shutil
import vmecPlot2
import numpy as np
import pandas as pd
from mpi4py import MPI
import booz_xform as bx
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve, least_squares_serial_solve
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)
mpi = MpiPartition()
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
############################################################################
#### Input Parameters
############################################################################
MAXITER = 40
max_modes = [1]
QA_or_QH = 'QH'
opt_quasisymmetry = False
opt_EP = True
opt_well = False
opt_iota = False

s_initial = 0.3  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
nparticles = 128  # number of particles
tfinal = 1e-5  # total time of tracing in seconds
nsamples = 10000 # number of time steps
multharm = 3 # angular grid factor
ns_s = 3 # spline order over s
ns_tp = 3 # spline order over theta and phi
nper = 1000 # number of periods for initial field line
npoiper = 100 # number of points per period on this field line
npoiper2 = 128 # points per period for integrator step
notrace_passing = 0 # if 1 skips tracing of passing particles, else traces them

nruns_robustness = 20 # number of runs when testing for robustness of cost function

iota_target = -0.42
weight_optEP = 10.0
if QA_or_QH == 'QA':
    B_scale = 8.58 / 2
    Aminor_scale = 8.5 / 2
    aspect_ratio_target = 6
else:
    B_scale = 6.55 / 2
    Aminor_scale = 12.14 / 2
    aspect_ratio_target = 7

diff_rel_step = 1e-5
diff_abs_step = 1e-7
######################################
######################################
if QA_or_QH == 'QA': filename = os.path.join(os.path.dirname(__file__), 'initial_configs', 'input.nfp2_QA')
else: filename = os.path.join(os.path.dirname(__file__), 'initial_configs', 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)
vmec.keep_all_files = True
surf = vmec.boundary
g_particle = ChargedParticleEnsemble(r_initial=s_initial)
######################################
this_path = Path(__file__).parent.resolve()
OUT_DIR=os.path.join(this_path,f'out_s{s_initial}_NFP{vmec.indata.nfp}')
if opt_quasisymmetry: OUT_DIR+=f'_{QA_or_QH}'
if opt_well: OUT_DIR+=f'_well'
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
######################################
def EPcostFunction(v: Vmec):
    v.run()
    g_field_temp = Simple(wout_filename=v.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale, multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
    g_orbits_temp = ParticleEnsembleOrbit_Simple(g_particle,g_field_temp,tfinal=tfinal,nparticles=nparticles,nsamples=nsamples,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
    final_loss_fraction = g_orbits_temp.total_particles_lost
    print(f'Loss fraction = {final_loss_fraction}')
    # print(f'VMEC dofs = {v.x}')
    return final_loss_fraction
optEP = make_optimizable(EPcostFunction, vmec)
######################################
pprint("Testing robustness of cost function")
start_time = time.time()
costfunction_array=[]
for i in range(nruns_robustness):
    costfunction_array.append(optEP.J())
total_time = time.time() - start_time
params_dict = {
    'QA_or_QH': QA_or_QH,
    's_initial': s_initial,
    'nparticles': nparticles,
    'tfinal': tfinal,
    'multharm': multharm,
    'ns_s': ns_s,
    'ns_tp': ns_tp,
    'nper': nper,
    'npoiper': npoiper,
    'npoiper2': npoiper2,
    'notrace_passing': notrace_passing,
    'nruns': nruns_robustness,
    'total_time': total_time,
    'std_cost_function': np.std(costfunction_array),
    'mean_cost_function': np.mean(costfunction_array),
    'coefficient_variation': np.std(costfunction_array)/np.mean(costfunction_array)
}
output_path_parameters=os.path.join(this_path, 'opt_parameters.csv')
print(output_path_parameters)
df = pd.DataFrame(data=[params_dict])
if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
df.to_csv(output_path_parameters, mode='a', header=False, index=False)
print(np.std(costfunction_array)/np.mean(costfunction_array))
# exit()
######################################
pprint("Initial aspect ratio:", vmec.aspect())
pprint("Initial mean iota:", vmec.mean_iota())
pprint("Initial magnetic well:", vmec.vacuum_well())
if MPI.COMM_WORLD.rank == 0:
    g_field = Simple(wout_filename=vmec.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale, multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
    g_orbits = ParticleEnsembleOrbit_Simple(g_particle,g_field,tfinal=tfinal,nparticles=nparticles,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
    pprint("Initial loss fraction:", g_orbits.total_particles_lost)
######################################
if QA_or_QH == 'QH': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)    
opt_tuple = [(vmec.aspect, aspect_ratio_target, 1)]
if opt_well: opt_tuple.append((vmec.vacuum_well, 0.1, 1))
if opt_iota: opt_tuple.append((vmec.mean_iota, iota_target, 1))
if opt_EP: opt_tuple.append((optEP.J, 0, weight_optEP))
if opt_quasisymmetry: opt_tuple.append((qs.residuals, 0, 1))
pprint("Quasisymmetry objective before optimization:", qs.total())
######################################
initial_dofs=np.copy(surf.x)
def fun(dofss):
    prob.x = initial_dofs
    return prob.objective()
for max_mode in max_modes:
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    initial_dofs=np.copy(surf.x)
    dofs=surf.x
    ######################################
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    if MPI.COMM_WORLD.rank == 0: pprint("Total objective before optimization:", prob.objective())
    pprint('-------------------------')
    pprint(f'Optimizing with max_mode = {max_mode}')
    pprint('-------------------------')
    res = minimize(fun, dofs, method='BFGS', options={'maxiter': MAXITER}, tol=1e-9)
    # least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
    # least_squares_serial_solve(prob, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
    ######################################
    pprint("Final aspect ratio:", vmec.aspect())
    pprint("Final mean iota:", vmec.mean_iota())
    pprint("Final magnetic well:", vmec.vacuum_well())
    pprint("Quasisymmetry objective after optimization:", qs.total())
    if MPI.COMM_WORLD.rank == 0:
        g_field = Simple(wout_filename=vmec.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale,multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
        g_orbits = ParticleEnsembleOrbit_Simple(g_particle,g_field,tfinal=tfinal,nparticles=nparticles,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
        pprint("Final loss fraction:", g_orbits.total_particles_lost)
        pprint("Total objective after optimization:", prob.objective())
    ######################################
if MPI.COMM_WORLD.rank == 0:
    try:
        for objective_file in glob.glob("objective_*"):
            os.remove(objective_file)
        for residuals_file in glob.glob("residuals_*"):
            os.remove(residuals_file)
        for jac_file in glob.glob("jac_log_*"):
            os.remove(jac_file)
        for threed_file in glob.glob("threed1.*"):
            os.remove(threed_file)
        for threed_file in glob.glob("wout_*"):
            os.remove(threed_file)
        for threed_file in glob.glob("input.*"):
            os.remove(threed_file)
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
boozxform_nsurfaces=10
pprint('Defining surfaces where to compute Boozer coordinates')
booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
pprint(f' booz_surfaces={booz_surfaces}')
b1.register(booz_surfaces)
pprint('Running BOOZ_XFORM')
b1.run()
if mpi.proc0_world:
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