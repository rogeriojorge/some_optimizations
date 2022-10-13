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
import matplotlib.pyplot as plt
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve
from GX_io import GX_Runner, read_GX_output
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
from scipy.optimize import dual_annealing
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)
############################################################################
#### Input Parameters
############################################################################
MAXITER = 6
max_modes = [1]
initial_config = 'input.nfp4_QH'
plot_result = True
optimizer = 'least_squares' #'dual_annealing'
use_previous_results_if_available = False
weight_optTurbulence = 100.0
aspect_ratio_target = 7
diff_rel_step = 1e-3
diff_abs_step = 1e-5
no_local_search = True
output_path_parameters=f'output_{optimizer}.csv'
######################################
######################################
OUT_DIR_APPENDIX=f'output'
OUT_DIR = os.path.join(this_path, OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
######################################
dest = os.path.join(OUT_DIR,OUT_DIR_APPENDIX+'_previous')
if use_previous_results_if_available and (os.path.isfile(os.path.join(OUT_DIR,'input.final')) or os.path.isfile(os.path.join(dest,'input.final'))):
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(dest, exist_ok=True)
        if os.path.isfile(os.path.join(OUT_DIR, 'input.final')) and not os.path.isfile(os.path.join(dest, 'input.final')):
            files = os.listdir(OUT_DIR)
            for f in files:
                shutil.move(os.path.join(OUT_DIR, f), dest)
    else:
        time.sleep(0.5)
    filename = os.path.join(dest, 'input.final')
else:
    filename = os.path.join(this_path, initial_config)
os.chdir(OUT_DIR)
vmec = Vmec(filename, mpi=mpi, verbose=False)
vmec.keep_all_files = True
surf = vmec.boundary
######################################
def output_dofs_to_csv(dofs,mean_iota,aspect,heat_flux):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['heat_flux']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[heat_flux]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
    df.to_csv(output_path_parameters, mode='a', header=False, index=False)
######################################
######################################
##### CALCULATE HEAT FLUX HERE #######
######################################
######################################
def CalculateHeatFlux(v: Vmec):
    s_surface = 0.1
    alpha_surface = [0]
    theta_min_max = 2*np.pi
    ntheta_PEST = 30
    theta_PEST = np.linspace(-theta_min_max, theta_min_max, ntheta_PEST)
    fl = vmec_fieldlines(v, s_surface, alpha_surface, theta1d=theta_PEST)
    iota = fl.iota
    shat = fl.shat
    B_cross_grad_s_dot_grad_alpha = fl.B_cross_grad_s_dot_grad_alpha
    B_cross_grad_B_dot_grad_alpha = fl.B_cross_grad_B_dot_grad_alpha
    B_cross_grad_B_dot_grad_psi = fl.B_cross_grad_B_dot_grad_psi
    B_cross_kappa_dot_grad_psi = fl.B_cross_kappa_dot_grad_psi
    B_cross_kappa_dot_grad_alpha = fl.B_cross_kappa_dot_grad_alpha
    grad_alpha_dot_grad_alpha = fl.grad_alpha_dot_grad_alpha
    grad_alpha_dot_grad_psi = fl.grad_alpha_dot_grad_psi
    grad_psi_dot_grad_psi = fl.grad_psi_dot_grad_psi
    L_reference = fl.L_reference
    B_reference = fl.B_reference
    toroidal_flux_sign = fl.toroidal_flux_sign
    bmag = fl.bmag
    gradpar_theta_pest = fl.gradpar_theta_pest
    gradpar_phi = fl.gradpar_phi
    gds2 = fl.gds2
    gds21 = fl.gds21
    gds22 = fl.gds22
    gbdrift = fl.gbdrift
    gbdrift0 = fl.gbdrift0
    cvdrift = fl.cvdrift
    cvdrift0 = fl.cvdrift0
    try:
        gx_class = GX_Runner(os.path.join(this_path,'gx-sample.in'))
        gx_class.execute()
        heat_flux = read_GX_output('gx_output')
        return heat_flux
    except Exception as e:
        print(e)
        return 1
######################################
######################################
######################################
def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return 1e3
    heat_flux = CalculateHeatFlux(v)
    out_str = f'Heat flux = {heat_flux:1f} with aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s'
    print(out_str)
    output_dofs_to_csv(v.x,v.mean_iota(),v.aspect(),heat_flux)
    return heat_flux
optTurbulence = make_optimizable(TurbulenceCostFunction, vmec)
######################################
try:
    pprint("Initial aspect ratio:", vmec.aspect())
    pprint("Initial mean iota:", vmec.mean_iota())
    pprint("Initial magnetic well:", vmec.vacuum_well())
except Exception as e: pprint(e)
if MPI.COMM_WORLD.rank == 0:
    heat_flux = CalculateHeatFlux(vmec)
    pprint("Initial heat flux:", heat_flux)
######################################
initial_dofs=np.copy(surf.x)
def fun(dofss):
    prob.x = dofss
    return prob.objective()
for max_mode in max_modes:
    output_path_parameters=f'output_{optimizer}_maxmode{max_mode}.csv'
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    initial_dofs=np.copy(surf.x)
    dofs=surf.x
    ######################################  
    opt_tuple = [(vmec.aspect, aspect_ratio_target, 1)]
    opt_tuple.append((optTurbulence.J, 0, weight_optTurbulence))
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    if MPI.COMM_WORLD.rank == 0: pprint("Total objective before optimization:", prob.objective())
    pprint('-------------------------')
    pprint(f'Optimizing with max_mode = {max_mode}')
    pprint('-------------------------')
    if optimizer == 'dual_annealing':
        initial_temp = 1000
        visit = 2.0
        bounds = [(-0.25,0.25) for _ in dofs]
        res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, initial_temp=initial_temp,visit=visit, no_local_search=no_local_search, x0=dofs)
    elif optimizer == 'least_squares':
        least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
    else: print('Optimizer not available')
    ######################################
    if MPI.COMM_WORLD.rank == 0:
        try: 
            pprint("Final aspect ratio:", vmec.aspect())
            pprint("Final mean iota:", vmec.mean_iota())
            pprint("Final magnetic well:", vmec.vacuum_well())
            heat_flux = CalculateHeatFlux(vmec)
            pprint("Final heat flux:", heat_flux)
        except Exception as e: pprint(e)
    ######################################
if MPI.COMM_WORLD.rank == 0: vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
######################################
### PLOT RESULT
######################################
if plot_result and MPI.COMM_WORLD.rank==0:
    vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'), mpi=mpi)
    vmec_final.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
    vmec_final.indata.niter_array[:3] = [ 4000, 10000,  4000]#,  5000, 10000]
    vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
    vmec_final.run()
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