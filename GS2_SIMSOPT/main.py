#!/usr/bin/env python3
import os
import glob
import time
import shutil
import netCDF4
import vmecPlot2
import subprocess
import numpy as np
import pandas as pd
from mpi4py import MPI
import booz_xform as bx
from pathlib import Path
from tempfile import mkstemp
from datetime import datetime
import matplotlib.pyplot as plt
from shutil import move, copymode
from os import fdopen, remove
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd.vmec_diagnostics import to_gs2
from scipy.optimize import dual_annealing
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)
start_time = time.time()
#########
## FOR GS2 to work with SIMSOPT's MPI, USE_MPI should be undefined during GS2's compilation
## override USE_MPI = 
#########
############################################################################
#### Input Parameters
############################################################################
MAXITER = 350
max_modes = [2]
initial_config = 'input.nfp2_QA'# 'input.nfp2_QA' #'input.nfp4_QH'
aspect_ratio_target = 7
opt_quasisymmetry = True
plot_result = True
optimizer = 'least_squares'#'dual_annealing' #'least_squares'
use_previous_results_if_available = False
s_radius = 0.25
alpha_fieldline = 0
phi_GS2 = np.linspace(-6*np.pi, 6*np.pi, 71)
nlambda = 15
nstep = 100
weight_optTurbulence = 10
diff_rel_step = 1e-3
diff_abs_step = 1e-5
MAXITER_LOCAL = 3
MAXFUN_LOCAL = 30
no_local_search = False
output_path_parameters=f'output_{optimizer}.csv'
HEATFLUX_THRESHOLD = 1e18
GROWTHRATE_THRESHOLD = 10
aspect_ratio_weight = 1e-1
gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2'
######################################
######################################
OUT_DIR_APPENDIX=f'output_MAXITER{MAXITER}_{optimizer}_{initial_config[6:]}'
if opt_quasisymmetry: OUT_DIR_APPENDIX+=f'_{initial_config[-2:]}'
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
        time.sleep(0.2)
    filename = os.path.join(dest, 'input.final')
else:
    filename = os.path.join(this_path, initial_config)
os.chdir(OUT_DIR)
vmec = Vmec(filename, verbose=False, mpi=mpi)
vmec.keep_all_files = True
surf = vmec.boundary
######################################
def output_dofs_to_csv(dofs,mean_iota,aspect,growth_rate,quasisymmetry_total):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['growth_rate'],['quasisymmetry_total']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[growth_rate],[quasisymmetry_total]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
    df.to_csv(output_path_parameters, mode='a', header=False, index=False)
########################################
########################################
##### CALCULATE GROWTH RATE HERE #######
########################################
########################################
def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)
def CalculateGrowthRate(v: Vmec):
    try:
        v.run()
        f_wout = v.output_file.split('/')[-1]
        gs2_input_name = f"gs2-{f_wout[5:-3]}"
        gs2_input_file = os.path.join(OUT_DIR,f'{gs2_input_name}.in')
        shutil.copy(os.path.join(this_path,'gs2Input.in'),gs2_input_file)
        gridout_file = os.path.join(OUT_DIR,f'grid_{gs2_input_name}.out')
        replace(gs2_input_file,' gridout_file = "grid.out"',f' gridout_file = "grid_{gs2_input_name}.out"')
        replace(gs2_input_file,' nstep = 150',f' nstep = {nstep}')
        to_gs2(gridout_file, v, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=nlambda)
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        # f_log = os.path.join(OUT_DIR,f"{gs2_input_name}.log")
        # with open(f_log, 'w') as fp:
        p = subprocess.Popen(bashCommand.split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)#stdout=fp)
        p.wait()
        # subprocess.call(bashCommand, shell=True)
        fractionToConsider = 0.4 # fraction of time from the simulation period to consider
        file2read = netCDF4.Dataset(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"),'r')
        tX = file2read.variables['t'][()]
        qparflux2_by_ky = file2read.variables['qparflux2_by_ky'][()]
        startIndexX  = int(len(tX)*(1-fractionToConsider))
        qavg = np.mean(qparflux2_by_ky[startIndexX:,0,:])
        # omega_average = file2read.variables['omega_average'][()]
        # growth_rate = np.max(np.array(omega_average)[-1,:,0,1])
        phi2 = np.log(file2read.variables['phi2'][()])
        t = file2read.variables['t'][()]
        startIndex = int(len(t)*(1-fractionToConsider))
        mask = np.isfinite(phi2)
        data_x = t[mask]
        data_y = phi2[mask]
        fit = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)
        growth_rate = fit[0]/2
        if not np.isfinite(qavg): qavg = HEATFLUX_THRESHOLD
        if not np.isfinite(growth_rate): growth_rate = HEATFLUX_THRESHOLD

    except Exception as e:
        pprint(e)
        qavg = HEATFLUX_THRESHOLD
        growth_rate = GROWTHRATE_THRESHOLD

    try: os.remove(os.path.join(OUT_DIR,f'{v.input_file.split("/")[-1]}_{gs2_input_name[-10:]}'))
    except Exception as e: pass
    try: os.remove(os.path.join(OUT_DIR,v.output_file))
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"*{gs2_input_name}*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f".{gs2_input_name}*")): os.remove(objective_file)
    except Exception as e: pass
    
    return growth_rate#qavg
######################################
######################################
######################################
def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return GROWTHRATE_THRESHOLD
    try:
        growth_rate = CalculateGrowthRate(v)
    except Exception as e:
        growth_rate = GROWTHRATE_THRESHOLD
    if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(v, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
    else: qs = QuasisymmetryRatioResidual(v, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
    out_str = f'{datetime.now().strftime("%H:%M:%S")} - Growth rate = {growth_rate:1f}, quasisymmetry = {qs.total():1f} with aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s'
    print(out_str)
    output_dofs_to_csv(v.x,v.mean_iota(),v.aspect(),growth_rate,qs.total())
    return growth_rate
optTurbulence = make_optimizable(TurbulenceCostFunction, vmec)
######################################
try:
    pprint("Initial aspect ratio:", vmec.aspect())
    pprint("Initial mean iota:", vmec.mean_iota())
    pprint("Initial magnetic well:", vmec.vacuum_well())
except Exception as e: pprint(e)
if MPI.COMM_WORLD.rank == 0: pprint("Initial growth rate:", CalculateGrowthRate(vmec))
######################################
initial_dofs=np.copy(surf.x)
def fun(dofss):
    prob.x = dofss
    objective = prob.objective()

    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"input*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"wout*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"gs2-*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f".gs2-*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"grid_gs2-*")): os.remove(objective_file)
    except Exception as e: pass

    return objective
for max_mode in max_modes:
    output_path_parameters=f'output_{optimizer}_maxmode{max_mode}.csv'
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    initial_dofs=np.copy(surf.x)
    dofs=surf.x
    ######################################  
    opt_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight)]
    opt_tuple.append((optTurbulence.J, 0, weight_optTurbulence))
    if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
    else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
    if opt_quasisymmetry: opt_tuple.append((qs.residuals, 0, 1))
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    pprint('## Now calculating total objective function ##')
    if MPI.COMM_WORLD.rank == 0: pprint("Total objective before optimization:", prob.objective())
    pprint('-------------------------')
    pprint(f'Optimizing with max_mode = {max_mode}')
    pprint('-------------------------')
    if optimizer == 'dual_annealing':
        initial_temp = 1000
        visit = 2.0
        bounds = [(-0.25,0.25) for _ in dofs]
        minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds, "options": {'maxiter': MAXITER_LOCAL, 'maxfev': MAXFUN_LOCAL, 'disp': True}}
        if MPI.COMM_WORLD.rank == 0: res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, initial_temp=initial_temp,visit=visit, no_local_search=no_local_search, x0=dofs, minimizer_kwargs=minimizer_kwargs)
    elif optimizer == 'least_squares':
        least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
    else: print('Optimizer not available')
    ######################################
    try: 
        pprint("Final aspect ratio:", vmec.aspect())
        pprint("Final mean iota:", vmec.mean_iota())
        pprint("Final magnetic well:", vmec.vacuum_well())
        pprint("Final quasisymmetry:", qs.total())
        if MPI.COMM_WORLD.rank == 0: pprint("Final growth rate:", CalculateGrowthRate(vmec))
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
##############################################################################
##############################################################################
pprint(f'Whole optimization took {(time.time()-start_time):1f}s')
