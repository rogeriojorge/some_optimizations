#!/usr/bin/env python3
import os
import glob
import time
import shutil
import subprocess
import vmecPlot2
import numpy as np
import pandas as pd
from mpi4py import MPI
import booz_xform as bx
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
from simsopt.turbulence.GX_io import GX_Runner, GX_Output
from scipy.optimize import dual_annealing
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
def pprint(*args, **kwargs):
    # if MPI.COMM_WORLD.rank == 0:
    print(*args, **kwargs)
############################################################################
#### Input Parameters
############################################################################
MAXITER = 100
max_modes = [1]
initial_config = 'input.nfp2_QA' #'input.nfp4_QH'
plot_result = True
optimizer = 'least_squares'# 'dual_annealing' #'least_squares'
use_previous_results_if_available = False
weight_optTurbulence = 100.0
aspect_ratio_target = 6
diff_rel_step = 1e-5
diff_abs_step = 1e-7
no_local_search = True
output_path_parameters=f'output_{optimizer}.csv'
HEATFLUX_THRESHOLD = 1e18
######################################
######################################
OUT_DIR_APPENDIX=f'output_MAXITER{MAXITER}_{optimizer}_{initial_config[6:]}'
OUT_DIR = os.path.join(this_path, OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
######################################
dest = os.path.join(OUT_DIR,OUT_DIR_APPENDIX+'_previous')
if use_previous_results_if_available and (os.path.isfile(os.path.join(OUT_DIR,'input.final')) or os.path.isfile(os.path.join(dest,'input.final'))):
    # if MPI.COMM_WORLD.rank == 0:
    os.makedirs(dest, exist_ok=True)
    if os.path.isfile(os.path.join(OUT_DIR, 'input.final')) and not os.path.isfile(os.path.join(dest, 'input.final')):
        files = os.listdir(OUT_DIR)
        for f in files:
            shutil.move(os.path.join(OUT_DIR, f), dest)
    # else:
    #     time.sleep(0.5)
    filename = os.path.join(dest, 'input.final')
else:
    filename = os.path.join(this_path, initial_config)
os.chdir(OUT_DIR)
vmec = Vmec(filename, verbose=False) #, mpi=mpi)
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
gx_ran = False
def CalculateHeatFlux(v: Vmec, first_restart=False):
    """
        get wout, 
        make fluxtube, 
        run gx, 
        wait, 
        read output
    """
    try:
        v.run()
        f_wout = v.output_file.split('/')[-1]
        # print(' found', f_wout)

        gx = GX_Runner(os.path.join(this_path,"gx-input.in"))

        shutil.copy(os.path.join(this_path,'gx-geometry-sample.ing'),os.path.join(OUT_DIR,'gx-geometry-sample.ing'))
        shutil.copy(os.path.join(this_path,'convert_VMEC_to_GX'),os.path.join(OUT_DIR,'convert_VMEC_to_GX'))

        gx.make_fluxtube(f_wout)

        # cmd = f"{os.path.join(this_path, 'convert_VMEC_to_GX')} {os.path.join(OUT_DIR,'scan-gx-simsopt-psi-0.50')}"
        # # os.system(cmd)
        # subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        tag = f_wout[5:-3]
        ntheta = gx.inputs['Dimensions']['ntheta']
        f_geo = f"gx_wout_{tag}_psiN_0.500_nt_{ntheta}_geo.nc"
        gx.set_gx_wout(f_geo)

        if (first_restart):
            print(' GX: First restart')
            gx.inputs['Controls']['init_amp'] = 1.0e-5
            gx.inputs['Restart']['restart'] = 'false'

        #slurm_sample = 'batch-gx-stellar.sh'
        #gx.load_slurm( slurm_sample )

        fname = f"GX-{tag}"
        fnamein = os.path.join(OUT_DIR,fname+'.in')
        gx.write(fout=fnamein, skip_overwrite=False)
        #f_slurm = f"{tag}.sh"
        #gx.run_slurm( f_slurm, fname )

        #gx_cmd = f"srun -t 3:00:00 --reservation=gpu2022 --gpus-per-task=1 --ntasks=1 gx {fnamein}"
        #os.system(gx_cmd)

        # use this for salloc
        shutil.copy(os.path.join(this_path,'gx'),os.path.join(OUT_DIR,'gx'))

        ## gx_cmd = ["mpiexec","-n","1", "gx", f"{fnamein}"]
        ## gx_cmd = ["srun","./gx", f"{fnamein}"]
        ## use this for login node
        ## gx_cmd = ["srun", "-t", "1:00:00", #"--reservation=gpu2022",
        ##             "--gpus-per-task=1", "--ntasks=1", "gx", f"{fnamein}"]
        global gx_ran
        if not gx_ran:
            gx_cmd = ["./gx", f"{fnamein}"]
            gx_ran = True
        else:
            gx_cmd = ["./gx", f"{fnamein}", "1"]
        os.remove(os.path.join(OUT_DIR,fname+".nc")) if os.path.exists(os.path.join(OUT_DIR,fname+".nc")) else None
        f_log = os.path.join(OUT_DIR,fname+".log")
        with open(f_log, 'w') as fp:
            p = subprocess.Popen(gx_cmd,stdout=fp)
        # pprint(' *** Waiting for GX ***', flush=True)
        p.wait()
        # pprint(' *** GX finished ***')
        # print(' *** GX finished, waiting 3 more s ***')
        # print( datetime.now().strftime("%H:%M:%S") )
        # os.system("sleep 3")

        # read
        fout = os.path.join(OUT_DIR,fname+".nc")
        gx_out = GX_Output(fout)

        qavg, dqavg = gx_out.exponential_window_estimator()
        # print(f" *** GX non-linear qflux: {qavg} ***")
    except Exception as e:
        pprint(e)
        qavg = HEATFLUX_THRESHOLD

    return qavg
######################################
######################################
######################################
def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return HEATFLUX_THRESHOLD
    try:
        heat_flux = CalculateHeatFlux(v)
    except Exception as e:
        heat_flux = HEATFLUX_THRESHOLD
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
# if MPI.COMM_WORLD.rank == 0:
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
    pprint('## Now calculating total objective function ##')
    pprint("Total objective before optimization:", prob.objective())
    print('reached here')
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
    try: 
        pprint("Final aspect ratio:", vmec.aspect())
        pprint("Final mean iota:", vmec.mean_iota())
        pprint("Final magnetic well:", vmec.vacuum_well())
        heat_flux = CalculateHeatFlux(vmec)
        pprint("Final heat flux:", heat_flux)
    except Exception as e: pprint(e)
    ######################################
# if MPI.COMM_WORLD.rank == 0:
vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
######################################
### PLOT RESULT
######################################
if plot_result:# and MPI.COMM_WORLD.rank==0:
    vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'))#, mpi=mpi)
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
try:
    os.remove(os.path.join(OUT_DIR,"gx"))
    os.remove(os.path.join(OUT_DIR,"convert_VMEC_to_GX"))
except Exception as e:
    pprint(e)

try:
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"grid.gx_wout_{initial_config[6:]}_000_000*")):
        os.remove(objective_file)
    for residuals_file in glob.glob(os.path.join(OUT_DIR,f"gx_wout_{initial_config[6:]}_000_000*")):
        os.remove(residuals_file)
    for jac_file in glob.glob(os.path.join(OUT_DIR,f"GX-{initial_config[6:]}_000_000*")):
        os.remove(jac_file)
    for threed_file in glob.glob(os.path.join(OUT_DIR,f"input.{initial_config[6:]}_000_000*")):
        os.remove(threed_file)
    for threed_file in glob.glob(os.path.join(OUT_DIR,f"wout_{initial_config[6:]}_000_000*")):
        os.remove(threed_file)
except Exception as e:
    pprint(e)
