#!/usr/bin/env python3
import os
import glob
import shutil
import netCDF4
import subprocess
import matplotlib
import numpy as np
from time import time
from pathlib import Path
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from joblib import Parallel, delayed
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import to_gs2, vmec_fieldlines
this_path = Path(__file__).parent.resolve()
######## INPUT PARAMETERS ########
gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2'

# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT/output_MAXITER350_dual_annealing_nfp4_QH/see_min/wout_nfp4_QH_000_000000.nc'
# output_dir = 'out_map_nfp4_QH_dual_annealing'
# phi_GS2 = np.linspace(-1*np.pi, 1*np.pi, 131)
vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT/wout_nfp4_QH.nc'
output_dir = 'out_map_nfp4_QH_initial'
phi_GS2 = np.linspace(-2*np.pi, 2*np.pi, 131)
s_radius = 0.25
alpha_fieldline = 0
nlambda = 25
nstep = 300
LN_array = np.linspace(0.5,6,8)
LT_array = np.linspace(0.5,6,8)
n_processes_parallel = 8
########################################
# Go into the output directory
OUT_DIR = os.path.join(this_path,output_dir)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
vmec = Vmec(vmec_file)
# Function to replace text in a file
def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)
# Run GS2
gridout_file = os.path.join(OUT_DIR,f'grid_gs2.out')
to_gs2(gridout_file, vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=nlambda)
growth_rate_array = np.zeros((len(LN_array),len(LT_array)))
fl1 = vmec_fieldlines(vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, plot=True, show=False)
plt.savefig(f'geometry_profiles_s{s_radius}_alpha{alpha_fieldline}.png');plt.close()
def run_gs2(ln, lt):
    start_time_local = time()
    try:
        gs2_input_name = f"gs2Input-LN{ln}-LT{lt}"
        gs2_input_file = os.path.join(OUT_DIR,f'{gs2_input_name}.in')
        shutil.copy(os.path.join(this_path,'gs2Input.in'),gs2_input_file)
        replace(gs2_input_file,' gridout_file = "grid.out"',f' gridout_file = "grid_gs2.out"')
        replace(gs2_input_file,' fprim = 1.0 ! -1/n (dn/drho)',f' fprim = {ln} ! -1/n (dn/drho)')
        replace(gs2_input_file,' tprim = 3.0 ! -1/T (dT/drho)',f' tprim = {lt} ! -1/T (dT/drho)')
        replace(gs2_input_file,' nstep = 100 ! Maximum number of timesteps',f' nstep = {nstep} ! Maximum number of timesteps')
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        p = subprocess.Popen(bashCommand.split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)#stdout=fp)
        p.wait()
        file2read = netCDF4.Dataset(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"),'r')
        omega_average = file2read.variables['omega_average'][()]
    except Exception as e:
        print(e)
    growth_rate = np.max(np.array(omega_average)[-1,:,0,1])
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"*{gs2_input_name}*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f".{gs2_input_name}*")): os.remove(objective_file)
    except Exception as e: pass
    print(f'  LN={ln:1f}, LT={lt:1f}, growth rate={growth_rate:1f} took {(time()-start_time_local):1f}s')
    return growth_rate
print('Starting GS2 scan')
start_time = time()
growth_rate_array = np.reshape(Parallel(n_jobs=n_processes_parallel)(delayed(run_gs2)(ln, lt) for lt in LT_array for ln in LN_array),(len(LT_array),len(LN_array)))
# for i, ln in enumerate(LN_array):
#     for j, lt in enumerate(LT_array):
#         growth_rate_array[i,j]=run_gs2(ln, lt)
print(f'Running GS2 scan took {time()-start_time}s')
## Save growth rates to csv file
print(growth_rate_array)
# Plot
plt.figure(figsize=(6, 6))
plotExtent=[min(LN_array),max(LN_array),min(LT_array),max(LT_array)]
im = plt.imshow(growth_rate_array, cmap='jet', extent=plotExtent, origin='lower', interpolation='hermite')
clb = plt.colorbar(im,fraction=0.046, pad=0.04)
clb.ax.set_title(r'$\gamma$', usetex=True)
plt.xlabel(r'$a/L_n$')
plt.ylabel(r'$a/L_T$')
matplotlib.rc('font', size=16)
plt.savefig(os.path.join(OUT_DIR,'gs2_scan.pdf'), format='pdf', bbox_inches='tight')
# plt.show()
plt.close()