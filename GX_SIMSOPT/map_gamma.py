#!/usr/bin/env python3
import os
import glob
import shutil
import netCDF4
import subprocess
import matplotlib
import numpy as np
import pandas as pd
from time import time, sleep
from pathlib import Path
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from joblib import Parallel, delayed
from simsopt.mhd import Vmec
this_path = Path(__file__).parent.resolve()
######## INPUT PARAMETERS ########
gx_executable = '/m100/home/userexternal/rjorge00/gx/gx'
convert_VMEC_to_GX = '/m100/home/userexternal/rjorge00/gx/geometry_modules/vmec/convert_VMEC_to_GX'

vmec_file = '/m100/home/userexternal/rjorge00/some_optimizations/GX_SIMSOPT/wout_nfp4_QH.nc'
output_dir = 'out_map_nfp4_QH_initial'

nstep = 10000
dt = 0.015
nzgrid = 60
npol = 2
desired_normalized_toroidal_flux = 0.25
alpha_fieldline = 0
nhermite  = 26
nlaguerre = 8
nu_hyper = 1.5
ny = 60

LN_array = np.linspace(0.5,6,2)
LT_array = np.linspace(0.5,6,2)
# n_processes_parallel = 8

plot_extent_fix = False
plot_min = 0
plot_max = 0.40

########################################
# Go into the output directory
OUT_DIR = os.path.join(this_path,output_dir)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
output_csv = os.path.join(OUT_DIR,output_dir+'.csv')
vmec = Vmec(vmec_file)
#### Auxiliary functions
##### Function to obtain gamma and omega for each ky
def gammabyky(stellFile):
    fX   = netCDF4.Dataset(stellFile,'r',mmap=False)
    tX   = fX.variables['time'][()]
    kyX  = fX.variables['ky'][()]
    omega_average_array = np.array(fX.groups['Special']['omega_v_time'][()])
    realFrequencyX = omega_average_array[-1,:,0,0] # only looking at one kx
    growthRateX = omega_average_array[-1,:,0,1] # only looking at one kx
    max_index = np.argmax(growthRateX)
    max_growthrate_gamma = growthRateX[max_index]
    max_growthrate_omega = realFrequencyX[max_index]
    max_growthrate_ky = kyX[max_index]

    numRows = 1
    numCols = 3

    plt.subplot(numRows, numCols, 1)
    plt.plot(kyX,growthRateX,'.-')
    plt.xlabel('ky')
    plt.ylabel('gamma')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)

    plt.subplot(numRows, numCols, 2)
    plt.plot(kyX,realFrequencyX,'.-')
    plt.xlabel('ky')
    plt.ylabel('omega')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)

    plt.subplot(numRows, numCols, 3)
    for count, ky in enumerate(kyX): plt.plot(tX[2:],omega_average_array[2:,count,0,1],'.-', label=f'gamma at ky={ky}')
    plt.xlabel('time')
    plt.ylabel('gamma')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size=12),loc=0)

    plt.tight_layout()
    plt.savefig(stellFile+"_GammaOmegaKy.png")
    plt.close()
    return max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky
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
# Function to create GX gridout and input file
def create_gx_inputs(ln, lt):
    f_wout = vmec_file.split('/')[-1]
    if not os.path.isfile(os.path.join(OUT_DIR,f_wout)):
        shutil.copy(vmec_file,os.path.join(OUT_DIR,f_wout))
    gridout_file = f'grid.gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux}_nt_{2*nzgrid}'
    if not os.path.isfile(os.path.join(OUT_DIR,gridout_file)):
        if ln==LN_array[0] and lt==LT_array[0]:
            shutil.copy(os.path.join(this_path,'gx-geometry-sample.ing'),os.path.join(OUT_DIR,'gx-geometry-sample.ing'))
            replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'nzgrid = 32',f'nzgrid = {nzgrid}')
            replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'npol = 2',f'npol = {npol}')
            replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'desired_normalized_toroidal_flux = 0.12755',f'desired_normalized_toroidal_flux = {desired_normalized_toroidal_flux:.3f}')
            replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'vmec_file = "wout_gx.nc"',f'vmec_file = "{f_wout}"')
            replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'alpha = 0.0"',f'alpha = {alpha_fieldline}')
            shutil.copy(convert_VMEC_to_GX,os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
            p = subprocess.Popen(f"./convert_VMEC_to_GX gx-geometry-sample".split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)
            p.wait()
        else: sleep(0.3)
    fname = f"gxInput_nzgrid{nzgrid}_npol{npol}_nstep{nstep}_dt{dt}_ln{ln}_lt{lt}_nhermite{nhermite}_nlaguerre{nlaguerre}_nu_hyper{nu_hyper}"
    fnamein = os.path.join(OUT_DIR,fname+'.in')
    shutil.copy(os.path.join(this_path,'gx-input.in'),fnamein)
    replace(fnamein,' geofile = "gx_wout.nc"',f' geofile = "gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}_geo.nc"')
    replace(fnamein,' gridout_file = "grid.out"',f' gridout_file = "{gridout_file}"')
    replace(fnamein,' nstep  = 7000',f' nstep  = {nstep}')
    replace(fnamein,' fprim = [ 1.0,       1.0     ]',f' fprim = [ {ln},       {ln}     ]')
    replace(fnamein,' tprim = [ 3.0,       3.0     ]',f' tprim = [ {lt},       {lt}     ]')
    replace(fnamein,' dt = 0.015',f' dt = {dt}')
    replace(fnamein,' ntheta = 80',f' ntheta = {2*nzgrid}')
    replace(fnamein,' nhermite  = 18',f' nhermite = {nhermite}')
    replace(fnamein,' nlaguerre = 6',f' nlaguerre = {nlaguerre}')
    replace(fnamein,' nu_hyper_m = 1.0',f' nu_hyper_m = {nu_hyper}')
    replace(fnamein,' nu_hyper_l = 1.0',f' nu_hyper_l = {nu_hyper}')
    replace(fnamein,' ny = 40',f' ny = {ny}')
    return fname
# Function to remove spurious GX files
def remove_gx_files(gx_input_name):
    for f in glob.glob('*.restart.nc'): remove(f)
    for f in glob.glob('*.log'): remove(f)
    ## REMOVE ALSO INPUT FILE
    for f in glob.glob('*.in'): remove(f)
    ## REMOVE ALSO OUTPUT FILE
    for f in glob.glob('*.out.nc'): remove(f)
# Function to output inputs and growth rates to a CSV file
def output_to_csv(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper, growth_rate, frequency, ky, ln, lt):
    keys=np.concatenate([['ln'],['lt'],['nzgrid'],['npol'],['nstep'],['nhermite'],['nlaguerre'],['dt'],['growth_rate'],['frequency'],['ky'],['nu_hyper']])
    values=np.concatenate([[ln],[lt],[nzgrid],[npol],[nstep],[nhermite],[nlaguerre],[dt],[growth_rate],[frequency],[ky],[nu_hyper]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_csv): pd.DataFrame(columns=df.columns).to_csv(output_csv, index=False)
    df.to_csv(output_csv, mode='a', header=False, index=False)
# Function to run GX and extract growth rate
def run_gx(ln, lt):
    gx_input_name = create_gx_inputs(ln, lt)
    f_log = os.path.join(OUT_DIR,gx_input_name+".log")
    gx_cmd = [f"{gx_executable}", f"{os.path.join(OUT_DIR,gx_input_name+'.in')}", "1"]
    with open(f_log, 'w') as fp:
        p = subprocess.Popen(gx_cmd,stdout=fp)
    p.wait()
    fout = os.path.join(OUT_DIR,gx_input_name+".nc")
    max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky = gammabyky(fout)
    remove_gx_files(gx_input_name)
    output_to_csv(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper, max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, ln, lt)
    return max_growthrate_gamma
print('Starting GX scan')
start_time = time()
# growth_rate_array = np.reshape(Parallel(n_jobs=n_processes_parallel)(delayed(run_gx)(ln, lt) for lt in LT_array for ln in LN_array),(len(LT_array),len(LN_array)))
growth_rate_array = np.zeros((len(LN_array),len(LT_array)))
for i, ln in enumerate(LN_array):
    for j, lt in enumerate(LT_array):
        growth_rate_array[i,j]=run_gx(ln, lt)
print(f'Running GX scan took {time()-start_time}s')
## Save growth rates to csv file
print('growth rates:')
print(growth_rate_array.transpose())
# Plot
plt.figure(figsize=(6, 6))
plotExtent=[min(LN_array),max(LN_array),min(LT_array),max(LT_array)]
im = plt.imshow(growth_rate_array, cmap='jet', extent=plotExtent, origin='lower', interpolation='hermite')
clb = plt.colorbar(im,fraction=0.046, pad=0.04)
clb.ax.set_title(r'$\gamma$', usetex=True)
plt.xlabel(r'$a/L_n$')
plt.ylabel(r'$a/L_T$')
matplotlib.rc('font', size=16)
if plot_extent_fix: plt.clim(plot_min,plot_max) 
plt.savefig(os.path.join(OUT_DIR,'gx_scan.pdf'), format='pdf', bbox_inches='tight')
# plt.show()
plt.close()

f_wout = vmec_file.split('/')[-1]
os.remove(os.path.join(OUT_DIR,f_wout))
os.remove(os.path.join(OUT_DIR,'convert_VMEC_to_GX'))

for f in glob.glob('*.out'): remove(f)
## REMOVE ALSO INPUT FILES
for f in glob.glob('*.in'): remove(f)
## REMOVE ALSO OUTPUT FILES
for f in glob.glob('*.out.nc'): remove(f)