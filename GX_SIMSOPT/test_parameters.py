#!/usr/bin/env python3
import os
import glob
import shutil
import netCDF4
import subprocess
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from joblib import Parallel, delayed
from simsopt.mhd import Vmec
import matplotlib
matplotlib.use('Agg') 
# from simsopt.turbulence.GX_io import GX_Runner, GX_Output
this_path = Path(__file__).parent.resolve()
######## INPUT PARAMETERS ########
gx_executable = '/m100/home/userexternal/rjorge00/gx/gx'
convert_VMEC_to_GX = '/m100/home/userexternal/rjorge00/gx/geometry_modules/vmec/convert_VMEC_to_GX'
vmec_file = '/m100/home/userexternal/rjorge00/some_optimizations/GX_SIMSOPT/wout_nfp2_QA.nc'
output_dir = 'test_out_nfp2_QA_initial'
##
LN = 6.0
LT = 6.0
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
########################################
# Go into the output directory
OUT_DIR = os.path.join(this_path,output_dir)
output_csv = os.path.join(OUT_DIR,output_dir+'.csv')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
vmec = Vmec(vmec_file)
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
# Function to create GS2 gridout and input file
def create_gx_inputs(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper):
    f_wout = vmec_file.split('/')[-1]
    shutil.copy(vmec_file,os.path.join(OUT_DIR,f_wout))
    #gx = GX_Runner(os.path.join(this_path,"gx-input.in"))
    shutil.copy(os.path.join(this_path,'gx-geometry-sample.ing'),os.path.join(OUT_DIR,'gx-geometry-sample.ing'))
    replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'nzgrid = 32',f'nzgrid = {nzgrid}')
    replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'npol = 2',f'npol = {npol}')
    replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'desired_normalized_toroidal_flux = 0.12755',f'desired_normalized_toroidal_flux = {desired_normalized_toroidal_flux:.3f}')
    replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'vmec_file = "wout_gx.nc"',f'vmec_file = "{f_wout}"')
    replace(os.path.join(OUT_DIR,'gx-geometry-sample.ing'),'alpha = 0.0"',f'alpha = {alpha_fieldline}')
    shutil.copy(convert_VMEC_to_GX,os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
    p = subprocess.Popen(f"./convert_VMEC_to_GX gx-geometry-sample".split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)
    p.wait()
    gridout_file = f'grid.gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux}_nt_{2*nzgrid}'
    os.remove(os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
    fname = f"gxInput_nzgrid{nzgrid}_npol{npol}_nstep{nstep}_dt{dt}_ln{LN}_lt{LT}_nhermite{nhermite}_nlaguerre{nlaguerre}_nu_hyper{nu_hyper}"
    fnamein = os.path.join(OUT_DIR,fname+'.in')
    shutil.copy(os.path.join(this_path,'gx-input.in'),fnamein)
    replace(fnamein,' geofile = "gx_wout.nc"',f' geofile = "gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}_geo.nc"')
    replace(fnamein,' gridout_file = "grid.out"',f' gridout_file = "{gridout_file}"')
    replace(fnamein,' nstep  = 7000',f' nstep  = {nstep}')
    replace(fnamein,' fprim = [ 1.0,       1.0     ]',f' fprim = [ {LN},       {LN}     ]')
    replace(fnamein,' tprim = [ 3.0,       3.0     ]',f' tprim = [ {LT},       {LT}     ]')
    replace(fnamein,' dt = 0.015',f' dt = {dt}')
    replace(fnamein,' ntheta = 80',f' ntheta = {2*nzgrid}')
    replace(fnamein,' nhermite  = 18',f' nhermite = {nhermite}')
    replace(fnamein,' nlaguerre = 6',f' nlaguerre = {nlaguerre}')
    replace(fnamein,' nu_hyper_m = 1.0',f' nu_hyper_m = {nu_hyper}')
    replace(fnamein,' nu_hyper_l = 1.0',f' nu_hyper_l = {nu_hyper}')
    replace(fnamein,' ny = 40',f' ny = {ny}')
    os.remove(os.path.join(OUT_DIR,f_wout))
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
# Function to run GS2 and extract growth rate
def run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper):
    gx_input_name = create_gx_inputs(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
    f_log = os.path.join(OUT_DIR,gx_input_name+".log")
    gx_cmd = [f"{gx_executable}", f"{os.path.join(OUT_DIR,gx_input_name+'.in')}", "1"]
    with open(f_log, 'w') as fp:
        p = subprocess.Popen(gx_cmd,stdout=fp)
    p.wait()
    fout = os.path.join(OUT_DIR,gx_input_name+".nc")
    max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky = gammabyky(fout)
    remove_gx_files(gx_input_name)
    output_to_csv(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper, max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, LN, LT)
    return max_growthrate_gamma
###
### Run GS2
###
print('Starting GS2 runs')
# Default run
start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
# Double nzgrid
nzgrid = 2*nzgrid-1;start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nzgrid = int((nzgrid+1)/2)
# Double npol
nzgrid = 2*nzgrid-1;npol=2*npol;start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nzgrid = int((nzgrid+1)/2);npol=int(npol/2)
# Double nstep
nstep = 2*nstep;start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nstep = int(nstep/2)
# Half dt
nstep = 2*nstep;dt=dt/2;start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nstep = int(nstep/2);dt=dt*2
# Double nhermite
nhermite = 2*nhermite;start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nhermite = int(nhermite/2)
# Double nlaguerre
nlaguerre = 2*nlaguerre;start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nlaguerre = int(nlaguerre/2)
# Half n_hyper
nu_hyper = nu_hyper/2;start_time = time();growth_rate=run_gx(nzgrid, npol, nstep, dt, nhermite, nlaguerre, nu_hyper)
print(f'nzgrid={nzgrid} npol={npol} nstep={nstep} dt={dt} nhermite={nhermite} nlaguerre={nlaguerre} nu_hyper={nu_hyper} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nu_hyper = nu_hyper*2
###
### Plot result
###
df = pd.read_csv(output_csv)
df.plot(use_index=True, y=['growth_rate'])
plt.savefig('growth_rate.png')
