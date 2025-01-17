#!/usr/bin/env python3
import os
import glob
import shutil
import netCDF4
import subprocess
import matplotlib
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import to_gs2
import matplotlib
matplotlib.use('Agg') 
this_path = Path(__file__).parent.resolve()
######## INPUT PARAMETERS ########
gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2'
# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/output_MAXITER350_least_squares_nfp2_QA_QA/wout_final.nc'
# output_dir = 'test_out_nfp2_QA_QA_least_squares'
vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/wout_nfp2_QA.nc'
output_dir = 'test_out_nfp2_QA_initial'
nphi= 101
nlambda = 25
nperiod = 23
nstep = 170
dt = 0.4
# # vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/output_MAXITER350_least_squares_nfp4_QH_QH/wout_final.nc'
# # output_dir = 'test_out_nfp4_QH_QH_least_squares'
# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/wout_nfp4_QH.nc'
# output_dir = 'test_out_nfp4_QH_initial'
# nphi= 111
# nlambda = 27
# nperiod = 7
# nstep = 170
# dt = 0.4
##
LN = 3.0
LT = 1.0
s_radius = 0.25
alpha_fieldline = 0
ngauss = 3
negrid = 9
naky = 10
aky_min = 0.2
aky_max = 2.0
########################################
# Go into the output directory
OUT_DIR = os.path.join(this_path,f'{output_dir}_ln{LN}_lt{LT}')
output_csv = os.path.join(OUT_DIR,f'{output_dir}_ln{LN}_lt{LT}.csv')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
vmec = Vmec(vmec_file)
## Auxiliary functions
# Get growth rates
def getgamma(stellFile, fractionToConsider=0.35):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    phi2 = np.log(f.variables['phi2'][()])
    t = f.variables['t'][()]
    startIndex = int(len(t)*(1-fractionToConsider))
    mask = np.isfinite(phi2)
    data_x = t[mask]
    data_y = phi2[mask]
    fit = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)
    poly = np.poly1d(fit)
    GrowthRate = fit[0]/2
    omega_average_array = np.array(f.variables['omega_average'][()])
    omega_average_array_omega = omega_average_array[-1,:,0,0]
    omega_average_array_gamma = omega_average_array[-1,:,0,1]
    max_index = np.nanargmax(omega_average_array_gamma)
    gamma = omega_average_array_gamma[max_index]
    omega = omega_average_array_omega[max_index]
    # gamma  = np.mean(f.variables['omega'][()][startIndex:,0,0,1])
    # omega  = np.mean(f.variables['omega'][()][startIndex:,0,0,0])
    #fitRes = np.poly1d(coeffs)
    # if not os.path.exists(stellFile+'_phi2.pdf'):
    plt.figure(figsize=(7.5,4.0))
    ##############
    plt.plot(t, phi2,'.', label=r'data - $\gamma_{GS2} = $'+str(gamma))
    plt.plot(t, poly(t),'-', label=r'fit - $\gamma = $'+str(GrowthRate))
    ##############
    plt.legend(loc=0,fontsize=14)
    plt.xlabel(r'$t$');plt.ylabel(r'$\ln |\hat \phi|^2$')
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
    plt.savefig(stellFile+'_phi2.pdf', format='pdf')
    plt.close()
    return GrowthRate, abs(omega)
# Save final eigenfunction
def eigenPlot(stellFile):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    y = f.variables['phi'][()]
    x = f.variables['theta'][()]
    plt.figure(figsize=(7.5,4.0))
    omega_average_array = np.array(f.variables['omega_average'][()])
    omega_average_array_gamma = omega_average_array[-1,:,0,1]
    max_index = np.nanargmax(omega_average_array_gamma)
    phiR0= y[max_index,0,int((len(x))/2),0]
    phiI0= y[max_index,0,int((len(x))/2),1]
    phi02= phiR0**2+phiI0**2
    phiR = (y[max_index,0,:,0]*phiR0+y[max_index,0,:,1]*phiI0)/phi02
    phiI = (y[max_index,0,:,1]*phiR0-y[max_index,0,:,0]*phiI0)/phi02
    ##############
    plt.plot(x, phiR, label=r'Re($\hat \phi/\hat \phi_0$)')
    plt.plot(x, phiI, label=r'Im($\hat \phi/\hat \phi_0$)')
    ##############
    plt.xlabel(r'$\theta$');plt.ylabel(r'$\hat \phi$')
    plt.legend(loc="upper right");#plt.yscale('log')
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.93)
    plt.savefig(stellFile+'_eigenphi.pdf', format='pdf')
    plt.close()
    return 0
##### Function to obtain gamma and omega for each ky
def gammabyky(stellFile,fractionToConsider=0.3):
    # Compute growth rate:
    fX   = netCDF4.Dataset(stellFile,'r',mmap=False)
    tX   = fX.variables['t'][()]
    kyX  = fX.variables['ky'][()]
    phi2_by_kyX  = fX.variables['phi2_by_ky'][()]
    omegaX  = fX.variables['omega'][()]
    startIndexX  = int(len(tX)*(1-fractionToConsider))
    growthRateX  = []
    ## assume that kyX=kyNA
    for i in range(len(kyX)):
        maskX  = np.isfinite(phi2_by_kyX[:,i])
        data_xX = tX[maskX]
        data_yX = phi2_by_kyX[maskX,i]
        fitX  = np.polyfit(data_xX[startIndexX:], np.log(data_yX[startIndexX:]), 1)
        thisGrowthRateX  = fitX[0]/2
        growthRateX.append(thisGrowthRateX)
    # Compute real frequency:
    realFreqVsTimeX  = []
    realFrequencyX   = []
    for i in range(len(kyX)):
        realFreqVsTimeX.append(omegaX[:,i,0,0])
        realFrequencyX.append(np.mean(realFreqVsTimeX[i][startIndexX:]))
    numRows = 1
    numCols = 2

    plt.subplot(numRows, numCols, 1)
    plt.plot(kyX,growthRateX,'.-')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\gamma$')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size='xx-small'),loc=0)

    plt.subplot(numRows, numCols, 2)
    plt.plot(kyX,realFrequencyX,'.-')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\omega$')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size=12),loc=0)

    plt.tight_layout()
    #plt.subplots_adjust(left=0.14, bottom=0.15, right=0.98, top=0.96)
    plt.savefig(stellFile+"_GammaOmegaKy.pdf", format='pdf')
    plt.close()
    return kyX, growthRateX, realFrequencyX
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
def create_gs2_inputs(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max):
    gridout_file = os.path.join(OUT_DIR,f'grid_gs2_nphi{nphi}_nperiod{nperiod}.out')
    phi_GS2 = np.linspace(-nperiod*np.pi, nperiod*np.pi, nphi)
    to_gs2(gridout_file, vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=nlambda)
    gs2_input_name = f"gs2Input_nphi{nphi}_nperiod{nperiod}_nlambda{nlambda}_negrid{negrid}_ngauss{ngauss}_nstep{nstep}_naky{naky}_aky_min{aky_min}_aky_max{aky_max}_dt{dt}_ln{LN}_lt{LT}"
    gs2_input_file = os.path.join(OUT_DIR,f'{gs2_input_name}.in')
    shutil.copy(os.path.join(this_path,'gs2Input.in'),gs2_input_file)
    replace(gs2_input_file,' gridout_file = "grid.out"',f' gridout_file = "{gridout_file}"')
    replace(gs2_input_file,' nstep = 150 ! Maximum number of timesteps',f' nstep = {nstep} ! Maximum number of timesteps"')
    replace(gs2_input_file,' fprim = 1.0 ! -1/n (dn/drho)',f' fprim = {LN} ! -1/n (dn/drho)')
    replace(gs2_input_file,' tprim = 3.0 ! -1/T (dT/drho)',f' tprim = {LT} ! -1/T (dT/drho)')
    replace(gs2_input_file,' delt = 0.4 ! Time step',f' delt = {dt} ! Time step')
    replace(gs2_input_file,' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.',
    f' ngauss = {ngauss} ! Number of untrapped pitch-angles moving in one direction along field line.')
    replace(gs2_input_file,' negrid = 10 ! Total number of energy grid points',
    f' negrid = {negrid} ! Total number of energy grid points')
    replace(gs2_input_file,' naky = 6',f' naky = {naky}')
    replace(gs2_input_file,' aky_min = 0.3',f' aky_min = {aky_min}')
    replace(gs2_input_file,' aky_max = 10.0',f' aky_max = {aky_max}')
    return gs2_input_name
# Function to remove spurious GS2 files
def remove_gs2_files(gs2_input_name):
    for f in glob.glob('*.amoments'): remove(f)
    for f in glob.glob('*.eigenfunc'): remove(f)
    for f in glob.glob('*.error'): remove(f)
    for f in glob.glob('*.fields'): remove(f)
    for f in glob.glob('*.g'): remove(f)
    for f in glob.glob('*.lpc'): remove(f)
    for f in glob.glob('*.mom2'): remove(f)
    for f in glob.glob('*.moments'): remove(f)
    for f in glob.glob('*.vres'): remove(f)
    for f in glob.glob('*.vres2'): remove(f)
    for f in glob.glob('*.exit_reason'): remove(f)
    for f in glob.glob('*.optim'): remove(f)
    for f in glob.glob('*.out'): remove(f)
    for f in glob.glob('*.used_inputs.in'): remove(f)
    for f in glob.glob('*.vspace_integration_error'): remove(f)
    ## REMOVE ALSO OUTPUT FILE
    for f in glob.glob('*.out.nc'): remove(f)
# Function to output inputs and growth rates to a CSV file
def output_to_csv(nphi, nperiod, nlambda, nstep, growth_rate, negrid, ngauss, naky, aky_min, aky_max, dt, ln, lt):
    keys=np.concatenate([['ln'],['lt'],['nphi'],['nperiod'],['nlambda'],['nstep'],['dt'],['growth_rate'],['negrid'],['naky'],['aky_min'],['aky_max'],['ngauss']])
    values=np.concatenate([[ln],[lt],[nphi],[nperiod],[nlambda],[nstep],[dt],[growth_rate],[negrid],[naky],[aky_min],[aky_max],[ngauss]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_csv): pd.DataFrame(columns=df.columns).to_csv(output_csv, index=False)
    df.to_csv(output_csv, mode='a', header=False, index=False)
# Function to run GS2 and extract growth rate
def run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max):
    gs2_input_name = create_gs2_inputs(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
    p = subprocess.Popen(f"{gs2_executable} {gs2_input_name}.in".split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)
    p.wait()
    eigenPlot(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"))
    growth_rate, omega = getgamma(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"))
    kyX, growthRateX, realFrequencyX = gammabyky(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"))
    # growth_rate = np.max(np.array(netCDF4.Dataset(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"),'r').variables['omega_average'][()])[-1,:,0,1])
    remove_gs2_files(gs2_input_name)
    output_to_csv(nphi, nperiod, nlambda, nstep, growth_rate, negrid, ngauss, naky, aky_min, aky_max, dt, LN, LT)
    return growth_rate
###
### Run GS2
###
print('Starting GS2 runs')
# Default run
start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
# Double nphi
nphi = 2*nphi-1;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nphi = int((nphi+1)/2)
# Double nperiod
nperiod = int(2*nperiod);nphi = 2*nphi-1;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nperiod = int(nperiod/2);nphi = int((nphi+1)/2)
# Double nlambda
nlambda = 2*nlambda;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nlambda = int(nlambda/2)
# Double nstep
nstep = 2*nstep;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
nstep = int(nstep/2)
# Half dt
dt = dt/2;nstep = 2*nstep;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
dt = dt*2;nstep = int(nstep/2)
# # Double negrid
# negrid = 2*negrid;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
# print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
# negrid = int(negrid/2)
# # Double ngauss
# ngauss = 2*ngauss;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
# print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
# ngauss = int(ngauss/2)
# Double naky
naky = 2*naky;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
naky = int(naky/2)
# Half kymin
aky_min = aky_min/2;naky = 2*naky;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
aky_min = aky_min*2;naky = int(naky/2)
# Double kymax
aky_max = 2*aky_max;naky = 2*naky;start_time = time();growth_rate=run_gs2(nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, naky, aky_min, aky_max)
print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} naky={naky} aky_min={aky_min} aky_max{aky_max} growth_rate={growth_rate:1f} took {(time()-start_time):1f}s')
aky_max = aky_max/2;naky = int(naky/2)
###
### Plot result
###
df = pd.read_csv(output_csv)
df.plot(use_index=True, y=['growth_rate'])
plt.savefig(os.path.join(OUT_DIR,'growth_rate.png'))