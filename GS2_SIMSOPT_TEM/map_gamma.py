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
from simsopt.mhd.vmec_diagnostics import to_gs2, vmec_fieldlines
import matplotlib
matplotlib.use('Agg') 
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()
######## INPUT PARAMETERS ########
gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2'

# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/output_MAXITER350_least_squares_nfp2_QA_QA_ln3_lt3/wout_final.nc'
# output_dir = 'out_map_nfp2_QA_QA_least_squares_ln3_lt3'
# # vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/wout_nfp2_QA.nc'
# # output_dir = 'out_map_nfp2_QA_initial'
# phi_GS2 = np.linspace(-23*np.pi, 23*np.pi, 101)
# nlambda = 25
# nstep = 170
# dt = 0.4

# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/output_MAXITER350_least_squares_nfp4_QH_QH_ln3_lt3/wout_final.nc'
# output_dir = 'out_map_nfp4_QH_QH_least_squares_ln3_lt3'
vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/wout_nfp4_QH.nc'
output_dir = 'out_map_nfp4_QH_initial'
phi_GS2 = np.linspace(-7*np.pi, 7*np.pi, 111)
nlambda = 27
nstep = 170
dt = 0.4

s_radius = 0.25
alpha_fieldline = 0
LN_array = np.linspace(0.5,6,12)
LT_array = np.linspace(0.5,6,12)
n_processes_parallel = 4
plot_extent_fix = True
plot_min = 0
plot_max = 0.9
ngauss = 3
negrid = 9
naky = 10
aky_min = 0.2
aky_max = 2.0
########################################
# Go into the output directory
OUT_DIR = os.path.join(this_path,output_dir)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
output_csv = os.path.join(OUT_DIR,f'{output_dir}.csv')
vmec = Vmec(vmec_file)
#### Auxiliary functions
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
    omega_average_array_omega = np.mean(omega_average_array[startIndex:,:,0,0],axis=0)
    omega_average_array_gamma = np.mean(omega_average_array[startIndex:,:,0,1],axis=0)
    max_index = np.nanargmax(omega_average_array_gamma)
    gamma = omega_average_array_gamma[max_index]
    omega = omega_average_array_omega[max_index]
    kyX  = f.variables['ky'][()]
    ky_max = kyX[max_index]
    # gamma  = np.mean(f.variables['omega'][()][startIndex:,0,0,1])
    # omega  = np.mean(f.variables['omega'][()][startIndex:,0,0,0])
    #fitRes = np.poly1d(coeffs)
    # if not os.path.exists(stellFile+'_phi2.pdf'):
    plt.figure()
    ##############
    plt.plot(t, phi2,'.', label=r'data - $\gamma_{GS2} = $'+str(gamma))
    plt.plot(t, poly(t),'-', label=r'fit - $\gamma = $'+str(GrowthRate))
    ##############
    plt.legend(loc=0,fontsize=14)
    plt.xlabel(r'$t$');plt.ylabel(r'$\ln |\hat \phi|^2$')
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
    plt.savefig(stellFile+'_phi2.png')
    plt.close()
    return GrowthRate, omega, ky_max
# Save final eigenfunction
def eigenPlot(stellFile):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    y = f.variables['phi'][()]
    x = f.variables['theta'][()]
    plt.figure()
    omega_average_array = np.array(f.variables['omega_average'][()])
    fractionToConsider=0.4
    tX   = f.variables['t'][()]
    startIndexX  = int(len(tX)*(1-fractionToConsider))
    omega_average_array_gamma = np.mean(omega_average_array[startIndexX:,:,0,1],axis=0)
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
    plt.legend(loc="upper right")
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.93)
    plt.savefig(stellFile+'_eigenphi.png')
    plt.close()
    return 0
##### Function to obtain gamma and omega for each ky
def gammabyky(stellFile,fractionToConsider=0.4):
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
    plt.savefig(stellFile+"_GammaOmegaKy.png")
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
def output_to_csv(growth_rate, omega, ky, ln, lt):
    keys=np.concatenate([['ln'],['lt'],['growth_rate'],['omega'],['ky']])
    values=np.concatenate([[ln],[lt],[growth_rate],[omega],[ky]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_csv): pd.DataFrame(columns=df.columns).to_csv(output_csv, index=False)
    df.to_csv(output_csv, mode='a', header=False, index=False)
# Run GS2
gridout_file = os.path.join(OUT_DIR,f'grid_gs2.out')
to_gs2(gridout_file, vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=nlambda)
growth_rate_array = np.zeros((len(LN_array),len(LT_array)))
fl1 = vmec_fieldlines(vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, plot=True, show=False)
plt.savefig(f'geometry_profiles_s{s_radius}_alpha{alpha_fieldline}.png');plt.close()
def run_gs2(ln, lt):
    start_time_local = time()
    try:
        gs2_input_name = f"gs2Input-LN{ln:.1f}-LT{lt:.1f}"
        gs2_input_file = os.path.join(OUT_DIR,f'{gs2_input_name}.in')
        shutil.copy(os.path.join(this_path,'gs2Input.in'),gs2_input_file)
        replace(gs2_input_file,' gridout_file = "grid.out"',f' gridout_file = "grid_gs2.out"')
        replace(gs2_input_file,' fprim = 1.0 ! -1/n (dn/drho)',f' fprim = {ln} ! -1/n (dn/drho)')
        replace(gs2_input_file,' tprim = 3.0 ! -1/T (dT/drho)',f' tprim = {lt} ! -1/T (dT/drho)')
        replace(gs2_input_file,' nstep = 150 ! Maximum number of timesteps',f' nstep = {nstep} ! Maximum number of timesteps')
        replace(gs2_input_file,' delt = 0.4 ! Time step',f' delt = {dt} ! Time step')
        replace(gs2_input_file,' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.',
        f' ngauss = {ngauss} ! Number of untrapped pitch-angles moving in one direction along field line.')
        replace(gs2_input_file,' negrid = 10 ! Total number of energy grid points',
        f' negrid = {negrid} ! Total number of energy grid points')
        replace(gs2_input_file,' naky = 6',f' naky = {naky}')
        replace(gs2_input_file,' aky_min = 0.3',f' aky_min = {aky_min}')
        replace(gs2_input_file,' aky_max = 10.0',f' aky_max = {aky_max}')
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        p = subprocess.Popen(bashCommand.split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)#stdout=fp)
        p.wait()
        file2read = os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc")
        # omega_average = netCDF4.Dataset(file2read,'r').variables['omega_average'][()]
        # growth_rate = np.max(np.array(omega_average)[-1,:,0,1])
        eigenPlot(file2read)
        growth_rate, omega, ky = getgamma(file2read)
        kyX, growthRateX, realFrequencyX = gammabyky(file2read)
        output_to_csv(growth_rate, omega, ky, ln, lt)
    except Exception as e:
        print(e)
        exit()
    print(f'  LN={ln:1f}, LT={lt:1f}, growth rate={growth_rate:1f}, omega={omega:1f}, ky={ky:1f} took {(time()-start_time_local):1f}s')
    return growth_rate, omega, ky
print('Starting GS2 scan')
start_time = time()
growth_rate_array_temp, omega_array_temp, ky_array_temp = np.array(Parallel(n_jobs=n_processes_parallel)(delayed(run_gs2)(ln, lt) for lt in LT_array for ln in LN_array)).transpose()
growth_rate_array = np.reshape(growth_rate_array_temp, (len(LT_array),len(LN_array)))
omega_array = np.reshape(omega_array_temp, (len(LT_array),len(LN_array)))
ky_array = np.reshape(ky_array_temp, (len(LT_array),len(LN_array)))
# for i, ln in enumerate(LN_array):
#     for j, lt in enumerate(LT_array):
#         growth_rate_array[i,j]=run_gs2(ln, lt)
print(f'Running GS2 scan took {time()-start_time}s')
## Save growth rates to csv file
print('growth rates:')
print(growth_rate_array.transpose())
# Plot
plotExtent=[0*min(LN_array),max(LN_array),0*min(LT_array),max(LT_array)]

fig=plt.figure();ax=plt.subplot(111);fig.set_size_inches(4.5, 4.5)
im = plt.imshow(growth_rate_array, cmap='jet', extent=plotExtent, origin='lower', interpolation='hermite')
clb = plt.colorbar(im,fraction=0.046, pad=0.04);clb.ax.set_title(r'$\gamma$', usetex=True)
plt.xlabel(r'$a/L_n$', fontsize=16);plt.ylabel(r'$a/L_T$', fontsize=16);matplotlib.rc('font', size=20)
if plot_extent_fix: plt.clim(plot_min,plot_max) 
plt.gca().set_aspect('equal')
ax.tick_params(axis='x', labelsize=14);ax.tick_params(axis='y', labelsize=14);plt.tight_layout();
plt.savefig(os.path.join(OUT_DIR,'gs2_scan_gamma.pdf'), format='pdf', bbox_inches='tight')

fig=plt.figure();ax=plt.subplot(111);fig.set_size_inches(4.5, 4.5)
im = plt.imshow(omega_array, cmap='jet', extent=plotExtent, origin='lower', interpolation='hermite')
clb = plt.colorbar(im,fraction=0.046, pad=0.04);clb.ax.set_title(r'$\omega$', usetex=True)
plt.xlabel(r'$a/L_n$', fontsize=16);plt.ylabel(r'$a/L_T$', fontsize=16);#matplotlib.rc('font', size=20)
plt.gca().set_aspect('equal')
ax.tick_params(axis='x', labelsize=14);ax.tick_params(axis='y', labelsize=14);plt.tight_layout();
plt.savefig(os.path.join(OUT_DIR,'gs2_scan_omega.pdf'), format='pdf', bbox_inches='tight')

fig=plt.figure();ax=plt.subplot(111);fig.set_size_inches(4.5, 4.5)
im = plt.imshow(ky_array, cmap='jet', extent=plotExtent, origin='lower', interpolation='hermite')
clb = plt.colorbar(im,fraction=0.046, pad=0.04);clb.ax.set_title(r'$k_y$', usetex=True)
plt.xlabel(r'$a/L_n$', fontsize=16);plt.ylabel(r'$a/L_T$', fontsize=16);#matplotlib.rc('font', size=20)
plt.gca().set_aspect('equal')
ax.tick_params(axis='x', labelsize=14);ax.tick_params(axis='y', labelsize=14);plt.tight_layout();
plt.savefig(os.path.join(OUT_DIR,'gs2_scan_ky.pdf'), format='pdf', bbox_inches='tight')

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
for f in glob.glob('*.scratch'): remove(f)
for f in glob.glob('*.used_inputs.in'): remove(f)
for f in glob.glob('*.vspace_integration_error'): remove(f)
## THIS SHOULD ONLY REMOVE FILES STARTING WTH .gs2
for f in glob.glob('.gs2*'): remove(f)
## REMOVE ALSO INPUT FILES
# for f in glob.glob('*.in'): remove(f)
## REMOVE ALSO OUTPUT FILES
for f in glob.glob('*.out.nc'): remove(f)