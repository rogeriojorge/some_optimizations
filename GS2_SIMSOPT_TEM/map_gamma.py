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

# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/output_MAXITER350_least_squares_nfp2_QA_QA/wout_final.nc'
# output_dir = 'out_map_nfp2_QA_QA_least_squares'
vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/wout_nfp2_QA.nc'
output_dir = 'out_map_nfp2_QA_initial'
phi_GS2 = np.linspace(-10*np.pi, 10*np.pi, 121)
# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/output_MAXITER350_least_squares_nfp4_QH_QH/wout_final.nc'
# output_dir = 'out_map_nfp4_QH_QH_least_squares'
# vmec_file = '/Users/rogeriojorge/local/some_optimizations/GS2_SIMSOPT_TEM/wout_nfp4_QH.nc'
# output_dir = 'out_map_nfp4_QH_initial'
# phi_GS2 = np.linspace(-8*np.pi, 8*np.pi, 51)
s_radius = 0.25
alpha_fieldline = 0
nlambda = 25
nstep = 350
delt = 0.1
LN_array = np.linspace(0.5,6,8)
LT_array = np.linspace(0.5,6,8)
n_processes_parallel = 8
plot_extent_fix = False
plot_min = 0
plot_max = 0.40
########################################
# Go into the output directory
OUT_DIR = os.path.join(this_path,output_dir)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
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
    omega_average_array_omega = omega_average_array[-1,:,0,0]
    omega_average_array_gamma = omega_average_array[-1,:,0,1]
    max_index = np.argmax(omega_average_array_gamma)
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
    plt.savefig(stellFile+'_phi2.png')
    plt.close()
    return GrowthRate, abs(omega)
# Save final eigenfunction
def eigenPlot(stellFile):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    y = f.variables['phi'][()]
    x = f.variables['theta'][()]
    plt.figure(figsize=(7.5,4.0))
    phiR0= y[0,0,int((len(x)-1)/2+1),0]
    phiI0= y[0,0,int((len(x)-1)/2+1),1]
    phi02= phiR0**2+phiI0**2
    phiR = (y[0,0,:,0]*phiR0+y[0,0,:,1]*phiI0)/phi02
    phiI = (y[0,0,:,1]*phiR0-y[0,0,:,0]*phiI0)/phi02
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
def gammabyky(stellFile,fractionToConsider=0.6):
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
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        p = subprocess.Popen(bashCommand.split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)#stdout=fp)
        p.wait()
        file2read = os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc")
        # omega_average = netCDF4.Dataset(file2read,'r').variables['omega_average'][()]
        # growth_rate = np.max(np.array(omega_average)[-1,:,0,1])
        eigenPlot(file2read)
        growth_rate, omega = getgamma(file2read)
        kyX, growthRateX, realFrequencyX = gammabyky(file2read)
    except Exception as e:
        print(e)
        exit()
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
plt.savefig(os.path.join(OUT_DIR,'gs2_scan.pdf'), format='pdf', bbox_inches='tight')
# plt.show()
plt.close()

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