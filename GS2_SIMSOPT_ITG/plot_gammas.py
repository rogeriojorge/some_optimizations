#!/usr/bin/env python
import os
import netCDF4
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from quasilinear_estimate import quasilinear_estimate
import warnings
import matplotlib
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()

out_dirs = ([['out_map_nfp2_QA_QA_least_squares',
              'out_map_nfp2_QA_QA_onlyQS'],
             ['out_map_nfp4_QH_QH_least_squares',
              'out_map_nfp4_QH_QH_onlyQS']])
file2read = 'gs2Input-LN1.0-LT3.0.out.nc'

for i, out_dir_qaorqh in enumerate(out_dirs):
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111)
    for j, out_dir in enumerate(out_dir_qaorqh):
        os.chdir(os.path.join(this_path,out_dir))
        f = netCDF4.Dataset(file2read,'r',mmap=False)
        ky  = f.variables['ky'][()]
        # naky = len(ky)
        weighted_growth_rate = quasilinear_estimate(file2read)
        plt.plot(ky,weighted_growth_rate, label=('QA' if i==0 else 'QH')+(' + ITG' if j==0 else ' only'), linewidth=2.0)
        os.chdir(this_path)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.ylabel(r'$\gamma/\langle k_{\perp}^2 \rangle$', fontsize=22)
    plt.xlabel(r'$k_y$', fontsize=22)
    plt.xlim([0,max(ky)])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=18)
    plt.tight_layout()
    fig.savefig('weighted_gamma_'+('QA' if i==0 else 'QH')+'.pdf', dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111)
    for j, out_dir in enumerate(out_dir_qaorqh):
        os.chdir(os.path.join(this_path,out_dir))
        f = netCDF4.Dataset(file2read,'r',mmap=False)
        kyX  = f.variables['ky'][()]
        fractionToConsider = 0.3
        tX   = f.variables['t'][()]
        phi2_by_kyX  = f.variables['phi2_by_ky'][()]
        startIndexX  = int(len(tX)*(1-fractionToConsider))
        growthRateX  = []
        for ii in range(len(kyX)):
            maskX  = np.isfinite(phi2_by_kyX[:,ii])
            data_xX = tX[maskX]
            data_yX = phi2_by_kyX[maskX,ii]
            fitX  = np.polyfit(data_xX[startIndexX:], np.log(data_yX[startIndexX:]), 1)
            thisGrowthRateX  = fitX[0]/2
            growthRateX.append(thisGrowthRateX)
        plt.plot(ky,growthRateX, label=('QA' if i==0 else 'QH')+(' + ITG' if j==0 else ' only'), linewidth=2.0)
        os.chdir(this_path)
    plt.ylabel(r'$\gamma$', fontsize=22)
    plt.xlabel(r'$k_y$', fontsize=22)
    plt.xlim([0,max(ky)])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=18)
    plt.tight_layout()
    fig.savefig('gamma_'+('QA' if i==0 else 'QH')+'.pdf', dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
