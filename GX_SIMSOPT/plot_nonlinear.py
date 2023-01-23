#!/usr/bin/env python3
import os
import netCDF4
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
this_path = Path(__file__).parent.resolve()
import matplotlib
matplotlib.use('Agg') 
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)

options = ['QH']

for option in options:
    try:
        if option == 'QA':
            file_labels = (['$\omega_{f_{Q}}=0$','$\omega_{f_{Q}}=10$'])
            folders = (['nonlinear_nfp2_QA_initial_LN1.0_LT3.0','nonlinear_nfp2_QA_final_LN1.0_LT3.0'])
        elif option == 'QH':
            file_labels = (['$\omega_{f_{Q}}=0$','$\omega_{f_{Q}}=10$'])
            folders = (['nonlinear_nfp4_QH_initial_LN1.0_LT3.0','nonlinear_nfp4_QH_final_LN1.0_LT3.0'])

        file_suffix = 'nzgrid125_npol4_nstep400_dt0.1_ln1.0_lt3.0_nhermite16_nlaguerre8_nu_hyper0.5_D_hyper0.05_ny90_nx140_y025.0.nc'

        fig = plt.figure(figsize=(8, 6), dpi=200);ax=plt.subplot(111)
        for i, (label, folder) in enumerate(zip(file_labels,folders)):
            # if np.mod(i,2)==0: fX = netCDF4.Dataset(os.path.join(this_path,folder,f'gx_{folder[10:17]}_{file_suffix}'),'r',mmap=False)
            fX = netCDF4.Dataset(os.path.join(this_path,folder,f'gx_final_{file_suffix}'),'r',mmap=False)
            time  = np.array(fX.variables['time'][:])
            qflux = np.array(fX.groups['Fluxes'].variables['qflux'][:,0])
            plt.plot(time,qflux,label=label)

        plt.xlabel('Time $(v_{ti}/a)$', fontsize=22)
        plt.ylabel('$Q_i/Q_{GB}$', fontsize=22)
        # if option == 'QA':   plt.ylim([0,20])
        # elif option == 'QH': plt.ylim([0,8])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        plt.legend(fontsize=16, loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(this_path,f'GX_{option}_heatFluxes.pdf'))
        # plt.show()
    except Exception as e: print(e)