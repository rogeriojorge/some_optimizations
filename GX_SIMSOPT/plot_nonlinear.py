#!/usr/bin/env python3
import os
import netCDF4
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
this_path = Path(__file__).parent.resolve()

file_labels = ['QH initial','QH final']#,'QA initial','QA final']
folders = ['nonlinear_nfp4_QH_initial_LN1.0_LT3.0','nonlinear_nfp4_QH_final_LN1.0_LT3.0']
          #,'nonlinear_nfp2_QA_initial_LN1.0_LT3.0','nonlinear_nfp2_QA_final_LN1.0_LT3.0']

file_suffix = 'nzgrid125_npol4_nstep200000_dt0.2_ln1.0_lt3.0_nhermite16_nlaguerre8_nu_hyper0.5_D_hyper0.05_ny80_nx140_y012.0.nc'

plt.figure()
for i, (label, folder) in enumerate(zip(file_labels,folders)):
    if np.mod(i,2)==0: fX = netCDF4.Dataset(os.path.join(this_path,folder,f'gx_{folder[10:17]}_{file_suffix}'),'r',mmap=False)
    else: fX = netCDF4.Dataset(os.path.join(this_path,folder,f'gx_final_{file_suffix}'),'r',mmap=False)
    time  = np.array(fX.variables['time'][:])
    qflux = np.array(fX.groups['Fluxes'].variables['qflux'][:,0])
    plt.plot(time,qflux,label=label)

plt.xlabel('time')
plt.ylabel(f'Q flux')
plt.legend()
plt.savefig(os.path.join(this_path,'GX_heatFluxes.png'))
# plt.show()
