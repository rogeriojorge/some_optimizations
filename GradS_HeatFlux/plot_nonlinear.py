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

file_labels = (['056','061'])
folders = (['nonlinear_056_LN1.0_LT3.0','nonlinear_061_LN1.0_LT3.0'])

file_suffix = 'nzgrid125_npol4_nstep400_dt0.5_ln1.0_lt3.0_nhermite16_nlaguerre8_nu_hyper0.5_D_hyper0.05_ny80_nx130_y016.0.nc'

fig= plt.figure(figsize = (8, 4), dpi = 200);ax=plt.subplot(111)
for i, (label, folder) in enumerate(zip(file_labels,folders)):
    if label=='056':fX = netCDF4.Dataset(os.path.join(this_path,folder,f'gx_20221118-01-056_desc_optimize_QA_magwell_maxmode5_{file_suffix}'),'r',mmap=False)
    else: fX = netCDF4.Dataset(os.path.join(this_path,folder,f'gx_20221118-01-061_desc_optimize_QA_magwell_grad_rho_threshold_2.5_maxmode5_{file_suffix}'),'r',mmap=False)
    time  = np.array(fX.variables['time'][:])
    qflux = np.array(fX.groups['Fluxes'].variables['qflux'][:,0])
    plt.plot(time,qflux,label=label)

plt.xlabel('Time $(v_{ti}/a)$', fontsize=22)
plt.ylabel('$Q_i/Q_{GB}$', fontsize=22)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.legend(fontsize=16, loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(this_path,'GX_heatFluxes.pdf'))
# plt.show()
