#!/usr/bin/env python3
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

initial_folder = "/marconi_scratch/userexternal/rjorge00/"
# initial_folder = "/Users/rogeriojorge/local/"

QAinitial = f"{initial_folder}some_optimizations/GS2_SIMSOPT_ITG/nonlinear_nfp2_QA_initial_ln1.0_lt3.0/gs2Input_ln1.0lt3.0.out.nc"
QAfinal = f"{initial_folder}/some_optimizations/GS2_SIMSOPT_ITG/nonlinear_nfp2_QA_final_ln1.0_lt3.0/gs2Input_ln1.0lt3.0.out.nc"
QHinitial = f"{initial_folder}/some_optimizations/GS2_SIMSOPT_ITG/nonlinear_nfp4_QH_initial_ln1.0_lt3.0/gs2Input_ln1.0lt3.0.out.nc"
QHfinal = f"{initial_folder}/some_optimizations/GS2_SIMSOPT_ITG/nonlinear_nfp4_QH_final_ln1.0_lt3.0/gs2Input_ln1.0lt3.0.out.nc"

flux_QA_initial = np.array(netCDF4.Dataset(QAinitial,'r',mmap=False).variables['es_heat_flux'][()])[:-1]
time_QA_initial = np.array(netCDF4.Dataset(QAinitial,'r',mmap=False).variables['t'][()])[:-1]

flux_QA_final   = np.array(netCDF4.Dataset(QAfinal,'r',mmap=False).variables['es_heat_flux'][()])[:-1]
time_QA_final   = np.array(netCDF4.Dataset(QAfinal,'r',mmap=False).variables['t'][()])[:-1]

flux_QH_initial = np.array(netCDF4.Dataset(QHinitial,'r',mmap=False).variables['es_heat_flux'][()])[:-1]
time_QH_initial = np.array(netCDF4.Dataset(QHinitial,'r',mmap=False).variables['t'][()])[:-1]

flux_QH_final   = np.array(netCDF4.Dataset(QHfinal,'r',mmap=False).variables['es_heat_flux'][()])[:-1]
time_QH_final   = np.array(netCDF4.Dataset(QHfinal,'r',mmap=False).variables['t'][()])[:-1]

plt.figure()
plt.plot(time_QA_initial,flux_QA_initial,label='QA initial')
plt.plot(time_QA_final,flux_QA_final,label='QA final')
plt.plot(time_QH_initial,flux_QH_initial,label='QH initial')
plt.plot(time_QH_final,flux_QH_final,label='QH final')

plt.xlabel(r'$t$');plt.ylabel(r'$Q$')
plt.legend(loc="upper right")
plt.tight_layout()
# plt.show()
plt.savefig('Qfluxes.pdf')
plt.close()
