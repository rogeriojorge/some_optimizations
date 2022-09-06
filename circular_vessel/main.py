#!/usr/bin/env python
from subprocess import run
from funcs import (output2regcoil, output2nescoil, importCoils, cartesianCoils2fourier,
                   getFourierCurve, export_coils, plot_stellarator, create_poincare)
import os
from pathlib import Path
from numpy import sqrt, dot, concatenate, sum
import matplotlib.pyplot as plt
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo.curverzfourier import CurveRZFourier
from scipy.io import netcdf_file
from simsopt.geo import curves_to_vtk, SurfaceRZFourier

name = 'preciseQA'
targetValue = 0.036
nCoilsPerNFP = 6
R0_coil= 1.0098736979848868
a_coil= 0.439

nPCoils = 15
accuracy= 50000

f = netcdf_file(f'wout_{name}.nc','r',mmap=False)
nfp = f.variables['nfp'][()]
raxis_cc = f.variables['raxis_cc'][()]
zaxis_cs = f.variables['zaxis_cs'][()]
f.close()

output2regcoil(name,targetValue,R0_coil,a_coil)
run(f"./regcoil regcoil_in.{name}".split())
output2nescoil(f'{name}_nescin.out',f'wout_{name}.nc',R0_coil,a_coil)
run(f"./cutCoilsFromRegcoil regcoil_out.{name}.nc {name}_nescin.out {nCoilsPerNFP} 0 -1".split())

print("Convert resulting coils to SIMSGEO curves")
this_path = Path(__file__).parent.resolve()
coilsCartesian, current  = importCoils("coils."+name)
outputFile      = name+"_coil_coeffs.dat"
cartesianCoils2fourier(coilsCartesian,outputFile,nPCoils,accuracy)
coils, currents = getFourierCurve(outputFile,current)
filename = "coils."+name+"2"
export_coils(coils,filename,currents,nfp)

nquadrature = 101
nfourier = len(raxis_cc)-1
curveFourier = concatenate((raxis_cc,zaxis_cs[1:]))
axis = CurveRZFourier(nquadrature,nfourier,nfp,True)
axis.set_dofs(curveFourier)

_, current  = importCoils("coils."+name)
curves, currents = getFourierCurve(name+"_coil_coeffs.dat",current)
curves_to_vtk(curves, "coils")
print("Look at resulting coils")
plot_stellarator("coils_FOURIER_", name, curves, nfp, axis)
coils = [Coil(curv, Current(curr)) for (curv, curr) in zip(curves, currents)]
bs = BiotSavart(coils)

s = SurfaceRZFourier.from_wout(f'wout_{name}.nc', nphi=150, ntheta=50, range="full torus")
bs.set_points(s.gamma().reshape((-1, 3)))
pointData = {"B_N": sum(bs.B().reshape((150, 50, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("surf_VMEC", extra_data=pointData)

create_poincare(bs = bs, vmec_file = f'wout_{name}.nc', ntheta_VMEC = 300, nfieldlines=5, tmax_fl=600)
bs.set_points(axis.gamma())
Bfield=bs.B()
Bstrength = [sqrt(dot(Bfieldi,Bfieldi)) for Bfieldi in Bfield]
plt.figure()
plt.plot(Bstrength)
plt.xlabel('phi')
plt.ylabel('B')
plt.savefig("Bonaxis"+name+'.pdf', bbox_inches = 'tight', pad_inches = 0)
#plt.show()
