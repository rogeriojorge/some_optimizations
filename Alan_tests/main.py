#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, Boozer
from simsopt.mhd import QuasisymmetryRatioResidual
import booz_xform as bx

vmec = Vmec(f'wout_nfp4_QH_000_000000.nc')


qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
print("Aspect ratio:", vmec.aspect())
print("Mean iota:", vmec.mean_iota())
print("Magnetic well:", vmec.vacuum_well())
print("Quasisymmetry objective:", qs.total())
print("Plot VMEC result")
import vmecPlot2
try: vmecPlot2.main(file=vmec.output_file, name='Alan_opt', figures_folder='')
except Exception as e: print(e)
print('Creating Boozer class for vmec_final')
b1 = Boozer(vmec, mpol=64, ntor=64)
boozxform_nsurfaces=10
print('Defining surfaces where to compute Boozer coordinates')
booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
print(f' booz_surfaces={booz_surfaces}')
b1.register(booz_surfaces)
print('Running BOOZ_XFORM')
b1.run()
b1.bx.write_boozmn("boozmn_alan.nc")
print("Plot BOOZ_XFORM")
fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
plt.savefig("Boozxform_surfplot_1.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
plt.savefig("Boozxform_surfplot_2.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
plt.savefig("Boozxform_surfplot_3.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
plt.savefig("Boozxform_symplot.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
plt.savefig("Boozxform_modeplot.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()