#!/usr/bin/env python3
import os
import numpy as np
import booz_xform as bx
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt

def main(name,wout, nsurfaces=14, figures_folder='figures', vmec_folder='vmec', mpi=None):
	if mpi==None:
		try:
			from mpi4py import MPI
			comm = MPI.COMM_WORLD
		except ImportError:
			comm = None
	else:
		comm = mpi.comm_world

	def pprint(*args, **kwargs):
		if comm == None or comm.rank == 0:
			print(*args, **kwargs)

	if comm == None or comm.rank == 0:
		b1 = bx.Booz_xform()
		b1.read_wout(wout)
		ns_vmec = Vmec(wout).wout.ns
		b1.compute_surfs = np.ceil(np.linspace(0,1,nsurfaces,endpoint=False)*ns_vmec)
		b1.mboz = 120
		b1.nboz = 40
		b1.run()
		b1.write_boozmn(os.path.join(vmec_folder,"boozmn_"+name+".nc"))
		# b1.read_boozmn("boozmn_"+name+".nc")
		pprint("Plot BOOZ_XFORM")
		fig = plt.figure(); bx.surfplot(b1, js=1,  fill=False, ncontours=35)
		plt.savefig(os.path.join(figures_folder, "Boozxform_surfplot_1_"+name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
		fig = plt.figure(); bx.surfplot(b1, js=int(nsurfaces/2), fill=False, ncontours=35)
		plt.savefig(os.path.join(figures_folder, "Boozxform_surfplot_2_"+name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
		fig = plt.figure(); bx.surfplot(b1, js=nsurfaces-1, fill=False, ncontours=35)
		plt.savefig(os.path.join(figures_folder, "Boozxform_surfplot_3_"+name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
		if name[0:2] == 'QH':
			helical_detail = True
		else:
			helical_detail = False
		fig = plt.figure(); bx.symplot(b1, helical_detail = helical_detail, sqrts=True)
		plt.savefig(os.path.join(figures_folder, "Boozxform_symplot_"+name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
		fig = plt.figure(); bx.modeplot(b1, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
		plt.savefig(os.path.join(figures_folder, "Boozxform_modeplot_"+name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()