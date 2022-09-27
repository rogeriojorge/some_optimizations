#!/usr/bin/env python
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
this_path = Path(__file__).parent.resolve()
from simsopt.mhd import Vmec
from simsopt.mhd import QuasisymmetryRatioResidual
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple

points1 = np.linspace(-0.2,0.2,120)
points2 = np.linspace(-0.03,0.03,6)

os.chdir(os.path.join(this_path,'test_optimization'))
vmec = Vmec(os.path.join(this_path,'initial_configs','input.nfp4_QH'), verbose=False)
qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
surf.fix("rc(0,0)")
loss_fraction_array = []
aspect_ratio_array = []
iota_array = []
quasisymmetry_array = []
magnetic_well_array = []
for point1 in points1:
    # for point2 in points2:
    start_time = time.time()
    # print(vmec.x)
    # vmec.x=[point1 if count==1 else point2 if count==3 else vx for count,vx in enumerate(vmec.x)]
    vmec.x=[point1 if count==0 else vx for count,vx in enumerate(vmec.x)]
    # print(vmec.x)
    vmec.run()
    g_particle = ChargedParticleEnsemble(r_initial=0.3)
    g_field = Simple(wout_filename=vmec.output_file, B_scale=5.7/vmec.wout.b0/2, Aminor_scale=1.7/vmec.wout.Aminor_p/2)
    g_orbits = ParticleEnsembleOrbit_Simple(g_particle,g_field,tfinal=5e-4,nparticles=2400,nsamples=3000)
    loss_fraction = g_orbits.total_particles_lost
    print(f'Loss fraction {loss_fraction:1f} for point {point1:1f}, aspect {np.abs(vmec.aspect()):1f} and iota {(vmec.mean_iota()):1f} took {(time.time()-start_time):1f}s')
    loss_fraction_array.append(loss_fraction)
    aspect_ratio_array.append(vmec.aspect())
    iota_array.append(np.abs(vmec.mean_iota()))
    quasisymmetry_array.append(qs.total())
    magnetic_well_array.append(vmec.vacuum_well())
fig = plt.figure();plt.plot(points1, loss_fraction_array, label='Loss fraction')
plt.ylabel('Loss fraction');plt.xlabel('RBC(1,0)');plt.savefig('loss_fraction_over_opt.pdf')
fig = plt.figure();plt.plot(points1, aspect_ratio_array, label='Aspect ratio')
plt.ylabel('Aspect ratio');plt.xlabel('RBC(1,0)');plt.savefig('aspect_ratio_over_opt.pdf')
fig = plt.figure();plt.plot(points1, iota_array, label='Rotational Transform (1/q)')
plt.ylabel('Rotational Transform (1/q)');plt.xlabel('RBC(1,0)');plt.savefig('iota_over_opt.pdf')
fig = plt.figure();plt.plot(points1, quasisymmetry_array, label='Quasisymmetry cost function')
plt.ylabel('Quasisymmetry cost function');plt.xlabel('RBC(1,0)');plt.savefig('quasisymmetry_over_opt.pdf')
fig = plt.figure();plt.plot(points1, magnetic_well_array, label='Magnetic well')
plt.ylabel('Magnetic well');plt.xlabel('RBC(1,0)');plt.savefig('magnetic_well_over_opt.pdf')
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_trisurf(points1, points2, np.array(loss_fraction_array))
plt.show()
