#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, Boozer
from simsopt.mhd import QuasisymmetryRatioResidual
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
import booz_xform as bx
#################################
max_mode = 2
QA_or_QH = 'QH'
optimizer = 'dual_annealing'
s_initial = 0.3

plt_opt_res = False
plot_vmec = False
run_simple = True

use_previous_results_if_available = True

nparticles = 1500  # number of particles
tfinal = 1e-2  # seconds
nsamples = 10000  # number of time steps
#################################
if QA_or_QH == 'QA': nfp=2
else: nfp=4
out_dir = f'out_s{s_initial}_NFP{nfp}'
out_csv = out_dir+f'/output_{optimizer}_{QA_or_QH}_maxmode{max_mode}.csv'
df = pd.read_csv(out_csv)
#################################
if plt_opt_res:
    df['aspect-6'] = df.apply(lambda row: np.abs(row.aspect - 7), axis=1)
    df['-iota'] = df.apply(lambda row: -np.abs(row.mean_iota), axis=1)
    df['iota'] = df.apply(lambda row: np.min([np.abs(row.mean_iota),1.5]), axis=1)
    df['normalized_time'] = df.apply(lambda row: np.min([np.max([np.mean(row.eff_time),0]),10]), axis=1)
    df.plot(use_index=True, y=['loss_fraction'])#,'iota'])#,'normalized_time'])
    plt.ylim([0,1.])
    df.plot.scatter(y='normalized_time', x='loss_fraction')
    df.plot.scatter(y='loss_fraction', x='iota')
    plt.show()
#################################
if plot_vmec or run_simple:
    location_min = df['loss_fraction'].nsmallest(3).index[0] # chose the index to see smales, second smallest, etc
    df_min = df.iloc[location_min]
    print('Location of minimum:')
    print(df_min)
    os.chdir(out_dir)
    os.makedirs('see_min', exist_ok=True)
    os.chdir('see_min')
if plot_vmec:
    if os.path.isfile(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc') and use_previous_results_if_available:
        vmec = Vmec(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc')
    else:
        vmec = Vmec(f'../../initial_configs/input.nfp{nfp}_{QA_or_QH}')
        surf = vmec.boundary
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")
        vmec.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
        vmec.indata.niter_array[:3] = [ 1000,  1000, 10000]#,  5000, 10000]
        vmec.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
        if max_mode==1:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)']]
        elif max_mode==2:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)'],
                    df_min['x(8)'],df_min['x(9)'],df_min['x(10)'],df_min['x(11)'],df_min['x(12)'],df_min['x(13)'],df_min['x(14)'],df_min['x(15)'],
                    df_min['x(16)'],df_min['x(17)'],df_min['x(18)'],df_min['x(19)'],df_min['x(20)'],df_min['x(21)'],df_min['x(22)'],df_min['x(23)']]
        else:
            print('Not available with that max_mode yet')
            exit()
        vmec.run()
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
    print("Aspect ratio:", vmec.aspect())
    print("Mean iota:", vmec.mean_iota())
    print("Magnetic well:", vmec.vacuum_well())
    print("Quasisymmetry objective after optimization:", qs.total())
    sys.path.insert(1, '../../')
    print("Plot VMEC result")
    import vmecPlot2
    try: vmecPlot2.main(file=vmec.output_file, name='EP_opt', figures_folder='')
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
    b1.bx.write_boozmn("boozmn_out.nc")
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
#################################
if run_simple:
    if os.path.isfile(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc') and use_previous_results_if_available:
        vmec = Vmec(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc')
    else:
        vmec = Vmec(f'../../initial_configs/input.nfp{nfp}_{QA_or_QH}')
        surf = vmec.boundary
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")
        vmec.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
        vmec.indata.niter_array[:3] = [ 4000, 10000,  4000]#,  5000, 10000]
        vmec.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
        if max_mode==1:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)']]
        elif max_mode==2:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)'],
                    df_min['x(8)'],df_min['x(9)'],df_min['x(10)'],df_min['x(11)'],df_min['x(12)'],df_min['x(13)'],df_min['x(14)'],df_min['x(15)'],
                    df_min['x(16)'],df_min['x(17)'],df_min['x(18)'],df_min['x(19)'],df_min['x(20)'],df_min['x(21)'],df_min['x(22)'],df_min['x(23)']]
        else:
            print('Not available with that max_mode yet')
            exit()
        vmec.run()

    wout_filename = vmec.output_file
    B_scale = 1  # Scale the magnetic field by a factor
    Aminor_scale = 1  # Scale the machine size by a factor
    notrace_passing = 0  # If 1 skip tracing of passing particles

    redux_B = 2
    redux_Aminor = 2
    if QA_or_QH == 'QA':
        B_scale = 8.58 / redux_B
        Aminor_scale = 8.5 / redux_Aminor
        aspect_ratio_target = 6
    else:
        B_scale = 6.55 / redux_B
        Aminor_scale = 12.14 / redux_Aminor
        aspect_ratio_target = 7

    g_field = Simple(wout_filename=wout_filename, B_scale=B_scale, Aminor_scale=Aminor_scale)
    g_particle = ChargedParticleEnsemble(r_initial=s_initial)
    print("Starting particle tracer")
    g_orbits = ParticleEnsembleOrbit_Simple(
        g_particle,
        g_field,
        tfinal=tfinal,
        nparticles=nparticles,
        nsamples=nsamples,
        notrace_passing=notrace_passing,
    )
    print(f"  Final loss fraction = {g_orbits.total_particles_lost}")
    # Plot resulting loss fraction
    g_orbits.plot_loss_fraction(show=False, save=True)