#!/usr/bin/env python
import os
import sys
import shutil
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, Boozer
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
import booz_xform as bx
#################################
max_mode = 4
QA_or_QH = 'QA'
optimizer = 'least_squares'#'dual_annealing' #'least_squares'
quasisymmetry = True
opt_turbulence = True

plt_opt_res = True
plot_vmec = True
run_simple = True
plot_loss_fractions = True
plot_neo = True

MAXITER=350

if QA_or_QH == 'QA':
    nfp=2
    aspect_target = 6.0
else:
    nfp=4
    aspect_target = 8.0

if quasisymmetry: quasisymmetry_weight=1e-3
else: quasisymmetry_weight=1e-8


aspect_weight = 1e0
mirror_weight = 1e1
use_final = True
use_previous_results_if_available = False

nparticles = 2000  # number of particles
tfinal = 1e-2  # seconds
nsamples = 20000  # number of time steps
#################################
# if QA_or_QH == 'QA': nfp=2
# elif QA_or_QH == 'QH': nfp=4
# elif QA_or_QH == 'QI': nfp=3
out_dir = f'output_MAXITER{MAXITER}_{optimizer}_nfp{nfp}_{QA_or_QH}'
if quasisymmetry: out_dir+=f'_{QA_or_QH}'
if not opt_turbulence: out_dir+=f'_onlyQS'
out_csv = out_dir+f'/output_{optimizer}_maxmode{max_mode}.csv'
if opt_turbulence:
    df = pd.read_csv(out_csv)
    if optimizer=='dual_annealing':
        location_min = (df['growth_rate']**2+(quasisymmetry_weight*df['quasisymmetry_total'])**2+aspect_weight*(df['aspect']-aspect_target)**2+mirror_weight*(df['mirror_ratio'])**2).nsmallest(3).index[0]#len(df.index)-1#df['growth_rate'].nsmallest(3).index[0] # chose the index to see smalest, second smallest, etc
    else:
        location_min = ([1 if opt_turbulence else 0]*df['growth_rate']**2+[1 if quasisymmetry else 0]*(quasisymmetry_weight*df['quasisymmetry_total'])**2+aspect_weight*(df['aspect']-aspect_target)**2).nsmallest(3).index[0]#len(df.index)-1#df['growth_rate'].nsmallest(3).index[0] # chose the index to see smalest, second smallest, etc
    #################################
    if plt_opt_res:
        df['aspect-aspect_target'] = df.apply(lambda row: np.abs(row.aspect - aspect_target), axis=1)
        df['-iota'] = df.apply(lambda row: -np.abs(row.mean_iota), axis=1)
        df['iota'] = df.apply(lambda row: np.min([np.abs(row.mean_iota),2.5]), axis=1)
        df['iota'] = df[df['iota']!=1.5]['iota']
        df['growth_rate'] = df[df['growth_rate']<1e17]['growth_rate']
        # df['quasisymmetry_total'] = df[df['quasisymmetry_total']<1e8]['quasisymmetry_total']
        df.plot(use_index=True, y=['growth_rate'])#,'iota'])#,'normalized_time'])
        plt.yscale('log')
        # plt.ylim([0,1.])
        plt.axvline(x = location_min, color = 'b', label = 'minimum Q')
        plt.legend()
        plt.savefig(out_dir+'/growth_rate_over_opt.pdf')
        df.plot(use_index=True, y=['aspect'])#,'iota'])#,'normalized_time'])
        plt.axvline(x = location_min, color = 'b', label = 'minimum Q')
        plt.legend()
        plt.savefig(out_dir+'/aspect_over_opt.pdf')
        df.plot(use_index=True, y=['iota'])#,'iota'])#,'normalized_time'])
        plt.axvline(x = location_min, color = 'b', label = 'minimum Q')
        plt.legend()
        plt.savefig(out_dir+'/iota_over_opt.pdf')
        df.plot(use_index=True, y=['quasisymmetry_total'])#,'iota'])#,'normalized_time'])
        plt.axvline(x = location_min, color = 'b', label = 'minimum Q')
        plt.yscale('log')
        plt.legend()
        plt.savefig(out_dir+'/qs_total_over_opt.pdf')
        if optimizer=='dual_annealing':
            df.plot(use_index=True, y=['mirror_ratio'])#,'iota'])#,'normalized_time'])
            plt.axvline(x = location_min, color = 'b', label = 'minimum Q')
            # plt.yscale('log')
            plt.legend()
            plt.savefig(out_dir+'/mirror_ratio_over_opt.pdf')
        df.plot.scatter(x='growth_rate', y='quasisymmetry_total')
        plt.yscale('log');plt.xscale('log')
        plt.savefig(out_dir+'/qs_total_vs_growth_rate.pdf')
        plt.show()
    #################################
    df_min = df.iloc[location_min]
    print('Location of minimum:')
    print(df_min)
os.chdir(out_dir)
os.makedirs('see_min', exist_ok=True)
os.chdir('see_min')
if plot_vmec:
    if use_final and os.path.isfile(f'../wout_final.nc'):
        vmec = Vmec(f'../wout_final.nc')
    elif os.path.isfile(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc') and use_previous_results_if_available:
        vmec = Vmec(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc')
    else:
        vmec = Vmec(f'../../input.nfp{nfp}_{QA_or_QH}')
        surf = vmec.boundary
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")
        vmec.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
        vmec.indata.niter_array[:3] = [ 1000,  1000, 10000]#,  5000, 10000]
        vmec.indata.ftol_array[:3]  = [1e-13, 1e-14, 1e-15]#, 1e-15, 1e-15]
        if max_mode==1:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)']]
        elif max_mode==2:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)'],
                    df_min['x(8)'],df_min['x(9)'],df_min['x(10)'],df_min['x(11)'],df_min['x(12)'],df_min['x(13)'],df_min['x(14)'],df_min['x(15)'],
                    df_min['x(16)'],df_min['x(17)'],df_min['x(18)'],df_min['x(19)'],df_min['x(20)'],df_min['x(21)'],df_min['x(22)'],df_min['x(23)']]
        elif max_mode==3:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)'],
                    df_min['x(8)'],df_min['x(9)'],df_min['x(10)'],df_min['x(11)'],df_min['x(12)'],df_min['x(13)'],df_min['x(14)'],df_min['x(15)'],
                    df_min['x(16)'],df_min['x(17)'],df_min['x(18)'],df_min['x(19)'],df_min['x(20)'],df_min['x(21)'],df_min['x(22)'],df_min['x(23)'],
                    df_min['x(24)'],df_min['x(25)'],df_min['x(26)'],df_min['x(27)'],df_min['x(28)'],df_min['x(29)'],df_min['x(30)'],df_min['x(31)'],
                    df_min['x(32)'],df_min['x(33)'],df_min['x(34)'],df_min['x(35)'],df_min['x(36)'],df_min['x(37)'],df_min['x(38)'],df_min['x(39)'],
                    df_min['x(40)'],df_min['x(41)'],df_min['x(42)'],df_min['x(43)'],df_min['x(44)'],df_min['x(45)'],df_min['x(46)'],df_min['x(47)']]
        else:
            print('Not available with that max_mode yet')
            exit()
        vmec.run()
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
    print("Aspect ratio:", vmec.aspect())
    print("Mean iota:", vmec.mean_iota())
    print("Magnetic well:", vmec.vacuum_well())
    print("Quasisymmetry objective after optimization:", qs.total())
    s_EP = 0.25;alphas_EP=[0]
    fl1 = vmec_fieldlines(vmec, s_EP, alphas_EP, theta1d=np.linspace(-4*np.pi, 4*np.pi, 250), plot=True, show=False)
    plt.savefig(f'Initial_profiles_s{s_EP}_alpha{alphas_EP[0]}.png');plt.close()
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
    fig = plt.figure(); ax = plt.subplot(111); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    ax.tick_params(axis='x', labelsize=16);ax.tick_params(axis='y', labelsize=16);plt.tight_layout()
    plt.savefig("Boozxform_surfplot_1.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
    ax.tick_params(axis='x', labelsize=16);ax.tick_params(axis='y', labelsize=16);plt.tight_layout()
    fig = plt.figure(); ax = plt.subplot(111); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    ax.tick_params(axis='x', labelsize=16);ax.tick_params(axis='y', labelsize=16);plt.tight_layout()
    plt.savefig("Boozxform_surfplot_2.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
    ax.tick_params(axis='x', labelsize=16);ax.tick_params(axis='y', labelsize=16);plt.tight_layout()
    fig = plt.figure(); ax = plt.subplot(111); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    ax.tick_params(axis='x', labelsize=16);ax.tick_params(axis='y', labelsize=16);plt.tight_layout()
    plt.savefig("Boozxform_surfplot_3.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); ax = plt.subplot(111); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
    ax.tick_params(axis='x', labelsize=16);ax.tick_params(axis='y', labelsize=16);plt.tight_layout()
    plt.savefig("Boozxform_symplot.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); ax = plt.subplot(111); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    ax.tick_params(axis='x', labelsize=16);ax.tick_params(axis='y', labelsize=16);plt.tight_layout()
    plt.savefig("Boozxform_modeplot.pdf", bbox_inches = 'tight', pad_inches = 0); plt.close()
#################################
if run_simple:
    if use_final and os.path.isfile(f'../wout_final.nc'):
        vmec = Vmec(f'../wout_final.nc')
    elif os.path.isfile(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc') and use_previous_results_if_available:
        vmec = Vmec(f'wout_nfp{nfp}_{QA_or_QH}_000_000000.nc')
    else:
        vmec = Vmec(f'../../input.nfp{nfp}_{QA_or_QH}')
        surf = vmec.boundary
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")
        vmec.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
        vmec.indata.niter_array[:3] = [ 4000, 10000, 14000]#,  5000, 10000]
        vmec.indata.ftol_array[:3]  = [1e-13, 1e-14, 1e-15]#, 1e-15, 1e-15]
        if max_mode==1:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)']]
        elif max_mode==2:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)'],
                    df_min['x(8)'],df_min['x(9)'],df_min['x(10)'],df_min['x(11)'],df_min['x(12)'],df_min['x(13)'],df_min['x(14)'],df_min['x(15)'],
                    df_min['x(16)'],df_min['x(17)'],df_min['x(18)'],df_min['x(19)'],df_min['x(20)'],df_min['x(21)'],df_min['x(22)'],df_min['x(23)']]
        elif max_mode==3:
            vmec.x = [df_min['x(0)'],df_min['x(1)'],df_min['x(2)'],df_min['x(3)'],df_min['x(4)'],df_min['x(5)'],df_min['x(6)'],df_min['x(7)'],
                    df_min['x(8)'],df_min['x(9)'],df_min['x(10)'],df_min['x(11)'],df_min['x(12)'],df_min['x(13)'],df_min['x(14)'],df_min['x(15)'],
                    df_min['x(16)'],df_min['x(17)'],df_min['x(18)'],df_min['x(19)'],df_min['x(20)'],df_min['x(21)'],df_min['x(22)'],df_min['x(23)'],
                    df_min['x(24)'],df_min['x(25)'],df_min['x(26)'],df_min['x(27)'],df_min['x(28)'],df_min['x(29)'],df_min['x(30)'],df_min['x(31)'],
                    df_min['x(32)'],df_min['x(33)'],df_min['x(34)'],df_min['x(35)'],df_min['x(36)'],df_min['x(37)'],df_min['x(38)'],df_min['x(39)'],
                    df_min['x(40)'],df_min['x(41)'],df_min['x(42)'],df_min['x(43)'],df_min['x(44)'],df_min['x(45)'],df_min['x(46)'],df_min['x(47)']]
        else:
            print('Not available with that max_mode yet')
            exit()
        vmec.run()

    wout_filename = vmec.output_file
    s_initial = 0.25 # Same s_initial as precise quasisymmetry paper
    B_scale = 5.7/vmec.wout.b0  # Scale the magnetic field by a factor
    Aminor_scale = 1.7/vmec.wout.Aminor_p  # Scale the machine size by a factor
    notrace_passing = 0  # If 1 skip tracing of passing particles

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
    data=np.column_stack([g_orbits.time, g_orbits.loss_fraction_array])
    datafile_path='./loss_history.dat'
    np.savetxt(datafile_path, data, fmt=['%s','%s'])

if plot_loss_fractions:
    # initial_loss_fractions = np.loadtxt('../../loss_history_QH_nfp4_initial.dat')
    # initial_loss_fractions_time = initial_loss_fractions[:,0]
    # initial_loss_fractions = initial_loss_fractions[:,1]
    final_loss_fractions   = np.loadtxt('loss_history.dat')
    final_loss_fractions_time = final_loss_fractions[:,0]
    final_loss_fractions = final_loss_fractions[:,1]
    fig = plt.figure()
    ax = plt.subplot(111)
    # plt.plot(initial_loss_fractions_time,initial_loss_fractions,label='QH initial')
    plt.plot(final_loss_fractions_time,final_loss_fractions,label='QH final')
    plt.ylabel('Loss fraction', fontsize=22)
    plt.xlabel('Time (s)', fontsize=22)
    # plt.legend(fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    plt.savefig('loss_fractions.pdf')
    plt.close()

if plot_neo:
    neo_executable = '/Users/rogeriojorge/local/STELLOPT/NEO/Release/xneo'
    booz_file = "boozmn_out.nc"
    neo_in_file = "../../neo_in.out"

    shutil.copyfile(neo_in_file,'neo_in.out')
    bashCommand = f'{neo_executable} out'
    run_neo = subprocess.Popen(bashCommand.split())
    run_neo.wait()

    token = open('neo_out.out','r')
    linestoken=token.readlines()
    eps_eff=[]
    s_radial=[]
    for x in linestoken:
        s_radial.append(float(x.split()[0])/150)
        eps_eff.append(float(x.split()[1])**(2/3))
    token.close()
    s_radial = np.array(s_radial)
    eps_eff = np.array(eps_eff)
    s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
    eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
    fig = plt.figure(figsize=(7, 3), dpi=200)
    ax = fig.add_subplot(111)
    plt.plot(s_radial,eps_eff, label='eps eff '+QA_or_QH)
    ax.set_yscale('log')
    plt.xlabel(r'$s=\psi/\psi_b$', fontsize=12)
    plt.ylabel(r'$\epsilon_{eff}$', fontsize=14)

    plt.tight_layout()
    fig.savefig('neo_out_'+QA_or_QH+'.pdf', dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)
