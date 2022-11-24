#!/usr/bin/env python
import os
import glob
import time
import random
import shutil
import netCDF4
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from scipy.optimize import dual_annealing
from matplotlib.animation import FuncAnimation
from simsopt.mhd import Vmec
from simsopt import make_optimizable
from simsopt.mhd.vmec_diagnostics import to_gs2
from simsopt.mhd import QuasisymmetryRatioResidual
import matplotlib
matplotlib.use('Agg') 
this_path = Path(__file__).parent.resolve()

min_bound = -0.20
max_bound = 0.20
vmec_index_scan_opt = 0
npoints_scan = 25
ftol = 1e-2
s_radius = 0.25
alpha_fieldline = 0
LN = 3.0
LT = 3.0

# initial_config = 'input.nfp4_QH'
# phi_GS2 = np.linspace(-7*np.pi, 7*np.pi, 111)
# nlambda = 27
# nstep = 170
# dt = 0.4

initial_config = 'input.nfp2_QA'
phi_GS2 = np.linspace(-23*np.pi, 23*np.pi, 101)
nlambda = 25
nstep = 170
dt = 0.4

naky = 10
aky_min = 0.2
aky_max = 2.0
s_radius = 0.25
alpha_fieldline = 0
ngauss = 3
negrid = 9

HEATFLUX_THRESHOLD = 1e18
GROWTHRATE_THRESHOLD = 10

MAXITER = 10
MAXFUN = 50
MAXITER_LOCAL = 2
MAXFUN_LOCAL = 5
run_scan = True
run_optimization = False
plot_result = True

output_path_parameters_opt = 'opt_dofs_loss.csv'
output_path_parameters_scan = 'scan_dofs_loss.csv'
output_path_parameters_min = 'min_dofs_loss.csv'

gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2'

OUT_DIR = os.path.join(this_path,f'test_optimization_{initial_config[-7:]}_ln{LN}_lt{LT}')

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
filename = os.path.join(this_path, initial_config)
vmec = Vmec(filename, verbose=False)
if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
surf.fix("rc(0,0)")
output_to_csv = True
# Get growth rates
def getgamma(stellFile, fractionToConsider=0.35, plot=False):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    phi2 = np.log(f.variables['phi2'][()])
    t = f.variables['t'][()]
    startIndex = int(len(t)*(1-fractionToConsider))
    mask = np.isfinite(phi2)
    data_x = t[mask]
    data_y = phi2[mask]
    fit = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)
    poly = np.poly1d(fit)
    GrowthRate = fit[0]/2
    omega_average_array = np.array(f.variables['omega_average'][()])
    omega_average_array_omega = omega_average_array[-1,:,0,0]
    omega_average_array_gamma = omega_average_array[-1,:,0,1]
    max_index = np.nanargmax(omega_average_array_gamma)
    gamma = omega_average_array_gamma[max_index]
    omega = omega_average_array_omega[max_index]
    if plot:
        plt.figure(figsize=(7.5,4.0))
        ##############
        plt.plot(t, phi2,'.', label=r'data - $\gamma_{GS2} = $'+str(gamma))
        plt.plot(t, poly(t),'-', label=r'fit - $\gamma = $'+str(GrowthRate))
        ##############
        plt.legend(loc=0,fontsize=14)
        plt.xlabel(r'$t$');plt.ylabel(r'$\ln |\hat \phi|^2$')
        plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
        plt.savefig(stellFile+'_phi2.png')
        plt.close()
    return GrowthRate, abs(omega)
# Save final eigenfunction
def eigenPlot(stellFile):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    y = f.variables['phi'][()]
    x = f.variables['theta'][()]
    plt.figure(figsize=(7.5,4.0))
    omega_average_array = np.array(f.variables['omega_average'][()])
    omega_average_array_gamma = omega_average_array[-1,:,0,1]
    max_index = np.nanargmax(omega_average_array_gamma)
    phiR0= y[max_index,0,int((len(x)-1)/2+1),0]
    phiI0= y[max_index,0,int((len(x)-1)/2+1),1]
    phi02= phiR0**2+phiI0**2
    phiR = (y[max_index,0,:,0]*phiR0+y[max_index,0,:,1]*phiI0)/phi02
    phiI = (y[max_index,0,:,1]*phiR0-y[max_index,0,:,0]*phiI0)/phi02
    ##############
    plt.plot(x, phiR, label=r'Re($\hat \phi/\hat \phi_0$)')
    plt.plot(x, phiI, label=r'Im($\hat \phi/\hat \phi_0$)')
    ##############
    plt.xlabel(r'$\theta$');plt.ylabel(r'$\hat \phi$')
    plt.legend(loc="upper right")
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.93)
    plt.savefig(stellFile+'_eigenphi.png')
    plt.close()
    return 0
##### Function to obtain gamma and omega for each ky
def gammabyky(stellFile,fractionToConsider=0.6):
    # Compute growth rate:
    fX   = netCDF4.Dataset(stellFile,'r',mmap=False)
    tX   = fX.variables['t'][()]
    kyX  = fX.variables['ky'][()]
    phi2_by_kyX  = fX.variables['phi2_by_ky'][()]
    omegaX  = fX.variables['omega'][()]
    startIndexX  = int(len(tX)*(1-fractionToConsider))
    growthRateX  = []
    ## assume that kyX=kyNA
    for i in range(len(kyX)):
        maskX  = np.isfinite(phi2_by_kyX[:,i])
        data_xX = tX[maskX]
        data_yX = phi2_by_kyX[maskX,i]
        fitX  = np.polyfit(data_xX[startIndexX:], np.log(data_yX[startIndexX:]), 1)
        thisGrowthRateX  = fitX[0]/2
        growthRateX.append(thisGrowthRateX)
    # Compute real frequency:
    realFreqVsTimeX  = []
    realFrequencyX   = []
    for i in range(len(kyX)):
        realFreqVsTimeX.append(omegaX[:,i,0,0])
        realFrequencyX.append(np.mean(realFreqVsTimeX[i][startIndexX:]))
    numRows = 1
    numCols = 2

    plt.subplot(numRows, numCols, 1)
    plt.plot(kyX,growthRateX,'.-')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\gamma$')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size='xx-small'),loc=0)

    plt.subplot(numRows, numCols, 2)
    plt.plot(kyX,realFrequencyX,'.-')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\omega$')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size=12),loc=0)

    plt.tight_layout()
    #plt.subplots_adjust(left=0.14, bottom=0.15, right=0.98, top=0.96)
    plt.savefig(stellFile+"_GammaOmegaKy.png")
    plt.close()
    return kyX, growthRateX, realFrequencyX
def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)
def remove_gs2_files(gs2_input_name):
    for f in glob.glob('*.amoments'): remove(f)
    for f in glob.glob('*.eigenfunc'): remove(f)
    for f in glob.glob('*.error'): remove(f)
    for f in glob.glob('*.fields'): remove(f)
    for f in glob.glob('*.g'): remove(f)
    for f in glob.glob('*.lpc'): remove(f)
    for f in glob.glob('*.mom2'): remove(f)
    for f in glob.glob('*.moments'): remove(f)
    for f in glob.glob('*.vres'): remove(f)
    for f in glob.glob('*.vres2'): remove(f)
    for f in glob.glob('*.exit_reason'): remove(f)
    for f in glob.glob('*.optim'): remove(f)
    for f in glob.glob('*.out'): remove(f)
    for f in glob.glob('*.used_inputs.in'): remove(f)
    for f in glob.glob(f'{gs2_input_name}.in'): remove(f)
    for f in glob.glob('*.vspace_integration_error'): remove(f)
    ## REMOVE ALSO OUTPUT FILE
    for f in glob.glob('*.out.nc'): remove(f)
def output_dofs_to_csv(csv_path,dofs,mean_iota,aspect,growth_rate,quasisymmetry,well,effective_1o_time=0):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['growth_rate'],['quasisymmetry'],['well'],['effective_1o_time']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[growth_rate],[quasisymmetry],[well],[effective_1o_time]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(csv_path): pd.DataFrame(columns=df.columns).to_csv(csv_path, index=False)
    df.to_csv(csv_path, mode='a', header=False, index=False)
def CalculateGrowthRate(v: Vmec):
    try:
        v.run()
        f_wout = v.output_file.split('/')[-1]
        gs2_input_name = f"gs2-{f_wout[5:-3]}"
        gs2_input_file = os.path.join(OUT_DIR,f'{gs2_input_name}.in')
        shutil.copy(os.path.join(this_path,'gs2Input.in'),gs2_input_file)
        gridout_file = os.path.join(OUT_DIR,f'grid_{gs2_input_name}.out')
        replace(gs2_input_file,' gridout_file = "grid.out"',f' gridout_file = "{gridout_file}"')
        replace(gs2_input_file,' nstep = 150 ! Maximum number of timesteps',f' nstep = {nstep} ! Maximum number of timesteps"')
        replace(gs2_input_file,' fprim = 1.0 ! -1/n (dn/drho)',f' fprim = {LN} ! -1/n (dn/drho)')
        replace(gs2_input_file,' tprim = 3.0 ! -1/T (dT/drho)',f' tprim = {LT} ! -1/T (dT/drho)')
        replace(gs2_input_file,' delt = 0.4 ! Time step',f' delt = {dt} ! Time step')
        replace(gs2_input_file,' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.',
        f' ngauss = {ngauss} ! Number of untrapped pitch-angles moving in one direction along field line.')
        replace(gs2_input_file,' negrid = 10 ! Total number of energy grid points',
        f' negrid = {negrid} ! Total number of energy grid points')
        replace(gs2_input_file,' naky = 6',f' naky = {naky}')
        replace(gs2_input_file,' aky_min = 0.3',f' aky_min = {aky_min}')
        replace(gs2_input_file,' aky_max = 10.0',f' aky_max = {aky_max}')
        to_gs2(gridout_file, v, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=nlambda)
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        # f_log = os.path.join(OUT_DIR,f"{gs2_input_name}.log")
        # with open(f_log, 'w') as fp:
        p = subprocess.Popen(bashCommand.split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)#stdout=fp)
        p.wait()
        file2read = os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc")#netCDF4.Dataset(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"),'r')
        # eigenPlot(file2read)
        growth_rate, omega = getgamma(file2read)
        # kyX, growthRateX, realFrequencyX = gammabyky(file2read)
        remove_gs2_files(file2read)
        if not np.isfinite(growth_rate): growth_rate = HEATFLUX_THRESHOLD

    except Exception as e:
        print(e)
        # qavg = HEATFLUX_THRESHOLD
        growth_rate = GROWTHRATE_THRESHOLD

    # try:
    #     for objective_file in glob.glob(os.path.join(OUT_DIR,f"*{gs2_input_name}*")): os.remove(objective_file)
    # except Exception as e: pass
    # try:
    #     for objective_file in glob.glob(os.path.join(OUT_DIR,f".{gs2_input_name}*")): os.remove(objective_file)
    # except Exception as e: pass
    
    return growth_rate#qavg
def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return GROWTHRATE_THRESHOLD
    try:
        growth_rate = CalculateGrowthRate(v)
    except Exception as e:
        growth_rate = GROWTHRATE_THRESHOLD
    # out_str = f'{datetime.now().strftime("%H:%M:%S")} - Growth rate = {growth_rate:1f}, quasisymmetry = {qs.total():1f} with aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s'
    out_str = f'Growth rate = {growth_rate:1f} for point {(vmec.x[vmec_index_scan_opt]):1f}, aspect {np.abs(v.aspect()):1f}, quasisymmetry = {qs.total():1f} and iota {(v.mean_iota()):1f} took {(time.time()-start_time):1f}s'
    print(out_str)
    if output_to_csv: output_dofs_to_csv(output_path_parameters_opt, v.x,v.mean_iota(),v.aspect(),growth_rate,qs.total(),v.vacuum_well())
    else: output_dofs_to_csv(output_path_parameters_scan, v.x,v.mean_iota(),v.aspect(),growth_rate,qs.total(),v.vacuum_well())
    return growth_rate
optTurbulence = make_optimizable(TurbulenceCostFunction, vmec)
def fun(dofss):
    vmec.x = [dofss[0] if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
    return optTurbulence.J()
if run_optimization:
    if os.path.exists(output_path_parameters_opt): os.remove(output_path_parameters_opt)
    if os.path.exists(output_path_parameters_min): os.remove(output_path_parameters_min)
    output_to_csv = True
    bounds = [(min_bound,max_bound)]
    minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds, "options": {'maxiter': MAXITER_LOCAL, 'maxfev': MAXFUN_LOCAL, 'disp': True}}
    global_minima_found = []
    def print_fun(x, f, context):
        if context==0: context_string = 'Minimum detected in the annealing process.'
        elif context==1: context_string = 'Detection occurred in the local search process.'
        elif context==2: context_string = 'Detection done in the dual annealing process.'
        else: print(context)
        print(f'New minimum found! x={x[0]:1f}, f={f:1f}. {context_string}')
        output_dofs_to_csv(output_path_parameters_min,vmec.x,vmec.mean_iota(),vmec.aspect(),f,qs.total(),vmec.vacuum_well())
        if len(global_minima_found)>4 and np.abs((f-global_minima_found[-1])/f)<ftol:
            # Stop optimization
            return True
        else:
            global_minima_found.append(f)
    no_local_search = False
    res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, maxfun=MAXFUN, x0=[random.uniform(min_bound,max_bound)], no_local_search=no_local_search, minimizer_kwargs=minimizer_kwargs, callback=print_fun)
    print(f"Global minimum: x = {res.x}, f(x) = {res.fun}")
    vmec.x = [res.x[0] if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
if run_scan:
    if os.path.exists(output_path_parameters_scan): os.remove(output_path_parameters_scan)
    output_to_csv = False
    for point1 in np.linspace(min_bound,max_bound,npoints_scan):
        vmec.x = [point1 if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
        growth_rate = optTurbulence.J()
if plot_result:
    df_scan = pd.read_csv(output_path_parameters_scan)

    try:
        df_opt = pd.read_csv(output_path_parameters_opt)
        fig, ax = plt.subplots()
        plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], label='Scan')
        ln, = ax.plot([], [], 'ro', markersize=1)
        vl = ax.axvline(0, ls='-', color='r', lw=1)
        patches = [ln,vl]
        ax.set_xlim(min_bound,max_bound)
        ax.set_ylim(np.min(0.8*df_scan['growth_rate']), np.max(df_scan['growth_rate']))
        def update(frame):
            ind_of_frame = df_opt.index[df_opt[f'x({vmec_index_scan_opt})'] == frame][0]
            df_subset = df_opt.head(ind_of_frame+1)
            xdata = df_subset[f'x({vmec_index_scan_opt})']
            ydata = df_subset['growth_rate']
            vl.set_xdata([frame,frame])
            ln.set_data(xdata, ydata)
            return patches
        ani = FuncAnimation(fig, update, frames=df_opt[f'x({vmec_index_scan_opt})'])
        ani.save('opt_animation.gif', writer='imagemagick', fps=5)

        fig = plt.figure()
        plt.plot(df_opt[f'x({vmec_index_scan_opt})'], df_opt['growth_rate'], 'ro', markersize=1, label='Optimizer')
        plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], label='Scan')
        plt.ylabel('Growth Rate');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('growth_rate_over_opt_scan.pdf')
    except Exception as e: print(e)
    points_scan = np.linspace(min_bound,max_bound,len(df_scan[f'x({vmec_index_scan_opt})']))
    fig = plt.figure();plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], label='Scan')
    plt.ylabel('Growth Rate');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('growth_rate_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['aspect'], label='Aspect ratio')
    plt.ylabel('Aspect ratio');plt.xlabel('RBC(1,0)');plt.savefig('aspect_ratio_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['mean_iota'], label='Rotational Transform (1/q)')
    plt.ylabel('Rotational Transform (1/q)');plt.xlabel('RBC(1,0)');plt.savefig('iota_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['quasisymmetry'], label='Quasisymmetry cost function')
    plt.ylabel('Quasisymmetry cost function');plt.xlabel('RBC(1,0)');plt.savefig('quasisymmetry_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['well'], label='Magnetic well')
    plt.ylabel('Magnetic well');plt.xlabel('RBC(1,0)');plt.savefig('magnetic_well_scan.pdf')
    # fig = plt.figure();plt.plot(points_scan, df_scan['effective_1o_time'], label='Effective 1/time')
    # plt.ylabel('Effective time');plt.xlabel('RBC(1,0)');plt.savefig('effective_1o_time_scan.pdf')

    fig=plt.figure(figsize=(6,4.5))
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    ax.set_xlabel('RBC(1,0)')
    line1, = ax.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], color="C0", label='Growth Rate')
    ax.set_ylabel("Growth Rate", color="C0")
    ax.tick_params(axis='y', colors="C0")
    line2, = ax2.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['quasisymmetry'], color="C1", label='Quasisymmetry cost function')
    ax2.yaxis.tick_right()
    ax2.set_xticks([])
    ax2.set_ylabel('Quasisymmetry cost function', color="C1") 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='y', colors="C1")
    plt.legend(handles=[line1, line2])
    plt.tight_layout()
    plt.savefig('quasisymmetry_vs_growthrate.pdf')