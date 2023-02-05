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
from simsopt.mhd import QuasisymmetryRatioResidual
import matplotlib
matplotlib.use('Agg') 
this_path = Path(__file__).parent.resolve()

min_bound = -0.2
max_bound = 0.2
vmec_index_scan_opt = 0
npoints_scan = 20
ftol = 1e-2

initial_config = 'input.nfp4_QH'# 'input.nfp2_QA' #'input.nfp4_QH'

HEATFLUX_THRESHOLD = 500
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

gx_executable = '/m100/home/userexternal/rjorge00/gx_latest/gx'
convert_VMEC_to_GX = '/m100/home/userexternal/rjorge00/gx_latest/geometry_modules/vmec/convert_VMEC_to_GX'
##
LN = 1.0
LT = 3.0
nstep = 10000
dt = 0.02
nzgrid = 91
npol = 4
desired_normalized_toroidal_flux = 0.25
alpha_fieldline = 0
nhermite  = 15
nlaguerre = 5
nu_hyper = 0.5
D_hyper = 0.05
ny = 80
nx = 80
y0 = 10.0
nonlinear = False # True

OUT_DIR = os.path.join(this_path,f'test_optimization_{initial_config[-7:]}')

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
filename = os.path.join(this_path, initial_config)
vmec = Vmec(filename, verbose=False)
vmec.keep_all_files = True
if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
surf.fix("rc(0,0)")
output_to_csv = True

# Save final eigenfunction
def eigenPlot(stellFile, fractionToConsider=0.4):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    y = np.array(f.groups['Special']['Phi_z'][()])
    x = np.array(f.variables['theta'][()])
    tX   = f.variables['time'][()]
    kyX  = f.variables['ky'][()]
    kxX  = f.variables['kx'][()]
    startIndexX  = int(len(tX)*(1-fractionToConsider))
    omega_average_array = np.array(f.groups['Special']['omega_v_time'][()])
    growthRateX = np.mean(omega_average_array[startIndexX:,:,:,1],axis=0)
    max_index = np.unravel_index(growthRateX.argmax(),growthRateX.shape)
    max_growthrate_ky = kyX[max_index[0]]
    max_growthrate_kx = kxX[max_index[1]]
    plt.figure()
    #######
    phiR0= y[max_index[0],max_index[1],int((len(x)-1)/2),0]
    phiI0= y[max_index[0],max_index[1],int((len(x)-1)/2),1]
    phi02= phiR0**2+phiI0**2
    phiR = (y[max_index[0],max_index[1],:,0]*phiR0+y[max_index[0],max_index[1],:,1]*phiI0)/phi02
    phiI = (y[max_index[0],max_index[1],:,1]*phiR0-y[max_index[0],max_index[1],:,0]*phiI0)/phi02
    plt.plot(x, phiR, label=r'Re($\hat \phi/\hat \phi_0$) $k_x$='+str(max_growthrate_kx)+r' $k_y$='+str(max_growthrate_ky))
    plt.plot(x, phiI, label=r'Im($\hat \phi/\hat \phi_0$) $k_x$='+str(max_growthrate_kx)+r' $k_y$='+str(max_growthrate_ky))
    #######
    phiR0= y[max_index[0],int((len(kxX)-1)/2),int((len(x)-1)/2),0]
    phiI0= y[max_index[0],int((len(kxX)-1)/2),int((len(x)-1)/2),1]
    phi02= phiR0**2+phiI0**2
    phiR = (y[max_index[0],int((len(kxX)-1)/2),:,0]*phiR0+y[max_index[0],int((len(kxX)-1)/2),:,1]*phiI0)/phi02
    phiI = (y[max_index[0],int((len(kxX)-1)/2),:,1]*phiR0-y[max_index[0],int((len(kxX)-1)/2),:,0]*phiI0)/phi02
    plt.plot(x, phiR, label=r'Re($\hat \phi/\hat \phi_0$) $k_x$='+str(kxX[int((len(kxX)-1)/2)])+r' $k_y$='+str(max_growthrate_ky))
    plt.plot(x, phiI, label=r'Im($\hat \phi/\hat \phi_0$) $k_x$='+str(kxX[int((len(kxX)-1)/2)])+r' $k_y$='+str(max_growthrate_ky))
    #######
    plt.xlabel(r'$\theta$');plt.ylabel(r'$\hat \phi$');plt.legend(loc="upper right")
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.93)
    plt.savefig(stellFile+'_eigenphi.png')
    plt.close()
    return 0
def gammabyky(stellFile, fractionToConsider=0.4):
    fX   = netCDF4.Dataset(stellFile,'r',mmap=False)
    tX   = fX.variables['time'][()]
    startIndexX  = int(len(tX)*(1-fractionToConsider))
    kyX  = fX.variables['ky'][()]
    kxX  = fX.variables['kx'][()]
    omega_average_array = np.array(fX.groups['Special']['omega_v_time'][()])
    realFrequencyX = np.mean(omega_average_array[startIndexX:,:,:,0],axis=0)
    growthRateX = np.mean(omega_average_array[startIndexX:,:,:,1],axis=0)
    max_index = np.unravel_index(growthRateX.argmax(),growthRateX.shape)
    max_growthrate_omega = realFrequencyX[max_index[0],max_index[1]]
    max_growthrate_gamma = growthRateX[max_index[0],max_index[1]]
    max_growthrate_ky = kyX[max_index[0]]
    max_growthrate_kx = kxX[max_index[1]]

    numRows = 2
    numCols = 2

    plt.subplot(numRows, numCols, 1)
    plt.plot(kyX,growthRateX[:,max_index[1]],'.-')
    plt.xlabel('ky')
    plt.ylabel('gamma')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)

    plt.subplot(numRows, numCols, 2)
    plt.plot(kyX,realFrequencyX,'.-')
    plt.xlabel('ky')
    plt.ylabel('omega')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)

    plt.subplot(numRows, numCols, 3)
    plt.plot(kxX,growthRateX[max_index[0],:],'.-')
    plt.xlabel('kx')
    plt.ylabel('gamma')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)

    plt.subplot(numRows, numCols, 4)
    for count, ky in enumerate(kyX): plt.plot(tX[2:],omega_average_array[2:,count,max_index[1],1],'.-', label=f'gamma at ky={ky}')
    plt.xlabel('time')
    plt.ylabel('gamma')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size=12),loc=0)

    plt.tight_layout()
    plt.savefig(stellFile+"_GammaOmegaKy.png")
    plt.close()
    return max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky
def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)
def output_dofs_to_csv(csv_path,dofs,mean_iota,aspect,growth_rate,quasisymmetry,well,qflux,effective_1o_time=0):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['growth_rate'],['quasisymmetry'],['well'],['qflux'],['effective_1o_time']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[growth_rate],[quasisymmetry],[well],[qflux],[effective_1o_time]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(csv_path): pd.DataFrame(columns=df.columns).to_csv(csv_path, index=False)
    df.to_csv(csv_path, mode='a', header=False, index=False)
def create_gx_inputs(vmec_file):
    f_wout = vmec_file.split('/')[-1]
    # if not os.path.isfile(os.path.join(OUT_DIR,f_wout)): shutil.copy(vmec_file,os.path.join(OUT_DIR,f_wout))
    geometry_file = os.path.join(OUT_DIR,f'gx-geometry_wout_{f_wout[5:-3]}.ing')
    shutil.copy(os.path.join(this_path,'gx-geometry-sample.ing'),geometry_file)
    replace(geometry_file,'nzgrid = 32',f'nzgrid = {nzgrid}')
    replace(geometry_file,'npol = 2',f'npol = {npol}')
    replace(geometry_file,'desired_normalized_toroidal_flux = 0.12755',f'desired_normalized_toroidal_flux = {desired_normalized_toroidal_flux:.3f}')
    replace(geometry_file,'vmec_file = "wout_gx.nc"',f'vmec_file = "{f_wout}"')
    replace(geometry_file,'alpha = 0.0"',f'alpha = {alpha_fieldline}')
    if not os.path.isfile(os.path.join(OUT_DIR,'convert_VMEC_to_GX')): shutil.copy(convert_VMEC_to_GX,os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
    p = subprocess.Popen(f"./convert_VMEC_to_GX gx-geometry_wout_{f_wout[5:-3]}".split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)
    p.wait()
    try: os.remove(geometry_file)
    except Exception as e: print(e)
    gridout_file = f'grid.gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux}_nt_{2*nzgrid}'
    os.remove(os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
    fname = f"gxInput_wout_{f_wout[5:-3]}"
    fnamein = os.path.join(OUT_DIR,fname+'.in')
    if nonlinear: shutil.copy(os.path.join(this_path,'gx-input_nl.in'),fnamein)
    else: shutil.copy(os.path.join(this_path,'gx-input.in'),fnamein)
    replace(fnamein,' geofile = "gx_wout.nc"',f' geofile = "gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}_geo.nc"')
    replace(fnamein,' gridout_file = "grid.out"',f' gridout_file = "{gridout_file}"')
    replace(fnamein,' nstep  = 9000',f' nstep  = {nstep}')
    replace(fnamein,' fprim = [ 1.0,       1.0     ]',f' fprim = [ {LN},       {LN}     ]')
    replace(fnamein,' tprim = [ 3.0,       3.0     ]',f' tprim = [ {LT},       {LT}     ]')
    replace(fnamein,' dt = 0.010',f' dt = {dt}')
    replace(fnamein,' ntheta = 80',f' ntheta = {2*nzgrid}')
    replace(fnamein,' nhermite  = 18',f' nhermite = {nhermite}')
    replace(fnamein,' nlaguerre = 10',f' nlaguerre = {nlaguerre}')
    replace(fnamein,' nu_hyper_m = 1.0',f' nu_hyper_m = {nu_hyper}')
    replace(fnamein,' nu_hyper_l = 1.0',f' nu_hyper_l = {nu_hyper}')
    replace(fnamein,' ny = 30',f' ny = {ny}')
    replace(fnamein,' nx = 1',f' nx = {nx}')
    replace(fnamein,' D_hyper = 0.05',f' D_hyper = {D_hyper}')
    replace(fnamein,' y0 = 20.0',f' y0 = {y0}')
    os.remove(os.path.join(OUT_DIR,f_wout))
    return fname
# Function to remove spurious GX files
def remove_gx_files(gx_input_name):
    f_wout_only = gx_input_name[11:]
    # print(f'removing grid.gx_wout_{f_wout_only}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}')
    try: os.remove(f'grid.gx_wout_{f_wout_only}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.in')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.nc')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.log')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.restart.nc')
    except Exception as e: pass#print(e)
    try: os.remove(f'gx_wout_{f_wout_only}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}_geo.nc')
    except Exception as e: pass#print(e)
    # for f in glob.glob('*.restart.nc'): remove(f)
    # for f in glob.glob('*.log'): remove(f)
    # for f in glob.glob('grid.*'): remove(f)
    # for f in glob.glob('gx_wout*'): remove(f)
    # for f in glob.glob('gxRun_*'): remove(f)
    # for f in glob.glob('input.*'): remove(f)
    ## REMOVE ALSO INPUT FILE
    # for f in glob.glob('*.in'): remove(f)
    ## REMOVE ALSO OUTPUT FILE
    # for f in glob.glob(f'{gx_input_name}.nc'): remove(f)
def get_qflux(stellFile, fractionToConsider=0.4):
    fX = netCDF4.Dataset(stellFile,'r',mmap=False)
    qflux = np.nan_to_num(np.array(fX.groups['Fluxes'].variables['qflux'][:,0]))
    time = np.array(fX.variables['time'][:])
    startIndexX  = int(len(time)*(1-fractionToConsider))
    Q_avg = np.mean(qflux[startIndexX:])
    plt.figure();plt.plot(time,qflux);plt.xlabel('time');plt.ylabel(f'Q (average = {Q_avg:.1f})')
    plt.savefig(f'{stellFile}_heatFlux.png')
    return Q_avg
# Function to run GS2 and extract growth rate
def run_gx(vmec: Vmec):
    gx_input_name = create_gx_inputs(vmec.output_file)
    f_log = os.path.join(OUT_DIR,gx_input_name+".log")
    gx_cmd = [f"{gx_executable}", f"{os.path.join(OUT_DIR,gx_input_name+'.in')}", "1"]
    with open(f_log, 'w') as fp:
        p = subprocess.Popen(gx_cmd,stdout=fp)
    p.wait()
    fout = os.path.join(OUT_DIR,gx_input_name+".nc")
    try:
        eigenPlot(fout)
        max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky = gammabyky(fout)
        qflux = get_qflux(fout)
    except Exception as e:
        print(e)
        max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux = HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD
    remove_gx_files(gx_input_name)
    return max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux
def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return GROWTHRATE_THRESHOLD
    try:
        growth_rate, max_growthrate_omega, max_growthrate_ky, qflux = run_gx(v)
    except Exception as e:
        growth_rate, max_growthrate_omega, max_growthrate_ky = GROWTHRATE_THRESHOLD, GROWTHRATE_THRESHOLD, GROWTHRATE_THRESHOLD
    # out_str = f'{datetime.now().strftime("%H:%M:%S")} - Growth rate = {growth_rate:1f}, quasisymmetry = {qs.total():1f} with aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s'
    out_str = f'Growth rate = {growth_rate:1f}, qflux = {qflux:1f} for point {(vmec.x[vmec_index_scan_opt]):1f}, aspect {np.abs(v.aspect()):1f}, quasisymmetry = {qs.total():1f} and iota {(v.mean_iota()):1f} took {(time.time()-start_time):1f}s'
    print(out_str)
    if output_to_csv: output_dofs_to_csv(output_path_parameters_opt, v.x,v.mean_iota(),v.aspect(),growth_rate,qs.total(),v.vacuum_well(), qflux)
    else: output_dofs_to_csv(output_path_parameters_scan, v.x,v.mean_iota(),v.aspect(),growth_rate,qs.total(),v.vacuum_well(), qflux)
    if nonlinear: return qflux
    else: return growth_rate
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
try: os.remove(os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
except Exception as e: print(e)
for f in glob.glob('grid.gx_wout*'): remove(f)
for f in glob.glob('gx_wout*'): remove(f)
for f in glob.glob('input.*'): remove(f)
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
        plt.ylabel('Growth Rate');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('growth_rate_over_opt_scan.png')

        fig = plt.figure()
        plt.plot(df_opt[f'x({vmec_index_scan_opt})'], df_opt['qflux'], 'ro', markersize=1, label='Optimizer')
        plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['qflux'], label='Scan')
        plt.ylabel('Heat flux');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('heat_flux_over_opt_scan.png')
    except Exception as e: print(e)
    points_scan = np.linspace(min_bound,max_bound,len(df_scan[f'x({vmec_index_scan_opt})']))
    fig = plt.figure();plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], label='Scan')
    plt.ylabel('Growth Rate');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('growth_rate_scan.png')
    fig = plt.figure();plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['qflux'], label='Scan')
    plt.ylabel('Heat Flux');plt.xlabel('RBC(1,0)');plt.legend();plt.savefig('heat_flux_scan.png')
    fig = plt.figure();plt.plot(points_scan, df_scan['aspect'], label='Aspect ratio')
    plt.ylabel('Aspect ratio');plt.xlabel('RBC(1,0)');plt.savefig('aspect_ratio_scan.png')
    fig = plt.figure();plt.plot(points_scan, df_scan['mean_iota'], label='Rotational Transform (1/q)')
    plt.ylabel('Rotational Transform (1/q)');plt.xlabel('RBC(1,0)');plt.savefig('iota_scan.png')
    fig = plt.figure();plt.plot(points_scan, df_scan['quasisymmetry'], label='Quasisymmetry cost function')
    plt.ylabel('Quasisymmetry cost function');plt.xlabel('RBC(1,0)');plt.savefig('quasisymmetry_scan.png')
    fig = plt.figure();plt.plot(points_scan, df_scan['well'], label='Magnetic well')
    plt.ylabel('Magnetic well');plt.xlabel('RBC(1,0)');plt.savefig('magnetic_well_scan.png')
    # fig = plt.figure();plt.plot(points_scan, df_scan['effective_1o_time'], label='Effective 1/time')
    # plt.ylabel('Effective time');plt.xlabel('RBC(1,0)');plt.savefig('effective_1o_time_scan.png')

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
    plt.savefig('quasisymmetry_vs_growthrate.png')

    fig=plt.figure(figsize=(6,4.5))
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    ax.set_xlabel('RBC(1,0)')
    line1, = ax.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['qflux'], color="C0", label='Heat Flux')
    ax.set_ylabel("Heat Flux", color="C0")
    ax.tick_params(axis='y', colors="C0")
    line2, = ax2.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['quasisymmetry'], color="C1", label='Quasisymmetry cost function')
    ax2.yaxis.tick_right()
    ax2.set_xticks([])
    ax2.set_ylabel('Quasisymmetry cost function', color="C1") 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='y', colors="C1")
    plt.legend(handles=[line1, line2])
    plt.tight_layout()
    plt.savefig('quasisymmetry_vs_heatflux.png')
