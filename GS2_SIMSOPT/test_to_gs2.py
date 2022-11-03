#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib
from pathlib import Path
import matplotlib.cm as cm
from scipy.io import netcdf
from tempfile import mkstemp
import matplotlib.pyplot as plt
from os import path, fdopen, remove
from shutil import move, copymode, copyfile
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import to_gs2
this_path = Path(__file__).parent.resolve()

gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2'
gs2_input = 'gs2Input.in'
gridout_file = "grid.out"
filename = 'wout_nfp4_QH_000_000000.nc'#'input.nfp4_QH'
s_radius = 0.5
alpha_fieldline = 0
theta = np.linspace(-np.pi, np.pi, 50)

vmec = Vmec(os.path.join(this_path,filename))
to_gs2(vmec, s_radius, alpha_fieldline, theta1d=theta)

plotfontSize=20;figSize1=7.5;figSize2=4.0;legendfontSize=14;annotatefontSize=8;
matplotlib.rc('font', size=plotfontSize);matplotlib.rc('axes', titlesize=plotfontSize);
matplotlib.rc('text', usetex=True);matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

## Function to replace text in files
def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

###### Function to remove spurious GS2 files
def removeGS2(Path):
	os.chdir(Path)
	for f in glob.glob('*.amoments'): remove(f); 
	for f in glob.glob('*.eigenfunc'): remove(f)
	for f in glob.glob('*.error'): remove(f); 
	for f in glob.glob('*.fields'): remove(f)
	for f in glob.glob('*.g'): remove(f);
	for f in glob.glob('*.lpc'): remove(f)
	for f in glob.glob('*.mom2'): remove(f);
	for f in glob.glob('*.moments'): remove(f)
	for f in glob.glob('*.vres'): remove(f);
	for f in glob.glob('*.out'): remove(f)

###### Function to obtain growth rate and plot phi2
def getgamma(stellFile):
	initialT=20
	f = netcdf.netcdf_file(stellFile,'r',mmap=False)
	y = np.log(f.variables['phi2'][()])
	x = f.variables['t'][()]
	coeffs = np.polyfit(x[initialT:], y[initialT:], 1)
	gamma  = coeffs[0]
	fitRes = np.poly1d(coeffs)
	plt.figure(figsize=(figSize1,figSize2))
	##############
	plt.plot(x, y,'.', label='data')
	plt.plot(x, fitRes(x),'-', label=r'fit - $\gamma = $'+str(gamma))
	##############
	plt.legend(loc=0,fontsize=legendfontSize);
	plt.xlabel(r'$t$');plt.ylabel(r'$\ln |\phi|^2$');
	plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
	plt.savefig(stellFile+'_phi2.pdf', format='pdf')
	plt.close()
	return gamma

###### Function to plot eigenfunctions
def eigenPlot(stellFile):
	f = netcdf.netcdf_file(stellFile,'r',mmap=False)
	y = f.variables['phi'][()]
	x = f.variables['theta'][()]
	plt.figure(figsize=(figSize1,figSize2))
	##############
	plt.plot(x, y[0,0,:,0], label=r'Re($\phi$)')
	plt.plot(x, y[0,0,:,1], label=r'Im($\phi$)')
	##############
	plt.xlabel(r'$\theta$');plt.ylabel(r'$\phi$');
	plt.legend(loc="upper right")
	plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
	plt.savefig(stellFile+'_eigenphi.pdf', format='pdf')
	plt.close()
	return 0

###### Function to plot geometry coefficients
def geomPlot(stells,rr,stellFileX,stellFileNA):
	fX  = netcdf.netcdf_file(stellFileX,'r',mmap=False)
	fNA = netcdf.netcdf_file(stellFileNA,'r',mmap=False)
	theta      = fX.variables['theta'][()]
	lambdaX    = fX.variables['lambda'][()]
	lambdaNA   = fNA.variables['lambda'][()]
	gbdriftX   = fX.variables['gbdrift'][()]
	gbdriftNA  = fNA.variables['gbdrift'][()]
	gbdrift0X  = fX.variables['gbdrift0'][()]
	gbdrift0NA = fNA.variables['gbdrift0'][()]
	cvdriftX   = fX.variables['cvdrift'][()]
	cvdriftNA  = fNA.variables['cvdrift'][()]
	cvdrift0X  = fX.variables['cvdrift0'][()]
	cvdrift0NA = fNA.variables['cvdrift0'][()]
	matplotlib.rc('font', size=6);
	nrows=5; ncols=2;fig = plt.figure()
	##
	plt.subplot(nrows, ncols, 1);
	plt.scatter(theta, fX.variables['gradpar'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['gradpar'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'gradpar')
	##
	plt.subplot(nrows, ncols, 2);
	plt.scatter(theta, fX.variables['bmag'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['bmag'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'bmag')
	##
	plt.subplot(nrows, ncols, 3);
	plt.scatter(theta, fX.variables['gds2'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['gds2'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'gds2')
	##
	plt.subplot(nrows, ncols, 4);
	plt.scatter(theta, fX.variables['gds21'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['gds21'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'gds21')
	##
	plt.subplot(nrows, ncols, 5);
	plt.scatter(theta, fX.variables['gds22'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['gds22'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'gds22')
	##
	plt.subplot(nrows, ncols, 6);
	plt.scatter(list(range(1, 1+len(fX.variables['lambda'][()]))),fX.variables['lambda'][()] , color='b', label='X', s=0.2)
	plt.scatter(list(range(1, 1+len(fNA.variables['lambda'][()]))),fNA.variables['lambda'][()], color='r', label='NA', s=0.2)
	plt.xlabel(r'');plt.ylabel(r'lambda')
	##
	plt.subplot(nrows, ncols, 7);
	plt.scatter(theta, fX.variables['gbdrift'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['gbdrift'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'gbdrift')
	##
	plt.subplot(nrows, ncols, 8);
	plt.scatter(theta, fX.variables['gbdrift0'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['gbdrift0'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'gbdrift0')
	##
	plt.subplot(nrows, ncols, 9);
	plt.scatter(theta, fX.variables['cvdrift'][()] , color='b', label='X', s=0.1)
	plt.scatter(theta, fNA.variables['cvdrift'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'cvdrift')
	##
	plt.subplot(nrows, ncols, 10);
	l1=plt.scatter(theta, fX.variables['cvdrift0'][()] , color='b', label='X', s=0.1)
	l2=plt.scatter(theta, fNA.variables['cvdrift0'][()], color='r', label='NA', s=0.1)
	plt.xlabel(r'$\theta$');plt.ylabel(r'cvdrift0')
	##
	plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.97, wspace=0.27, hspace=0.3)
	fig.legend([l1,l2], ['X', 'NA'], loc = 'lower center', ncol=2)
	plt.savefig(stells+'_r'+str(rr)+'_geom.pdf', format='pdf')
	plt.close()
	return 0

currentPath = os.getcwd()
j=0; gammaValX=np.array([]);gammaValNA=np.array([]);
for stells in stellDesigns:
	print(stells)
	if not path.exists(stells):
		os.mkdir(stells)
	copyfile("gs2",stells+"/gs2")
	copymode("gs2",stells+"/gs2")
	i=0; gammaTempX=np.array([]); gammaTempNA=np.array([]);
	for desired_normalized_toroidal_flux in normalizedfluxvec:
		print("r = "+str(normalizedfluxvec[i]))
		rxText=stells+"r"+str(normalizedfluxvec[i]);
		copyfile("../gs2grids/grid"+rxText+".out",stells+"/grid"+rxText+".out")
		copyfile("../gs2grids/grid"+rxText+"NA.out",stells+"/grid"+rxText+"NA.out")
		copyfile("gs2Input.in",stells+"/gs2Input_"+rxText+".in")
		replace(stells+"/gs2Input_"+rxText+".in",' gridout_file = "./gridESTELLr0.01NA.out"',' gridout_file = "./grid'+rxText+'.out"')
		copyfile("gs2Input.in",stells+"/gs2Input_"+rxText+"NA.in")
		replace(stells+"/gs2Input_"+rxText+"NA.in",' gridout_file = "./gridESTELLr0.01NA.out"',' gridout_file = "./grid'+rxText+'NA.out"')
		os.chdir(stells)
		bashCommand = "mpirun -n 4 ./gs2 gs2Input_"+rxText+".in"
		#output = subprocess.call(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		bashCommand = "mpirun -n 4 ./gs2 gs2Input_"+rxText+"NA.in"
		#output = subprocess.call(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		removeGS2(os.getcwd())
		gammaTempX=np.append(gammaTempX,getgamma("gs2Input_"+rxText+".out.nc"))
		gammaTempNA=np.append(gammaTempNA,getgamma("gs2Input_"+rxText+"NA.out.nc"))
		eigenPlot("gs2Input_"+rxText+".out.nc")
		eigenPlot("gs2Input_"+rxText+"NA.out.nc")
		geomPlot(stells,desired_normalized_toroidal_flux,"gs2Input_"+rxText+".out.nc","gs2Input_"+rxText+"NA.out.nc")
		os.chdir(currentPath)
		i=i+1
	if j==0:
		gammaValX=gammaTempX;
		gammaValNA=gammaTempNA;
	else:
		gammaValX=np.vstack((gammaValX,gammaTempX))
		gammaValNA=np.vstack((gammaValNA,gammaTempNA))
	j=j+1

## Plot growth rates
matplotlib.rc('font', size=plotfontSize);
for j, desired_normalized_toroidal_flux in enumerate(normalizedfluxvec):
	fig, ax = plt.subplots(figsize=(figSize1,figSize2))
	ax.scatter( gammaValNA[:,j],gammaValX[:,j])
	for i, txt in enumerate(stellDesigns):
		ax.annotate(txt, (gammaValNA[i,j],gammaValX[i,j]), fontsize=annotatefontSize)
	plt.ylim(ymin=0)
	plt.xlim(xmin=0)
	ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='r', ls='--')
	plt.xlabel(r'Near-Axis $\gamma$');plt.ylabel(r'Design $\gamma$');
	plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
	plt.savefig('gammaStells_r'+str(desired_normalized_toroidal_flux)+'.pdf', format='pdf')
	plt.close()
