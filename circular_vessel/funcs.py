from scipy.io import netcdf
import numpy as np
from numpy import loadtxt, concatenate
from numpy import asarray,transpose,savetxt
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from pandas import read_csv
from numpy import zeros,asarray,cos,sin,vstack,linspace,meshgrid,interp,cos,sin,ceil,sqrt
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from scipy.io import netcdf
from matplotlib import cm
from simsopt.mhd import Vmec
import time
import os
from simsopt.field import particles_to_vtk, compute_fieldlines

def output2regcoil(name,targetValue,R0_coil,a_coil):
    with open(f'regcoil_in.{name}', "w") as txt_file:
        txt_file.write('&regcoil_nml\n')
        txt_file.write(' general_option=5\n')
        txt_file.write(' nlambda=300\n')
        txt_file.write(' lambda_min = 1e-19\n')
        txt_file.write(' lambda_max = 1e-6\n')
        txt_file.write(' target_option = "max_Bnormal"\n')
        txt_file.write(f' target_value = {targetValue}\n')
        txt_file.write(' ntheta_plasma=64\n')
        txt_file.write(' ntheta_coil  =64\n')
        txt_file.write(' nzeta_plasma =64\n')
        txt_file.write(' nzeta_coil   =64\n')
        txt_file.write(' mpol_potential = 8\n')
        txt_file.write(' ntor_potential = 8\n')
        txt_file.write(' geometry_option_plasma = 2\n')
        txt_file.write(f" wout_filename = 'wout_{name}.nc'\n")
        # txt_file.write(' geometry_option_coil=2\n');coilSeparation=0.05
        # txt_file.write(f' separation = {coilSeparation}\n')
        # txt_file.write(f" nescin_filename = '{name}_nescin.out'\n")
        txt_file.write(' geometry_option_coil=1\n')
        txt_file.write(f' R0_coil={R0_coil}\n')
        txt_file.write(f' a_coil={a_coil}\n')
        txt_file.write(' symmetry_option=1\n')
        txt_file.write("/\n")

def output2nescoil(nescinfilename, vmecFile, R0_coil, a_coil):
    f = netcdf.netcdf_file(vmecFile,'r',mmap=False)
    bsubvmnc = f.variables['bsubvmnc'][()]
    nfp = f.variables['nfp'][()]
    f.close()
    curpol = 2*np.pi/nfp*(1.5*bsubvmnc[-1,0]-0.5*bsubvmnc[-2,0])
    with open(nescinfilename, "w") as txt_file:
        txt_file.write('------ Plasma information from VMEC ----\n')
        txt_file.write('np     iota_edge       phip_edge       curpol\n')
        txt_file.write(f'{nfp}  0.000000000000E+00  0.000000000000E+00  {curpol}\n')
        txt_file.write('------ Current Surface -----\n')
        txt_file.write('Number of fourier modes in table\n')
        txt_file.write(f'        2\n')
        txt_file.write('Table of fourier coefficients\n')
        txt_file.write('m,n,crc2,czs2,crs2,czc2\n')
        txt_file.write(f'      0     0  {R0_coil}  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00\n')
        txt_file.write(f'      1     0  {a_coil}  {a_coil}  0.000000000000E+00  0.000000000000E+00\n')
        txt_file.write('\n')
   
## Fourier coefficients calculation
from numpy import pi
# cos coefficient calculation.
def a(f, n, accuracy = 50, L=pi):
	from numpy import linspace,sum,cos
	a, b = 0, 2*L
	dx = (b - a) / accuracy
	integration = 0
	x=linspace(a, b, accuracy)
	integration=sum(f(x) * cos((n * pi * x) / L))
	# for x in np.linspace(a, b, accuracy):
	# 	integration += f(x) * np.cos((n * np.pi * x) / L)
	integration *= dx
	if n==0: integration=integration/2
	return (1 / L) * integration
# sin coefficient calculation.
def b(f, n, accuracy = 50, L=pi):
	from numpy import linspace,sum,sin
	a, b = 0, 2*L
	dx = (b - a) / accuracy
	integration = 0
	x=linspace(a, b, accuracy)
	integration=sum(f(x) * sin((n * pi * x) / L))
	# for x in np.linspace(a, b, accuracy):
	# 	integration += f(x) * np.sin((n * np.pi * x) / L)
	integration *= dx
	return (1 / L) * integration
# Fourier series.   
def Sf(f, x, n = 15, accuracy=50, L=pi):
	from numpy import zeros,cos,sin,arange,size
	a0 = a(f, 0, accuracy)
	sum = zeros(size(x))
	for i in arange(1, n + 1):
		sum += ((a(f, i, accuracy) * cos((i * pi * x) / L)) + (b(f, i, accuracy) * sin((i * pi * x) / L)))
	return (a0) + sum  

def importCoils(file):
	allCoils = read_csv(file)
	allCoilsValues = allCoils.values
	coilN=0
	coilPosN=0
	coilPos=[[],[]]
	for nVals in range(len(allCoilsValues)-2):
		listVals=allCoilsValues[nVals+2][0]
		vals=listVals.split()
		try:
			floatVals = [float(nVals) for nVals in vals][0:3]
			coilPos[coilN].append(floatVals)
			coilPosN=coilPosN+1
		except:
			try:
				floatVals = [float(nVals) for nVals in vals[0:3]][0:3]
				coilPos[coilN].append(floatVals)
				coilN=coilN+1
				coilPos.append([])
			except:
				break
	current=allCoilsValues[6][0].split()
	current=float(current[3])
	coilPos=coilPos[:-2]
	return coilPos, current

def getCoilsFourier(coil0,nPCoils,accuracy):
	from numpy import linspace,concatenate,zeros,sqrt,array
	from scipy import interpolate
	xArr=[i[0] for i in coil0]
	yArr=[i[1] for i in coil0]
	zArr=[i[2] for i in coil0]
	# xthetaArr=linspace(0,2*pi,len(xArr))
	# ythetaArr=linspace(0,2*pi,len(yArr))
	# zthetaArr=linspace(0,2*pi,len(zArr))
	# xf  = interpolate.interp1d(xthetaArr,xArr, kind='cubic')
	# yf  = interpolate.interp1d(ythetaArr,yArr, kind='cubic')
	# zf  = interpolate.interp1d(zthetaArr,zArr, kind='cubic')
	### Uniform arclength along the coil reparametrization
	### independent variable ivariable = sum_i[sqrt(dx_i^2+dy_i^2+dz_i^2)]
	### Renormalize ivariable - 0 to 2pi
	### xf  = interpolate.interp1d(ivariable,xArr, kind='cubic')
	L = [0 for i in range(len(xArr))]
	for itheta in range(1,len(xArr)):
		dx = xArr[itheta]-xArr[itheta-1]
		dy = yArr[itheta]-yArr[itheta-1]
		dz = zArr[itheta]-zArr[itheta-1]
		dL = sqrt(dx*dx+dy*dy+dz*dz)
		L[itheta]=L[itheta-1]+dL
	L=(1+1e-12)*array(L)*2*pi/L[-1]
	xf  = interpolate.interp1d(L,xArr, kind='cubic')
	yf  = interpolate.interp1d(L,yArr, kind='cubic')
	zf  = interpolate.interp1d(L,zArr, kind='cubic')
	coilsFourierXS=[b(xf,j,accuracy) for j in range(nPCoils)]
	coilsFourierXC=[a(xf,j,accuracy) for j in range(nPCoils)]
	coilsFourierYS=[b(yf,j,accuracy) for j in range(nPCoils)]
	coilsFourierYC=[a(yf,j,accuracy) for j in range(nPCoils)]
	coilsFourierZS=[b(zf,j,accuracy) for j in range(nPCoils)]
	coilsFourierZC=[a(zf,j,accuracy) for j in range(nPCoils)]
	return concatenate([coilsFourierXS,coilsFourierXC,coilsFourierYS,coilsFourierYC,coilsFourierZS,coilsFourierZC])

def cartesianCoils2fourier(coilPos,outputFile,nPCoils=20,accuracy=500):
	num_cores = cpu_count()
	coilsFourier = Parallel(n_jobs=num_cores)(delayed(getCoilsFourier)(coil0,nPCoils,accuracy) for coil0 in coilPos)
	#coilsFourier = [getCoilsFourier(coil0,nPCoils,accuracy) for coil0 in coilPos]
	coilsFourier = asarray(coilsFourier)
	coilsFourier = coilsFourier.reshape(6*len(coilPos),nPCoils)
	coilsFourier=transpose(coilsFourier)
	#coilsFourier = [np.ndarray.flatten(np.asarray(coilsF)) for coilsF in coilsFourier]
	savetxt(outputFile,coilsFourier, delimiter=',')
	return coilsFourier

def getFourierCurve(outputFile,current,ppp=10):
	coil_data = loadtxt(outputFile, delimiter=',')
	Nt_coils=len(coil_data)-1
	num_coils = int(len(coil_data[0])/6)
	coils = [CurveXYZFourier(Nt_coils*ppp, Nt_coils) for i in range(num_coils)]
	for ic in range(num_coils):
		dofs = coils[ic].get_dofs().reshape(3,Nt_coils*2+1)
		dofs[0][0] = coil_data[0, 6*ic + 1]
		dofs[1][0] = coil_data[0, 6*ic + 3]
		dofs[2][0] = coil_data[0, 6*ic + 5]
		for io in range(0, Nt_coils):
			dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
			dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
			dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
			dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
			dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
			dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
		coils[ic].set_dofs(concatenate(dofs).reshape(3*(Nt_coils*2+1)))
	currents = [current for i in range(num_coils)]
	return (coils, currents)

def export_coils(coils,filename,currents,NFP):
	from numpy import c_,savetxt,ones,append
	with open(filename, "w") as txt_file:
		txt_file.write("periods "+str(NFP)+"\n")
		txt_file.write("begin filament\n")
		txt_file.write("mirror NIL\n")
	for count,coil in enumerate(coils):
		with open(filename, "ab") as txt_file:
			coilE=coil.gamma()
			coilE=c_[coilE, currents[count]*ones(len(coilE))]
			coilLast=coilE[-1]
			coilE=coilE[:-1, :]
			savetxt(txt_file, coilE, fmt='%.8e')
			coilLast[3]=0.0
			coilLast=append(coilLast,"1")
			coilLast=append(coilLast,"Modular")
			savetxt(txt_file, coilLast, fmt='%.10s', newline=" ")
			txt_file.write(b"\n")
	with open(filename, "ab") as txt_file:
		txt_file.write(b"end\n")


def plot_stellarator(outName, qvfilename, coils, nfp, axis, qscfile=None):
	## Get Coils
	gamma = coils[0].gamma()
	N = gamma.shape[0]
	l = len(coils)
	data = zeros((l*(N+1), 3))
	for i in range(l):
		data[(i*(N+1)):((i+1)*(N+1)-1), :] = coils[i].gamma()
		data[((i+1)*(N+1)-1), :] = coils[i].gamma()[0, :]
	## Get Axis
	if axis is not None:
		N = axis.gamma().shape[0]
		ma_ = zeros((nfp*N+1, 3))
		ma0 = axis.gamma().copy()
		theta = 2*pi/nfp
		rotmat = asarray([
			[cos(theta), -sin(theta), 0],
			[sin(theta), cos(theta), 0],
			[0, 0, 1]]).T

		for i in range(nfp):
			ma_[(i*N):(((i+1)*N)), :] = ma0
			ma0 = ma0 @ rotmat
		ma_[-1, :] = axis.gamma()[0, :]
		data = vstack((data, ma_))
	## Plot coils and axis
	fig=plt.figure(figsize=(7,7))
	fig.patch.set_facecolor('white')
	ax = fig.gca(projection='3d')
	maxR=max(data[:,0])
	ax.scatter(data[:,0], data[:,1], data[:,2], '.-',s=0.3)
	ax.auto_scale_xyz([-maxR,maxR],[-maxR,maxR],[-maxR,maxR])
	## Plot surface
	if qscfile is not None:
		N_theta = 40
		N_phi = 150
		f = netcdf.netcdf_file(qscfile,mode='r',mmap=False)
		B0 = f.variables['B0'][()]
		r = f.variables['r'][()]
		eta_bar = f.variables['eta_bar'][()]
		mpol = f.variables['mpol'][()]
		ntor = f.variables['ntor'][()]
		RBC = f.variables['RBC'][()]
		RBS = f.variables['RBS'][()]
		ZBC = f.variables['ZBC'][()]
		ZBS = f.variables['ZBS'][()]
		theta1D = linspace(0,2*pi,N_theta)
		phi1D = linspace(0,2*pi,N_phi)
		phi2D,theta2D = meshgrid(phi1D,theta1D)
		R = zeros((N_theta,N_phi))
		z = zeros((N_theta,N_phi))
		for m in range(mpol+1):
			for jn in range(ntor*2+1):
				n = jn-ntor
				angle = m * theta2D - nfp * n * phi2D
				sinangle = sin(angle)
				cosangle = cos(angle)
				R += RBC[m,jn] * cosangle + RBS[m,jn] * sinangle
				z += ZBC[m,jn] * cosangle + ZBS[m,jn] * sinangle
		x = R * cos(phi2D)
		y = R * sin(phi2D)
		B = B0 * (1 + r * eta_bar * cos(theta2D))
		def toString(ncVar):
			temp = [c.decode('UTF-8') for c in ncVar]
			return (''.join(temp)).strip()
		order_r_option = toString(f.variables["order_r_option"][()])
		order_r_squared = (order_r_option != 'r1' and order_r_option != 'r1_compute_B2')
		if order_r_squared:
			B20 = f.variables['B20'][()]
			B2s = f.variables['B2s'][()]
			B2c = f.variables['B2c'][()]
			phi = f.variables['phi'][()]
			B20_interpolated = interp(phi1D,phi,B20,period=2*pi/nfp)
			B20_2D = repmat(B20_interpolated,N_theta,1)
			B += r * r * (B2s * sin(2*theta2D) + B2c * cos(2*theta2D) + B20_2D)
		# Rescale to lie in [0,1]:
		B_rescaled = (B - B.min()) / (B.max() - B.min())
		#ax.set_aspect('equal')
		ax.plot_surface(x, y, z, facecolors = cm.viridis(B_rescaled), rstride=1, cstride=1, antialiased=False)
	# Hide grid lines
	ax.grid(False)
	# Hide axes ticks
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	plt.axis('off')
	print("Saving coils PDF")
	plt.savefig(outName+qvfilename+'.pdf', bbox_inches = 'tight', pad_inches = 0)
	#plt.show()


def trace_fieldlines(bfield, R0, Z0, nfp, tmax_fl, nfieldlines, comm=None):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-12,# comm=comm,
        phis=phis, stopping_criteria=[])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm is None or comm.rank == 0:
        particles_to_vtk(fieldlines_tys, os.path.join(f'fieldlines_optimized_coils'))
        # plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(OUT_DIR, f'poincare_fieldline_optimized_coils.png'), dpi=150)
    return fieldlines_tys, fieldlines_phi_hits, phis

def create_poincare(bs, vmec_file: Vmec, ntheta_VMEC = 300, nfieldlines=4, tmax_fl=300):
    vmec = Vmec(vmec_file)
    nfp = vmec.wout.nfp
    nzeta=4
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    theta = np.linspace(0,2*np.pi,num=ntheta_VMEC)
    iradii = np.linspace(0,vmec.wout.ns-1,num=nfieldlines).round()
    iradii = [int(i) for i in iradii]
    R = np.zeros((nzeta,nfieldlines,ntheta_VMEC))
    Z = np.zeros((nzeta,nfieldlines,ntheta_VMEC))
    phis = zeta
    for itheta in range(ntheta_VMEC):
        for izeta in range(nzeta):
            for iradius in range(nfieldlines):
                for imode, xnn in enumerate(vmec.wout.xn):
                    angle = vmec.wout.xm[imode]*theta[itheta] - xnn*zeta[izeta]
                    R[izeta,iradius,itheta] += vmec.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
                    Z[izeta,iradius,itheta] += vmec.wout.zmns[imode, iradii[iradius]]*np.sin(angle)

    R0 = R[0,:,0]
    Z0 = Z[0,:,0]
    print('Beginning field line tracing')
    fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs, R0, Z0, nfp, tmax_fl, nfieldlines, comm=None)
    print('Creating Poincare plot R, Z')
    r = []
    z = []
    for izeta in range(len(phis)):
        r_2D = []
        z_2D = []
        for iradius in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[iradius][-1, 1] < 0
            data_this_phi = fieldlines_phi_hits[iradius][np.where(fieldlines_phi_hits[iradius][:, 1] == izeta)[0], :]
            if data_this_phi.size == 0:
                print(f'No Poincare data for iradius={iradius} and izeta={izeta}')
                continue
            r_2D.append(np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2))
            z_2D.append(data_this_phi[:, 4])
        r.append(r_2D)
        z.append(z_2D)
    r = np.array(r, dtype=object)
    z = np.array(z, dtype=object)
    print('Plotting Poincare plot')
    nrowcol = int(ceil(sqrt(len(phis))))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(12, 8))
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
        axs[row, col].set_xlabel("$R$")
        axs[row, col].set_ylabel("$Z$")
        axs[row, col].set_aspect('equal')
        axs[row, col].tick_params(direction="in")
        for j in range(nfieldlines):
            if j== 0 and i == 0:
                legend1 = 'Poincare plot'
                legend2 = 'VMEC QFM'
            else:
                legend1 = legend2 = '_nolegend_'
            try: axs[row, col].scatter(r[i][j], z[i][j], marker='o', s=0.7, linewidths=0, c='b', label = legend1)
            except Exception as e: print(e, i, j)
            axs[row, col].scatter(R[i,j], Z[i,j], marker='o', s=0.7, linewidths=0, c='r', label = legend2)
    plt.tight_layout()
    plt.savefig(f'CNT_poincare_QFM_fieldline_all.pdf', bbox_inches = 'tight', pad_inches = 0)
