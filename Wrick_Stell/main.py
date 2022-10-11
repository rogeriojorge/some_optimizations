from qsc import Qsc
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

MAXITER = 20
ftol=1e-12
nphi = 131
dofs_tol = 0.5

nfp = 6
rc0 = [ 2.4652780209255201,-0.2633865823288850,0.0590158050310366,-0.0131885435749914,0.0031892187225883,-0.0007915446581684,0.0001940186509923,-0.0000496663052723,0.0000053077922315,0.0000006169074966,0.0000000097509089 ]
zs0 = [ 0.0000000000000000,-0.2674241701809618,0.0593781083501999,-0.0132675229104405,0.0032222479675578,-0.0008284066267119,0.0002261042993884,-0.0000523756297605,0.0000040108521441,0.0000005835001905,0.0000000123539669 ]
k0 = 1.0715388924786766
k1 = -0.4002226450698441
tau0 = -1.6432567294024705
etabar = 0.7323016973383260
# maximum J = 0.00005832
# iota = -4.085232658164445
# iotaN = 1.9147673418355549
# int(tau*L/2pi) = -4.929215176052468

def curvature(k0, k1, s):
    return np.abs(k0+2*k1*np.cos(2*s))

stel = Qsc(rc=rc0, zs=zs0, nfp=nfp, nphi=nphi, etabar=etabar)
dofs = np.concatenate((stel.rc, stel.zs[1:], [k0, k1, tau0]))

def fun(dofs, stel):
    rc = dofs[:stel.nfourier]
    zs = np.concatenate(([zs0[0]],dofs[-stel.nfourier-2:-3]))
    k0 = dofs[-3]
    k1 = dofs[-2]
    tau0 = dofs[-1]
    stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar)
    J = (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2 + (stel.torsion-tau0)**2
    return J

def fun_elongation(etabar, stel):
    stel = Qsc(rc=stel.rc, zs=stel.zs, nfp=nfp, nphi=nphi, etabar=etabar)
    return stel.elongation**2
res_elongation = least_squares(fun_elongation, etabar, args=(stel,), max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
etabar = res_elongation.x[0]

bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs],[max((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs]]
res = least_squares(fun, dofs, args=(stel,), bounds=bounds, max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
new_dofs = res.x
new_rc = new_dofs[:stel.nfourier]
new_zs = np.concatenate(([zs0[0]],new_dofs[-stel.nfourier-2:-3]))
k0 = new_dofs[-3]
k1 = new_dofs[-2]
tau0 = new_dofs[-1]
stel = Qsc(rc=new_rc, zs=new_zs, nfp=nfp, nphi=nphi, etabar=etabar)
stel = Qsc(rc=stel.rc, zs=stel.zs, nfp=nfp, nphi=nphi, etabar=etabar)
dofs = np.concatenate((stel.rc, stel.zs[1:], [k0, k1, tau0]))
print(f'nfp = {nfp}')
print('rc0 = [',','.join([f'{elem:.16f}' for elem in stel.rc]),']')
print('zs0 = [',','.join([f'{elem:.16f}' for elem in stel.zs]),']')
print(f'k0 = {k0:.16f}')
print(f'k1 = {k1:.16f}')
print(f'tau0 = {tau0:.16f}')
print(f'etabar = {etabar:.16f}')
print(f'# maximum J = {np.max(fun(dofs, stel)):.8f}')
print(f'# iota = {stel.iota}')
print(f'# iotaN = {stel.iotaN}')
print(f'# int(tau*L/2pi) = {stel.axis_length*np.mean(stel.torsion)/(2*np.pi)}')

plt.figure()
plt.plot(stel.phi, stel.curvature, label='Final curvature')
plt.plot(stel.phi, curvature(k0, k1, stel.d_l_d_varphi*stel.varphi), label='Desired curvature')
plt.legend()

plt.figure()
plt.plot(stel.phi, stel.torsion, label='Final torsion')
plt.legend()

# plt.show()

# stel.plot_axis()