from qsc import Qsc
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

MAXITER = 20
ftol=1e-12
nphi = 131
dofs_tol = 0.5

nfp = 6
rc0 = [ 2.4652838171453677,-0.2633850188110334,0.0590159150025034,-0.0131884936201682,0.0031892753390715,-0.0007917481896376,0.0001940060038178,-0.0000497356268301,0.0000053407214459,0.0000006136618583,0.0000000097528044 ]
zs0 = [ 0.0000000000000000,-0.2674247667788165,0.0593778364925769,-0.0132675685336399,0.0032221629520219,-0.0008283252973367,0.0002260822865508,-0.0000524749875605,0.0000040436830533,0.0000005808436264,0.0000000124078591 ]
k0 = 1.0715375778875065
k1 = -0.4002203366209362
tau0 = -1.6432546557836538
etabar = 0.6518937513786018
# maximum J = 0.00005784
# iota = -3.7170011672617096
# iotaN = 2.2829988327382904
# max elongation = 8.185517937377234
# int(tau*L/2pi) = -4.929221716339454

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
    return stel.elongation**2 + 10*stel.inv_L_grad_B**2
res_elongation = least_squares(fun_elongation, etabar, args=(stel,), max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
etabar = res_elongation.x[0]
stel = Qsc(rc=stel.rc, zs=stel.zs, nfp=nfp, nphi=nphi, etabar=etabar)

bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs],[max((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs]]
res = least_squares(fun, dofs, args=(stel,), bounds=bounds, max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
dofs = res.x
rc = dofs[:stel.nfourier]
zs = np.concatenate(([zs0[0]],dofs[-stel.nfourier-2:-3]))
k0 = dofs[-3]
k1 = dofs[-2]
tau0 = dofs[-1]
stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar)
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
print(f'# max elongation = {stel.max_elongation}')
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