from qsc import Qsc
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

MAXITER = 500
ftol=1e-9
nphi = 999
dofs_tol = 0.3

nfp = 4
rc0 = [ 1.7126853376810032,-0.23790710824527608,0.06842525009102422,-0.016865464409359168,0.004365261171206435,-0.0011259676920913268 ]
zs0 = [ 0.0,-0.23089013235038983,0.06745921235340123,-0.018515425548769684,0.0053826861878741765,-0.0010089567824379557 ]
k0 = 0.936816920741
k1 = -0.513343671327
tau0 = -1.670373930731

def curvature(k0, k1, s):
    return np.abs(k0+2*k1*np.cos(2*s))

stel = Qsc(rc=rc0, zs=zs0, nfp=nfp, nphi=nphi, etabar=1)
dofs = np.concatenate((stel.rc, stel.zs[1:], [k0, k1, tau0]))

def fun(dofs, stel):
    rc = dofs[:stel.nfourier]
    zs = np.concatenate(([zs0[0]],dofs[-stel.nfourier-2:-3]))
    k0 = dofs[-3]
    k1 = dofs[-2]
    tau0 = dofs[-1]
    stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=1)
    # return (stel.torsion-tau0)**2
    # return (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2
    return (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2 + (stel.torsion-tau0)**2

bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs],[max((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs]]
# res = minimize(fun, dofs, args=(stel,), method='BFGS', options={'maxiter': MAXITER, 'disp': True}, tol=ftol)
res = least_squares(fun, dofs, args=(stel,), bounds=bounds, max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
new_dofs = res.x
new_rc = new_dofs[:stel.nfourier]
new_zs = np.concatenate(([zs0[0]],new_dofs[-stel.nfourier-2:-3]))
new_k0 = new_dofs[-3]
new_k1 = new_dofs[-2]
new_tau0 = new_dofs[-1]
stel = Qsc(rc=new_rc, zs=new_zs, nfp=nfp, nphi=nphi, etabar=1)
print('rc0 = [',','.join([str(elem) for elem in stel.rc]),']')
print('zs0 = [',','.join([str(elem) for elem in stel.zs]),']')
print(f'k0 = {new_k0:.12f}')
print(f'k1 = {new_k1:.12f}')
print(f'tau0 = {new_tau0:.12f}')

plt.figure()
plt.plot(stel.phi, stel.curvature, label='Final curvature')
plt.plot(stel.phi, curvature(new_k0, new_k1, stel.d_l_d_varphi*stel.varphi), label='Desired curvature')
plt.legend()

plt.figure()
plt.plot(stel.phi, stel.torsion, label='Final torsion')
plt.legend()

plt.show()

# stel.plot_axis()