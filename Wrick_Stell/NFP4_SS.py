from qsc import Qsc
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

MAXITER = 100
ftol=1e-9
nphi = 251
dofs_tol = 0.1

k0 = 0.9
k1 = -0.57
tau0= -1.71

nfp = 4
rc0 = [ 1.71,-0.22487727796072757,0.0709705498536279,-0.01973351148766593,0.005161657646859404,-0.0008640677780327807 ]
zs0 = [ 0.0,-0.22779716326837915,0.0659025353678517,-0.01810578853754581,0.0046840950871436795,-0.0011827270143195336 ]

def curvature(k0, k1, s):
    return np.abs(k0+2*k1*np.cos(2*s))

stel = Qsc(rc=rc0, zs=zs0, nfp=nfp, nphi=nphi, etabar=1)
dofs = np.concatenate((stel.rc[1:], stel.zs[1:], [tau0]))#, [k0, k1, tau0]))

def fun(dofs, stel):
    rc = np.concatenate(([rc0[0]],dofs[:stel.nfourier-1]))
    zs = np.concatenate(([zs0[0]],dofs[-stel.nfourier:-1]))#[-stel.nfourier-2:-2]))
    # k0 = dofs[-3]
    # k1 = dofs[-2]
    tau0 = dofs[-1]
    stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=1)
    # return (stel.torsion-tau0)**2
    return (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2
    # return (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2 + 1e-3*(stel.torsion-tau0)**2

# res = minimize(fun, dofs, args=(stel,), method='BFGS', options={'maxiter': MAXITER, 'disp': True}, tol=ftol)
bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs],[max((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs]]
res = least_squares(fun, dofs, args=(stel,), bounds=bounds, max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
new_dofs = res.x
new_rc = np.concatenate(([rc0[0]],new_dofs[:stel.nfourier-1]))
new_zs = np.concatenate(([zs0[0]],dofs[-stel.nfourier:-1]))#[-stel.nfourier-2:-2]))
new_k0 = k0#new_dofs[-3]
new_k1 = k1#new_dofs[-2]
new_tau0 = new_dofs[-1]
stel = Qsc(rc=new_rc, zs=new_zs, nfp=nfp, nphi=nphi, etabar=1)
print('rc0 = [',','.join([str(elem) for elem in stel.rc]),']')
print('zs0 = [',','.join([str(elem) for elem in stel.zs]),']')
print(f'Final k0 = {new_k0:.12f}')
print(f'Final k1 = {new_k1:.12f}')
print(f'Final tau0 = {new_tau0:.12f}')

plt.figure()
plt.plot(stel.phi, stel.curvature, label='Final curvature')
plt.plot(stel.phi, curvature(new_k0, new_k1, stel.d_l_d_varphi*stel.varphi), label='Desired curvature')
plt.legend()

plt.figure()
plt.plot(stel.phi, stel.torsion, label='Final torsion')
plt.legend()

plt.show()

# stel.plot_axis()