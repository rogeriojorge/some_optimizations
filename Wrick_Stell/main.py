from qsc import Qsc
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

MAXITER = 50
ftol=1e-10
nphi = 251
dofs_tol = 0.3

k0 = 0.900039274656
k1 = -0.571851975778
tau0=-1.710

# nfp = 2
# rc0 = [1.7144, 0.00001, -0.221, 0.00001, 0.0679, 0.00001, -0.0184, 0.00001, 0.00543, 0.00001, -0.00167, 0.00001]
# zs0 = [0.0,    0.00001, -0.228, 0.00001, 0.0684, 0.00001, -0.0185, 0.00001, 0.00545, 0.00001, -0.001684, 0.00001]

nfp = 4
rc0 = [ 1.7144,-0.22147399605870344,0.07055392115077455,-0.020251965989363872,0.005495642549876127,-0.0009069808141186342 ]
zs0 = [ 0.0,-0.22826284955080378,0.06615073613223654,-0.018100737033901686,0.004880640327493502,-0.0012186411065122835 ]

def curvature(k0, k1, s):
    return np.abs(k0+2*k1*np.cos(2*s))

stel = Qsc(rc=rc0, zs=zs0, nfp=nfp, nphi=nphi, etabar=1)
dofs = np.concatenate((stel.rc[1:], stel.zs[1:], [k0, k1, tau0]))

def fun(dofs, stel):
    rc = np.concatenate(([rc0[0]],dofs[:stel.nfourier-1]))
    zs = np.concatenate(([zs0[0]],dofs[-stel.nfourier-2:-3]))
    k0 = dofs[-3]
    k1 = dofs[-2]
    tau0 = dofs[-1]
    stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=1)
    # return (stel.torsion-tau0)**2
    return (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2

# res = minimize(fun, dofs, args=(stel,), method='BFGS', options={'maxiter': MAXITER, 'disp': True}, tol=ftol)
bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs],[max((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs]]
res = least_squares(fun, dofs, args=(stel,), bounds=bounds, max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
new_dofs = res.x
new_rc = np.concatenate(([rc0[0]],new_dofs[:stel.nfourier-1]))
new_zs = np.concatenate(([zs0[0]],new_dofs[-stel.nfourier-2:-3]))
new_k0 = new_dofs[-3]
new_k1 = new_dofs[-2]
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