from qsc import Qsc
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

MAXITER = 60

nphi = 51
k00=0.897
k10=-0.571
tau0=-1.710

rc0 = [1.7144, -0.221, 0.0679, -0.0184, 0.001]
zs0 = [0.0, 0.228, -0.0684, 0.0185, 0.001]

def curvature(k0, k1, s):
    return np.abs(k0+2*k1*np.cos(2*s))

stel = Qsc(rc=rc0, zs=zs0, nfp=1, nphi=nphi, etabar=1)
dofs = np.concatenate((stel.rc, stel.zs))### ADD THESE TO DOFS, [k00, k10, tau0]))

def fun(dofs, stel):
    rc = dofs[:-stel.nfourier]
    zs = dofs[-stel.nfourier:]
    stel = Qsc(rc=rc, zs=zs, nfp=1, nphi=nphi, etabar=1)
    return (curvature(k00, k10, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2 + (stel.torsion-tau0)**2

# res = minimize(fun, dofs, args=(stel,), method='BFGS', options={'maxiter': MAXITER}, tol=1e-12)
res = least_squares(fun, dofs, args=(stel,), max_nfev=MAXITER, verbose=2)
new_dofs = res.x
stel = Qsc(rc=new_dofs[:-stel.nfourier], zs=new_dofs[-stel.nfourier:], nfp=1, nphi=nphi, etabar=1)
print(f'Final rc = {stel.rc}')
print(f'Final zs = {stel.zs}')
plt.figure()
plt.plot(stel.phi, stel.curvature, label='Final curvature')
plt.plot(stel.phi, curvature(k00, k10, stel.d_l_d_varphi*stel.varphi), label='Desired curvature')
plt.legend()

plt.figure()
plt.plot(stel.phi, stel.torsion, label='Final torsion')
plt.legend()

plt.show()