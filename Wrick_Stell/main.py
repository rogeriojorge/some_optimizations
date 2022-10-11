from qsc import Qsc
import numpy as np
from scipy.optimize import minimize, least_squares, basinhopping, dual_annealing, shgo
import matplotlib.pyplot as plt

MAXITER = 20
ftol=1e-12
nphi = 131
dofs_tol = 0.5

nfp = 6
rc0 = [ 2.4652540319704785,-0.2633926859927981,0.0590156413367343,-0.0131886318833242,0.0031890141436049,-0.0007906939845513,0.0001940786419921,-0.0000493654654751,0.0000051734193022,0.0000006306836342,0.0000000097461855 ]
zs0 = [ 0.0000000000000000,-0.2674221321593625,0.0593783943439850,-0.0132679894780322,0.0032222356216268,-0.0008289220041744,0.0002261271147216,-0.0000519697941135,0.0000038744890018,0.0000005939766577,0.0000000121100939 ]
k0 = 1.0715450359579004
k1 = -0.4002320965907847
tau0 = -1.6432658375806168
# maximum J = 0.00006035

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
    J = (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2 + (stel.torsion-tau0)**2
    return J

bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs],[max((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs]]
res = least_squares(fun, dofs, args=(stel,), bounds=bounds, max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
new_dofs = res.x
new_rc = new_dofs[:stel.nfourier]
new_zs = np.concatenate(([zs0[0]],new_dofs[-stel.nfourier-2:-3]))
new_k0 = new_dofs[-3]
new_k1 = new_dofs[-2]
new_tau0 = new_dofs[-1]
stel = Qsc(rc=new_rc, zs=new_zs, nfp=nfp, nphi=nphi, etabar=1)
print('rc0 = [',','.join([f'{elem:.16f}' for elem in stel.rc]),']')
print('zs0 = [',','.join([f'{elem:.16f}' for elem in stel.zs]),']')
print(f'k0 = {new_k0:.16f}')
print(f'k1 = {new_k1:.16f}')
print(f'tau0 = {new_tau0:.16f}')
print(f'# maximum J = {np.max(fun(new_dofs, stel)):.8f}')

plt.figure()
plt.plot(stel.phi, stel.curvature, label='Final curvature')
plt.plot(stel.phi, curvature(new_k0, new_k1, stel.d_l_d_varphi*stel.varphi), label='Desired curvature')
plt.legend()

plt.figure()
plt.plot(stel.phi, stel.torsion, label='Final torsion')
plt.legend()

plt.show()

# stel.plot_axis()