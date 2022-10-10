from qsc import Qsc
import numpy as np
from scipy.optimize import minimize, least_squares, basinhopping, dual_annealing, shgo
import matplotlib.pyplot as plt

MAXITER = 250
ftol=1e-12
nphi = 151
dofs_tol = 0.5

nfp = 5
rc0 = [ 2.1313393726129375,-0.23698092775628918,0.05927197268658798,-0.0136340764784787,0.003360235100734414,-0.0008156785280669119,0.00022045854879744765,-1.9726965562863992e-05,1.5017720742959991e-06,1.1837037880197848e-06 ]
zs0 = [ 0.0,-0.24126987827482477,0.059554147994235286,-0.013747615139447292,0.003502475703980723,-0.0009631411571683385,0.00022473086606087219,-2.1816082716913782e-05,9.517870262900427e-07,8.94483837092982e-07 ]
k0 = 0.959356919545
k1 = -0.461005949994
tau0 = -1.700645647791
# maximum J = 0.000445

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
    J = (curvature(k0, k1, stel.d_l_d_varphi*stel.varphi)-stel.curvature)**2 + (stel.torsion-tau0)**2
    return J
    # print(f'max J={np.max(J)}')
    # return np.max(J)

# basinhopping, dual_annealing, shgo
# bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) ,max((1-dofs_tol)*dof,(1+dofs_tol)*dof)] for dof in dofs]
# res = minimize(fun, dofs, args=(stel,), bounds=bounds, options={'maxiter': MAXITER, 'disp': True}, tol=ftol)
# res = dual_annealing(fun, bounds=bounds, args=(stel,), maxiter=MAXITER, no_local_search=False, x0=dofs)
# res = basinhopping(fun, dofs, niter=MAXITER, disp=True, minimizer_kwargs={'args': (stel,)}, niter_success=5)
bounds = [[min((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs],[max((1-dofs_tol)*dof,(1+dofs_tol)*dof) for dof in dofs]]
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
print(f'# maximum J = {np.max(fun(new_dofs, stel)):.6f}')

plt.figure()
plt.plot(stel.phi, stel.curvature, label='Final curvature')
plt.plot(stel.phi, curvature(new_k0, new_k1, stel.d_l_d_varphi*stel.varphi), label='Desired curvature')
plt.legend()

plt.figure()
plt.plot(stel.phi, stel.torsion, label='Final torsion')
plt.legend()

plt.show()

# stel.plot_axis()