from qsc import Qsc
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

MAXITER = 10
ftol=1e-12
nphi = 131
dofs_tol = 0.3

nfp = 6
rc0 = [ 2.465291791214,-0.263382920320,0.059015974280,-0.013188484035,0.003189343968,-0.000792016256,0.000193995152,-0.000049828064,0.000005385980,0.000000609205,0.000000009755 ]
zs0 = [ 0.000000000000,-0.267425343068,0.059377909679,-0.013267277590,0.003222245044,-0.000828131438,0.000226082714,-0.000052599459,0.000004090212,0.000000577500,0.000000012491 ]
k0 = 1.071535923214
k1 = -0.400217124412
tau0 = -1.643251931600
etabar = 0.651905696276
# maximum J = 0.000057202853
# iota = -3.717273218667238
# iotaN = 2.282726781332762
# max elongation = 8.185444763137232
# int(tau*L/2pi) = -4.929226371916412

# nfp = 8
# rc0 = [ 3.0463315803822528,-0.3052146067547297,0.0608943284495521,-0.0133137742625560,0.0031672595875502,-0.0007874886951178,0.0001948585421500,-0.0000499484262787,0.0000102178411716,0.0000002491789280,0.0000000073963394 ]
# zs0 = [ 0.0000000000000000,-0.3087590584410459,0.0613015309722805,-0.0133990417474666,0.0031942213866334,-0.0008049348227515,0.0002155947657005,-0.0000569058263820,0.0000088966209120,0.0000002679732528,0.0000000119536205 ]
# k0 = 1.2496516400281528
# k1 = -0.3235356616787607
# tau0 = -1.5237337123230272
# etabar = 0.9502760143069352
# # maximum J = 0.00002019
# # iota = -3.348266361743124
# # iotaN = 4.651733638256876
# # max elongation = 3.9678607879763974
# # int(tau*L/2pi) = -6.097208642466513

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

# def fun_elongation(etabar, stel):
#     stel = Qsc(rc=stel.rc, zs=stel.zs, nfp=nfp, nphi=nphi, etabar=etabar)
#     return stel.elongation**2 + 10*stel.inv_L_grad_B**2
# res_elongation = least_squares(fun_elongation, etabar, args=(stel,), max_nfev=MAXITER, verbose=2, ftol=ftol, jac='3-point')
# etabar = res_elongation.x[0]
# stel = Qsc(rc=stel.rc, zs=stel.zs, nfp=nfp, nphi=nphi, etabar=etabar)

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
print('rc0 = [',','.join([f'{elem:.12f}' for elem in stel.rc]),']')
print('zs0 = [',','.join([f'{elem:.12f}' for elem in stel.zs]),']')
print(f'k0 = {k0:.12f}')
print(f'k1 = {k1:.12f}')
print(f'tau0 = {tau0:.12f}')
print(f'etabar = {etabar:.12f}')
print(f'# maximum J = {np.max(fun(dofs, stel)):.12f}')
print(f'# iota = {stel.iota}')
print(f'# iotaN = {stel.iotaN}')
print(f'# max elongation = {stel.max_elongation}')
print(f'# int(tau*L/2pi) = {stel.axis_length*np.mean(stel.torsion)/(2*np.pi)}')

plt.rcParams.update({'font.size': 12})

plt.figure()
plt.plot(stel.phi, stel.curvature, label='Axis curvature')
plt.plot(stel.phi, curvature(k0, k1, stel.d_l_d_varphi*stel.varphi), label=r'$\kappa_0 + 2 \kappa_1 \cos 2 \ell$')
plt.xlabel(r'$\phi$', fontsize=15)
plt.ylabel(r'$\kappa$', fontsize=15)
plt.legend(fontsize='12')
plt.tight_layout()
plt.savefig(f'nfp{nfp}_curvature.pdf')

plt.figure()
plt.plot(stel.phi, stel.torsion, label='Axis torsion')
plt.axhline(y=tau0, color='r', linestyle='-', label=r'$\tau_0$')
plt.xlabel(r'$\phi$', fontsize=15)
plt.ylabel(r'$\tau$', fontsize=15)
plt.legend(fontsize='12')
plt.tight_layout()
plt.savefig(f'nfp{nfp}_torsion.pdf')

# plt.show()

# stel.plot_axis(savefig=f'nfp{nfp}_axis3D.pdf', show=False)