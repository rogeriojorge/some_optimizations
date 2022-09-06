#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance


ncoils = 6
# Major radius for the initial circular coils:
R0 = 1.0
# Minor radius for the initial circular coils:
R1 = 0.5
# Number of Fourier modes describing each Cartesian component of each coil:
order = 5
# Weight on the curve lengths in the objective function:
ALPHA = 3e-5
# Weight on quadratic penalty in the objective function:
w = 0.01
# Threshhold for the coil-to-coil distance penalty in the objective function:
MIN_DIST = 0.1
# Weight on the coil-to-coil distance penalty term in the objective function:
BETA = 7e-3
AVERAGE_LENGTH_PER_COIL = 2.3
# Number of iterations to perform:
MAXITER = 400
# File for the desired boundary magnetic surface:
OUT_DIR = f"output_{AVERAGE_LENGTH_PER_COIL*ncoils}/"
os.makedirs(OUT_DIR, exist_ok=True)
filename = 'input.LandremanPaul2021_QH'


#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 100
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
s_full = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
curves = [c.curve for c in coils]

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jdist = MinimumDistance(curves, MIN_DIST)

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
#JF = Jf + ALPHA * sum(Jls) + BETA * Jdist
#JF = Jf + w * QuadraticPenalty(sum(Jls), 12) + BETA * Jdist
JF = Jf + w * sum(QuadraticPenalty(L, AVERAGE_LENGTH_PER_COIL) for L in Jls) + BETA * Jdist

bs.set_points(s.gamma().reshape((-1, 3)))
B_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jf.J()
    B_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
    print(f"J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}], ||∇J||={np.linalg.norm(grad):.3e}, B.n/|B|= {np.max(np.abs(B_n))/mean_AbsB}, ave(B.n)/|B| = {np.average(np.abs(B_n))/mean_AbsB}")
    return J, grad

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
x0 = Jf.x
dofs = JF.x
bs.set_points(s.gamma().reshape((-1, 3)))
Jf.x = x0 + (np.random.rand(len(x0))* 2 - 1) * 0.001
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 400}, tol=1e-15)

print("""Save the results""")
curves_to_vtk(curves, OUT_DIR + f"curves_opt_{AVERAGE_LENGTH_PER_COIL*ncoils}")
bs.save(OUT_DIR + f"biot_savart_opt_{AVERAGE_LENGTH_PER_COIL*ncoils}.json")

bs.set_points(s_full.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + f"surf_opt_{AVERAGE_LENGTH_PER_COIL*ncoils}", extra_data=pointData)

# bs.set_points(s.gamma().reshape((-1, 3)))
# pointData_halfperiod = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
# s.to_vtk(OUT_DIR + "surf_opt_halfperiod", extra_data=pointData_halfperiod)

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.gca(projection='3d')
xmax = np.max(curves[0].gamma())
xmin = np.min(curves[0].gamma())
for curve in curves:
    gamma = curve.gamma()
    xmax_temp = np.max(gamma)
    xmin_temp = np.min(gamma)
    if xmax_temp>xmax: xmax=xmax_temp
    if xmin_temp<xmin: xmin=xmin_temp
    plt.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2], linewidth=0.7, color='r')
ax.auto_scale_xyz([xmin, xmax], [xmin, xmax], [xmin, xmax])
plt.savefig(OUT_DIR+f'Coils_3Dplot_{AVERAGE_LENGTH_PER_COIL*ncoils}.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.close()