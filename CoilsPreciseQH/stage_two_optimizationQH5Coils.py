#!/usr/bin/env python
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation

ncoils = 4
R0 = 1.0
R1 = 0.5
order = 9
LENGTH_WEIGHT = 0.01
AVERAGE_LENGTH_PER_COIL = 3.1
CC_THRESHOLD = 0.1
CC_WEIGHT = 5e-1
CURVATURE_THRESHOLD = 10.
CURVATURE_WEIGHT = 1e-7
MSC_THRESHOLD = 10
MSC_WEIGHT = 1e-7
MAXITER = 10000

OUT_DIR = f"output_{AVERAGE_LENGTH_PER_COIL*ncoils}/"
os.makedirs(OUT_DIR, exist_ok=True)
filename = 'input.LandremanPaul2021_QH'

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 50
ntheta = 40
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
s_full = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=5*nphi, ntheta=3*ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1)*1e5 for i in range(ncoils)]
# base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((5*nphi, 3*ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

bs.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
# Jcsdist = CurveSurfaceDistance(curves, s, 12)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
# Jals = [ArclengthVariation(c) for c in base_curves]

JF = Jf \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), AVERAGE_LENGTH_PER_COIL*ncoils) \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
    # + ARCLENGTH_WEIGHT * sum(Jals)

def fun(dofs):
    start = time.time()
    JF.x = dofs
    J = JF.J()
    jf = Jf.J()
    grad = JF.dJ()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB = np.mean(bs.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    outstr += f", Max B·n={MaxBdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    # outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
    # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    # outstr += f", ⟨B·n⟩={BdotN:.1e}"
    stop = time.time()
    # outstr += f", duration={stop-start}"
    print(outstr)
    with open(OUT_DIR + f'output_{AVERAGE_LENGTH_PER_COIL*ncoils}.txt', "a") as myfile:
        myfile.write(outstr)
    return J, grad


print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
dofs = JF.x
bs.set_points(s.gamma().reshape((-1, 3)))
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)

print("""Save the results""")
curves_to_vtk(curves, OUT_DIR + f"curves_opt_{AVERAGE_LENGTH_PER_COIL*ncoils}")
bs.save(OUT_DIR + f"biot_savart_opt_{AVERAGE_LENGTH_PER_COIL*ncoils}.json")

bs.set_points(s_full.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((5*nphi, 3*ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + f"surf_opt_{AVERAGE_LENGTH_PER_COIL*ncoils}", extra_data=pointData)

# fig = plt.figure()
# fig.patch.set_facecolor('white')
# ax = fig.gca(projection='3d')
# xmax = np.max(curves[0].gamma())
# xmin = np.min(curves[0].gamma())
# for curve in curves:
#     gamma = curve.gamma()
#     xmax_temp = np.max(gamma)
#     xmin_temp = np.min(gamma)
#     if xmax_temp>xmax: xmax=xmax_temp
#     if xmin_temp<xmin: xmin=xmin_temp
#     plt.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2], linewidth=0.7, color='r')
# ax.auto_scale_xyz([xmin, xmax], [xmin, xmax], [xmin, xmax])
# plt.savefig(OUT_DIR+f'Coils_3Dplot_{AVERAGE_LENGTH_PER_COIL*ncoils}.pdf', bbox_inches = 'tight', pad_inches = 0)
# plt.close()