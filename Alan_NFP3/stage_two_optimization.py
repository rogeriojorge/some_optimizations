#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
from simsopt import load

# Select one of Alan's configurations
# nfp = 3
# R0 = 1.0
# R1 = 0.8
# order = 8
# initial_current = 1.67
# ## Initial weights
# # use_previous_coils = False
# # ncoils = 9
# # LENGTH_WEIGHT = 1e-3
# # LENGTH_THRESHOLD_ARRAY = [1.8]
# # CC_THRESHOLD = 0.04
# # CC_WEIGHT = 10
# # CURVATURE_THRESHOLD = 10
# # CURVATURE_WEIGHT = 1e-5
# # MSC_THRESHOLD = 10
# # MSC_WEIGHT = 1e-5
# # MAXITER = 250
# ## Final weights
# use_previous_coils = True
# ncoils = 9
# LENGTH_WEIGHT = 1e-9
# LENGTH_THRESHOLD_ARRAY = [3.2]
# CC_THRESHOLD = 0.05
# CC_WEIGHT = 10
# CURVATURE_THRESHOLD = 30
# CURVATURE_WEIGHT = 1e-9
# MSC_THRESHOLD = 30
# MSC_WEIGHT = 1e-9
# MAXITER = 4500

# nfp = 2
# R0 = 1.0
# R1 = 0.7
# order = 7
# initial_current = 2.29
# ## Initial weights
# # use_previous_coils = False
# # ncoils = 8
# # LENGTH_WEIGHT = 1e-7
# # LENGTH_THRESHOLD_ARRAY = [2.2]
# # CC_THRESHOLD = 0.06
# # CC_WEIGHT = 10
# # CURVATURE_THRESHOLD = 15
# # CURVATURE_WEIGHT = 1e-6
# # MSC_THRESHOLD = 15
# # MSC_WEIGHT = 1e-6
# # MAXITER = 60
# ## Final weights
# use_previous_coils = True
# ncoils = 8
# LENGTH_WEIGHT = 1e-8
# LENGTH_THRESHOLD_ARRAY = [3.4]
# CC_THRESHOLD = 0.05
# CC_WEIGHT = 10
# CURVATURE_THRESHOLD = 30
# CURVATURE_WEIGHT = 1e-8
# MSC_THRESHOLD = 30
# MSC_WEIGHT = 1e-8
# MAXITER = 4500

nfp = 1
R0 = 1.0
R1 = 0.7
order = 7
initial_current = 6.76
## Initial weights
# use_previous_coils = False
# ncoils = 18
# LENGTH_WEIGHT = 1e-4
# LENGTH_THRESHOLD_ARRAY = [2.1]
# CC_THRESHOLD = 0.05
# CC_WEIGHT = 10
# CURVATURE_THRESHOLD = 15
# CURVATURE_WEIGHT = 1e-5
# MSC_THRESHOLD = 15
# MSC_WEIGHT = 1e-5
# MAXITER = 60
## Final weights
use_previous_coils = True
ncoils = 18
LENGTH_WEIGHT = 1e-9
LENGTH_THRESHOLD_ARRAY = [3.4]
CC_THRESHOLD = 0.05
CC_WEIGHT = 10
CURVATURE_THRESHOLD = 30
CURVATURE_WEIGHT = 1e-8
MSC_THRESHOLD = 30
MSC_WEIGHT = 1e-8
MAXITER = 4000


# Weight on the curve lengths in the objective funcion:
LENGTH_CON_WEIGHT = 0.1

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent).resolve()
filename = TEST_DIR / f'input.nfp{nfp}_Alan'

# Directory for output
OUT_DIR = f"./output_nfp{nfp}/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi_half = 80
ntheta_half = 40
nphi_full= 256
ntheta_full = 128
s_half = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi_half, ntheta=ntheta_half)
s_full = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi_full, ntheta=ntheta_full)

# Create the initial coils
if use_previous_coils:
    bs_temporary = load(os.path.join(OUT_DIR, 'biot_savart_opt.json'))
    base_curves = [bs_temporary.coils[i]._curve for i in range(ncoils)]
    base_currents = [bs_temporary.coils[i]._current for i in range(ncoils)]
else:
    base_curves = create_equally_spaced_curves(ncoils, s_half.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    base_currents = [Current(initial_current)*1e5 for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s_half.nfp, True)
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
bs.set_points(s_full.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((nphi_full, ntheta_full, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

for LENGTH_THRESHOLD in LENGTH_THRESHOLD_ARRAY:

    bs.set_points(s_half.gamma().reshape((-1, 3)))

    # Define the individual terms objective function:
    Jf = SquaredFlux(s_half, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    # Jcsdist = CurveSurfaceDistance(curves, s_half, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves))])


    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    JF = Jf \
        + J_LENGTH_PENALTY \
        + CC_WEIGHT * Jccdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
        + LENGTH_WEIGHT * sum(Jls) \
        # + CS_WEIGHT * Jcsdist

    # We don't have a general interface in SIMSOPT for optimisation problems that
    # are not in least-squares form, so we write a little wrapper function that we
    # pass directly to scipy.optimize.minimize


    def fun(dofs, info={'Nfeval':0}):
        JF.x = dofs
        info['Nfeval'] += 1
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.abs(np.sum(bs.B().reshape((nphi_half, ntheta_half, 3)) * s_half.unitnormal(), axis=2))
        BdotN_mean = np.mean(BdotN)
        BdotN_min = np.min(BdotN)
        BdotN_max = np.max(BdotN)
        outstr = f"#{info['Nfeval']}, J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN_mean:.1e}, B·n max={BdotN_max:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"#, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad


    # print("""
    # ################################################################################
    # ### Perform a Taylor test ######################################################
    # ################################################################################
    # """)
    f = fun
    dofs = JF.x
    # np.random.seed(1)
    # h = np.random.uniform(size=dofs.shape)
    # J0, dJ0 = f(dofs)
    # dJh = sum(dJ0 * h)
    # for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #     J1, _ = f(dofs + eps*h)
    #     J2, _ = f(dofs - eps*h)
    #     print("err", (J1-J2)/(2*eps) - dJh)

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_{LENGTH_THRESHOLD}")

    bs.set_points(s_full.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi_full, ntheta_full, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
    s_full.to_vtk(OUT_DIR + f"surf_opt_{LENGTH_THRESHOLD}", extra_data=pointData)


    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    bs.save(OUT_DIR + f"biot_savart_opt_{LENGTH_THRESHOLD}.json")
# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + f"biot_savart_opt.json")
