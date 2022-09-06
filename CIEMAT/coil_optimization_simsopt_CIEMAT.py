#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.geo import RotatedCurve, curves_to_vtk, create_equally_spaced_curves
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil, coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, CurveSurfaceDistance, \
    MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation
from simsopt.geo.curveperturbed import GaussianSampler, CurvePerturbed, PerturbationSample
from simsopt.objectives import QuadraticPenalty, MPIObjective
from simsopt.mhd import Vmec
from simsopt.mhd import VirtualCasing
from randomgen import PCG64
from glob import glob

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    def pprint(*args, **kwargs):
        if comm.rank == 0:  # only print on rank 0
            print(*args, **kwargs)
except ImportError:
    comm = None
    pprint = print

MAXITER = 600
ncoils = 6
R0 = 2
R1 = 0.75
order = 12
LENGTH_WEIGHT = 1e-3
CC_THRESHOLD = 0.11
CC_WEIGHT = 20e+1
CS_THRESHOLD = 0.21
CS_WEIGHT = 1e+1
CURVATURE_THRESHOLD = 5
CURVATURE_WEIGHT = 5e-5
MSC_THRESHOLD = 14
MSC_WEIGHT = 1e-5
ARCLENGTH_WEIGHT = 2e-3
SIGMA = 1e-3
L = 0.75
N_SAMPLES = 8
N_OOS = 128
vc_src_nphi = 80

# File for the desired boundary magnetic surface:
filename = os.path.join(os.path.dirname(__file__), 'input.LNF4p1714_2537b1.5.vmec')
vmec_file = os.path.join(os.path.dirname(__file__), 'wout_LNF4p1714_2537b1.5.vmec.nc')

# Directory for output
OUT_DIR = "output"
if comm.rank == 0:
    filelist = glob(f"{OUT_DIR}/*")
    for f in filelist:
        os.remove(f)
    os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface; errors break symmetries, so consider the full torus
nphi = 64
ntheta = 64
# s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
s = SurfaceRZFourier.from_wout(vmec_file, range="full torus", nphi=nphi, ntheta=ntheta)

# Once the virtual casing calculation has been run once, the results
# can be used for many coil optimizations. Therefore here we check to
# see if the virtual casing output file alreadys exists. If so, load
# the results, otherwise run the virtual casing calculation and save
# the results.
head, tail = os.path.split(vmec_file)
vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
pprint('virtual casing data file:', vc_filename)
if os.path.isfile(vc_filename):
    pprint('Loading saved virtual casing result')
    vc = VirtualCasing.load(vc_filename)
else:
    # Virtual casing must not have been run yet.
    pprint('Running the virtual casing calculation')
    vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
total_current = Vmec(vmec_file).external_current() / (2 * s.nfp)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
# Since we know the total sum of currents, we only optimize for ncoils-1
# currents, and then pick the last one so that they all add up to the correct
# value.
base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
# Above, the factors of 1e-5 and 1e5 are included so the current
# degrees of freedom are O(1) rather than ~ MA.  The optimization
# algorithm may not perform well if the dofs are scaled badly.
total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "/curves_init")
# pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
pointData = {"B_N": BdotN[:, :, None]}
s.to_vtk(OUT_DIR + "/surf_init", extra_data=pointData)

Jf = SquaredFlux(s, bs, target=vc.B_external_normal)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]

seed = 0
rg = np.random.Generator(PCG64(seed, inc=0))

sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=1)
Jfs = []
curves_pert = []
for i in range(N_SAMPLES):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    Jfs.append(SquaredFlux(s, bs_pert))
Jmpi = MPIObjective(Jfs, comm, needs_splitting=True)

for i in range(len(curves_pert)):
    curves_to_vtk(curves_pert[i], OUT_DIR + f"/curves_init_{i}")

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CS = CS_WEIGHT * Jcsdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)

JF = \
    Jf \
    + J_LENGTH \
    # + J_ALS \
    # + J_MSC \
    # + J_CURVATURE \
    # + J_CC \
    # + J_CS \
    # + Jmpi \

def fun(dofs, info={'Nfeval':0}):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    # jf = Jmpi.J()
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    BdotN_mean = np.mean(BdotN)
    BdotN_max = np.max(BdotN)
    outstr = f"\nJ={J:.1e}, Jf={jf:.1e}, ⟨|B·n|⟩={BdotN_mean:.1e}, max(|B·n|)={BdotN_max:.1e}, ║∇J coils║={np.linalg.norm(grad):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f"\n Jf={jf:.1e}, J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CS={J_CS.J():.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}"
    cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
    outstr += f"\n Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
    pprint(outstr, flush=True)
    info['Nfeval']+=1
    pprint(f"Iteration {info['Nfeval']} of {MAXITER+3}")
    return J, grad

# pprint("""
# ################################################################################
# ### Perform a Taylor test ######################################################
# ################################################################################
# """)
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
# for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
#     J1, _ = f(dofs + eps*h)
#     J2, _ = f(dofs - eps*h)
#     pprint("err", (J1-J2)/(2*eps) - dJh)

pprint("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
from scipy.optimize import minimize
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 400}, tol=1e-15)
alen_string = ", ".join([f"{np.max(c.incremental_arclength())/np.min(c.incremental_arclength())-1:.2e}" for c in base_curves])
pprint(f"Final arclength variation max(|ℓ|)/min(|ℓ|) - 1=[{alen_string}]")

pprint("""
################################################################################
### Evaluate the obtained coils ################################################
################################################################################
""")
Jf.x = res.x
curves_to_vtk(curves, OUT_DIR + "/curves_opt")
for i in range(len(curves_pert)):
    curves_to_vtk(curves_pert[i], OUT_DIR + f"/curves_opt_{i}")
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
pointData = {"B_N": BdotN[:, :, None]}
bs.save('biot_savart_opt.json')
s.to_vtk(OUT_DIR + "/surf_opt", extra_data=pointData)
pprint(f"Mean Flux Objective across perturbed coils: {Jmpi.J():.3e}")
pprint(f"Flux Objective for exact coils coils      : {Jf.J():.3e}")

J = JF.J()
grad = JF.dJ()
jf = Jmpi.J()
# BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
BdotN_mean = np.mean(BdotN)
BdotN_max = np.max(BdotN)
outstr = f"\nJ={J:.1e}, Jf={jf:.1e}, ⟨|B·n|⟩={BdotN_mean:.1e}, max(|B·n|)={BdotN_max:.1e}, ║∇J coils║={np.linalg.norm(grad):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
outstr += f"\n Jf={jf:.1e}, J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CS={J_CS.J():.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}"
cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
outstr += f"\n Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
pprint(outstr)

# # now draw some fresh samples to evaluate the out-of-sample error
# rg = np.random.Generator(PCG64(seed+1, inc=0))
# val = 0
# for i in range(N_OOS):
#     # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
#     base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
#     coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
#     # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
#     coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
#     curves_pert.append([c.curve for c in coils_pert])
#     bs_pert = BiotSavart(coils_pert)
#     val += SquaredFlux(s, bs_pert).J()

# val *= 1./N_OOS
# pprint(f"Out-of-sample flux value                  : {val:.3e}")
