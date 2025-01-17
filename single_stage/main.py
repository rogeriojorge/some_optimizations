#!/usr/bin/env python

"""
Optimize a VMEC equilibrium for quasisymmetry and coils in
a single stage optimization problem
Rogerio Jorge, Lisbon, July 14 2022
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
import booz_xform as bx
from pathlib import Path
import matplotlib.pyplot as plt
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
import inputs
from opt_funcs import pprint, create_results_folders, create_initial_coils
from stage_1 import form_stage_1_objective_function
from stage_2 import form_stage_2_objective_function, inner_coil_loop
from simsopt import load
from simsopt.mhd import Vmec, Boozer
from simsopt.util.mpi import log
from simsopt.field import Current
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk
from simsopt.util import MpiPartition
from simsopt._core.derivative import Derivative
from simsopt.solve import least_squares_mpi_solve
from simsopt.geo import create_equally_spaced_curves
from simsopt._core.finite_difference import MPIFiniteDifference, finite_difference_steps
import logging
logger = logging.getLogger(__name__)
import argparse

# Start the timer
start = time.time()

QAQHselected=False
if len(sys.argv) > 1:
    if sys.argv[1]=='QA' or sys.argv[1]=='QH':
        QAorQH = sys.argv[1]
        QAQHselected=True
if not QAQHselected:
    pprint('First line argument (QA or QH) not selected. Defaulting to QA.')
    QAorQH = 'QA'

# Parse the command line arguments and overwrite inputs.py if needed
parser = argparse.ArgumentParser()
parser.add_argument("--vmec_input_start", default=inputs.vmec_input_start_QA if QAorQH=='QA' else inputs.vmec_input_start_QH)
parser.add_argument("--lengthbound", type=float, default=inputs.LENGTHBOUND_QA if QAorQH=='QA' else inputs.LENGTHBOUND_QH)
parser.add_argument("--cc_threshold", type=float, default=inputs.CC_THRESHOLD_QA if QAorQH=='QA' else inputs.CC_THRESHOLD_QH)
parser.add_argument("--msc_threshold", type=float, default=inputs.MSC_THRESHOLD_QA if QAorQH=='QA' else inputs.MSC_THRESHOLD_QH)
parser.add_argument("--curvature_threshold", type=float, default=inputs.CURVATURE_THRESHOLD_QA if QAorQH=='QA' else inputs.CURVATURE_THRESHOLD_QH)
parser.add_argument("--ncoils", type=float, default=inputs.ncoils_QA if QAorQH=='QA' else inputs.ncoils_QH)
parser.add_argument("--order", type=float, default=inputs.order)
parser.add_argument("--quasisymmetry_helicity_n", type=float, default=inputs.quasisymmetry_helicity_n_QA if QAorQH=='QA' else inputs.quasisymmetry_helicity_n_QH)
parser.add_argument("--aspect_ratio_target", type=float, default=inputs.aspect_ratio_target_QA if QAorQH=='QA' else inputs.aspect_ratio_target_QH)
parser.add_argument("--stage1", dest="stage1", default=inputs.stage_1, action="store_true")
parser.add_argument("--stage2", dest="stage2", default=inputs.stage_2, action="store_true")
parser.add_argument("--single_stage", dest="single_stage", default=inputs.single_stage, action="store_true")
parser.add_argument("--include_iota_target", dest="include_iota_target", default=inputs.include_iota_target_QA if QAorQH=='QA' else inputs.include_iota_target_QH, action="store_true")
if QAQHselected:
    args = parser.parse_args(sys.argv[2:])
else:
    args = parser.parse_args()
inputs.order = args.order
inputs.ncoils = args.ncoils
inputs.LENGTHBOUND = args.lengthbound
inputs.CC_THRESHOLD = args.cc_threshold
inputs.MSC_THRESHOLD = args.msc_threshold
inputs.vmec_input_start = args.vmec_input_start
inputs.CURVATURE_THRESHOLD = args.curvature_threshold
inputs.stage_1 = args.stage1
inputs.stage_2 = args.stage2
inputs.single_stage = args.single_stage
inputs.quasisymmetry_helicity_n = args.quasisymmetry_helicity_n
inputs.include_iota_target = args.include_iota_target
inputs.aspect_ratio_target = args.aspect_ratio_target
stage_string = ''
if args.stage1: stage_string+='1'
if args.stage2: stage_string+='2'
if args.single_stage: stage_string+='3'
if stage_string == '': stage_string = '123'
inputs.name = f'{QAorQH}_Stage{stage_string}_Lengthbound{args.lengthbound:.1f}_ncoils{args.ncoils}'

pprint("============================================")
pprint("Starting single stage optimization")
pprint("============================================")

mpi = MpiPartition()
## CREATE A NEW MpiPartition(len(prob.x)+1) object everytime you change the number of degrees of freedom
# log(level=logging.DEBUG)
# log(level=logging.INFO)

# Create directories where the results will be saved
# and backup directory if already exists
current_path = os.path.join(parent_path, f'{inputs.name}')
if mpi.proc0_world:
    if inputs.remove_previous_results and os.path.isdir(current_path):
        shutil.copytree(current_path, current_path + '_backup', dirs_exist_ok=True)
        shutil.rmtree(current_path)
Path(current_path).mkdir(parents=True, exist_ok=True)
current_path = str(Path(current_path).resolve())
if mpi.proc0_world:
    shutil.copy(os.path.join(parent_path,'inputs.py'), os.path.join(current_path,f'{inputs.name}.py'))
os.chdir(current_path)
coils_results_path, vmec_results_path, figures_results_path = create_results_folders(inputs)
inputs_dict = dict([(att, getattr(inputs,att)) for att in dir(inputs) if '__' not in att])
with open(os.path.join(current_path, f'inputs_{inputs.name}.json'), 'w', encoding='utf-8') as f:
    json.dump(inputs_dict, f, ensure_ascii=False, indent=4)

# Check if a previous optimization has already taken place and load it if exists
vmec_files_list = os.listdir(vmec_results_path)
if len(vmec_files_list)==0:
    vmec_input_filename = os.path.join(parent_path, 'vmec_inputs', inputs.vmec_input_start)
else:
    vmec_input_files = [file for file in vmec_files_list if 'input.' in file]
    vmec_input_files.sort(key=lambda item: (len(item), item), reverse=False)
    vmec_input_filename = vmec_input_files[-1]

pprint(f' Using vmec input file {os.path.join(vmec_results_path,vmec_input_filename)}')
vmec = Vmec(os.path.join(vmec_results_path,vmec_input_filename), mpi=mpi, verbose=inputs.vmec_verbose, nphi=inputs.nphi, ntheta=inputs.ntheta)
surf = vmec.boundary

# Check if there are already optimized coils we can use
bs_json_files = [file for file in os.listdir(coils_results_path) if '.json' in file]
# if Path(os.path.join(coils_results_path,"biot_savart_opt.json")).is_file():
if len(bs_json_files)==0:
    base_curves = create_equally_spaced_curves(inputs.ncoils, vmec.indata.nfp, stellsym=True, R0=inputs.R0, R1=inputs.R1, order=inputs.order)
    base_currents = [Current(inputs.initial_current*1e-5)*1e5 for i in range(inputs.ncoils)]
else:
    bs_temporary = load(os.path.join(coils_results_path, bs_json_files[-1]))
    base_curves = [bs_temporary.coils[i]._curve for i in range(inputs.ncoils)]
    base_currents = [bs_temporary.coils[i]._current for i in range(inputs.ncoils)]

# Create the initial coils
base_currents[0].fix_all()
bs, coils, curves = create_initial_coils(base_curves, base_currents, vmec.indata.nfp, surf, coils_results_path, inputs)

# Define objective function and its derivatives
class single_stage_obj_and_der():
    def __init__(self) -> None:
        pass

    def fun(self, dofs, prob_jacobian=None, info={'Nfeval':0}, max_mode=1, oustr_dict=[]):
        logger.info('Entering fun')
        info['Nfeval'] += 1
        JF.x = dofs[:-number_vmec_dofs]
        prob.x = dofs[-number_vmec_dofs:]
        bs.set_points(surf.gamma().reshape((-1, 3)))
        os.chdir(vmec_results_path)

        J_stage_1 = prob.objective()
        J_stage_2 = inputs.coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        if J > inputs.JACOBIAN_THRESHOLD:
            logger.info(f"Exception caught during function evaluation with J={J}. Returning J={inputs.JACOBIAN_THRESHOLD}")
            J = inputs.JACOBIAN_THRESHOLD
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)))
        outstr = f"\n\nfun#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        dict1 = {}
        dict1.update({
            'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf),'J_length':float(J_LENGTH.J()),
            'J_CC':float(J_CC.J()),'J_CS':float(J_CS.J()),'J_CURVATURE':float(J_CURVATURE.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),
            'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()), 'Lengths':float(sum(j.J() for j in Jls)),
            'curvatures':float(np.sum([np.max(c.kappa()) for c in base_curves])),'msc':float(np.sum([j.J() for j in Jmscs])),
            'B.n':float(BdotN),
            # 'gradJcoils':float(np.linalg.norm(JF.dJ())),
            'C-C-Sep':float(Jccdist.shortest_distance()), 'C-S-Sep':float(Jcsdist.shortest_distance())
        })

        if inputs.debug_coils_outputtxt:
            # outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}"
            outstr += f", , C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
            outstr += f"\nJf={jf:.1e}, J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CS={J_CS.J():.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}"
            cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
            kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
            msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
            outstr += f"\n Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
        try:
            outstr += f"\n surface dofs="+", ".join([f"{pr}" for pr in dofs[-number_vmec_dofs:]])
            if J<inputs.JACOBIAN_THRESHOLD:
                dict1.update({'Jquasisymmetry':float(qs.total()), 'Jiota':float((vmec.mean_iota()-inputs.iota_target)**2), 'Jaspect':float((vmec.aspect()-inputs.aspect_ratio_target)**2)})
                outstr += f"\n Quasisymmetry objective={qs.total()}"
                outstr += f"\n aspect={vmec.aspect()}"
                outstr += f"\n mean iota={vmec.mean_iota()}"
            else:
                dict1.update({'Jquasisymmetry':0, 'Jiota':0,'Jaspect':0})
        except Exception as e:
            pprint(e)

        if J<inputs.JACOBIAN_THRESHOLD:
            logger.info(f'Objective function {J} is smaller than the threshold {inputs.JACOBIAN_THRESHOLD}')
            ## Finite differences for the first-stage objective function
            prob_dJ = prob_jacobian.jac(prob.x)

            if inputs.coil_gradients_analytical:
                ## Finite differences for the second-stage objective function
                coils_dJ = JF.dJ()
                ## Mixed term - derivative of squared flux with respect to the surface shape
                n = surf.normal()
                absn = np.linalg.norm(n, axis=2)
                B = bs.B().reshape((inputs.nphi, inputs.ntheta, 3))
                dB_by_dX = bs.dB_by_dX().reshape((inputs.nphi, inputs.ntheta, 3, 3))
                Bcoil = bs.B().reshape(n.shape)
                B_N = np.sum(Bcoil * n, axis=2)
                dJdx = (B_N/absn)[:, :, None] * (np.sum(dB_by_dX*n[:, :, None, :], axis=3))
                dJdN = (B_N/absn)[:, :, None] * B - 0.5 * (B_N**2/absn**3)[:, :, None] * n
                deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(inputs.nphi*inputs.ntheta)) + surf.dgamma_by_dcoeff_vjp(dJdx/(inputs.nphi*inputs.ntheta))
                mixed_dJ = Derivative({surf: deriv})(surf)
                ## Put both gradients together
                grad_with_respect_to_coils = inputs.coils_objective_weight * coils_dJ
                grad_with_respect_to_surface = np.ravel(prob_dJ) + inputs.coils_objective_weight * mixed_dJ
            else:
                # Finite difference for the coil gradients
                grad_coils = np.zeros(len(dofs),)
                steps = finite_difference_steps(dofs, abs_step=inputs.finite_difference_abs_step, rel_step=inputs.finite_difference_rel_step)
                f0 = inputs.coils_objective_weight * JF.J()
                for j in range(len(dofs)):
                    x = np.copy(dofs)
                    x[j] = dofs[j] + steps[j]
                    JF.x = x[:-number_vmec_dofs]
                    if np.sum(prob.x-x[-number_vmec_dofs:])!=0:
                        prob.x = x[-number_vmec_dofs:]
                    bs.set_points(surf.gamma().reshape((-1, 3)))
                    fplus = inputs.coils_objective_weight * JF.J()
                    grad_coils[j] = (fplus - f0) / steps[j]
                grad_with_respect_to_coils = grad_coils[:-number_vmec_dofs]
                grad_with_respect_to_surface = np.ravel(prob_dJ) + grad_coils[-number_vmec_dofs:]

            grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
        else:
            logger.info(f'Objective function {J} is greater than the threshold {inputs.JACOBIAN_THRESHOLD}')
            grad = [0] * len(dofs)

        os.chdir(current_path)
        with open(inputs.debug_output_file, "a") as myfile:
            myfile.write(outstr)
            # if J<inputs.JACOBIAN_THRESHOLD:
            #     myfile.write(f"\n prob_dJ="+", ".join([f"{p}" for p in np.ravel(prob_dJ)])+"\n coils_dJ[3:10]="+", ".join([f"{p}" for p in coils_dJ[3:10]])+"\n mixed_dJ="+", ".join([f"{p}" for p in mixed_dJ]))
        oustr_dict.append(dict1)
        if np.mod(info['Nfeval'],5)==0:
            pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
            surf.to_vtk(os.path.join(coils_results_path,f"surf_intermediate_max_mode_{max_mode}_{info['Nfeval']}"), extra_data=pointData)
            curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_intermediate_max_mode_{max_mode}_{info['Nfeval']}"))

        return J, grad

# Loop over the number of predefined maximum poloidal/toroidal modes
if inputs.stage_1 or inputs.stage_2 or inputs.single_stage:
    oustr_dict_outer=[]
    previous_max_mode=0
    for max_mode in inputs.max_modes:
        if max_mode != previous_max_mode: oustr_dict_inner=[]
        pprint(f' Starting optimization with max_mode={max_mode}')
        pprint(f'  Forming stage 1 objective function')
        surf, vmec, qs, number_vmec_dofs, prob = form_stage_1_objective_function(vmec, surf, max_mode, inputs)
        pprint(f'  Forming stage 2 objective function')
        JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist, Jf, \
            J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY = form_stage_2_objective_function(surf, bs, base_curves, curves, inputs)

        # Stage 1 Optimization
        if inputs.stage_1:
            os.chdir(vmec_results_path)
            pprint(f'  Performing Stage 1 optimization with {inputs.MAXITER_stage_1} iterations')
            least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=inputs.MAXITER_stage_1)
            os.chdir(current_path)
            if mpi.proc0_world:
                with open(inputs.debug_output_file, "a") as myfile:
                    try:
                        myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                        myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                        myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                        myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                    except Exception as e:
                        myfile.write(e)

        # Stage 2 Optimization
        if (inputs.stage_2 and inputs.stage_1) or previous_max_mode==0:
            pprint(f'  Performing Stage 2 optimization with {inputs.MAXITER_stage_2+inputs.MAXITER_stage_2_simple} iterations')
            surf = vmec.boundary
            bs.set_points(surf.gamma().reshape((-1, 3)))
            if mpi.proc0_world:
                dofs, bs, JF = inner_coil_loop(mpi, JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist, Jf, J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY, vmec, curves, base_curves, surf, coils_results_path, number_vmec_dofs, bs, max_mode, inputs, figures_results_path)
                with open(inputs.debug_output_file, "a") as myfile:
                    try:
                        myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                        myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                        myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                        myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                    except Exception as e:
                        myfile.write(e)

        # Single stage Optimization
        if inputs.single_stage:
            pprint(f'  Performing Single Stage optimization with {inputs.MAXITER_single_stage} iterations')
            x0 = np.copy(np.concatenate((JF.x, vmec.x)))
            dofs = np.concatenate((JF.x, vmec.x))
            obj_and_der = single_stage_obj_and_der()
            with MPIFiniteDifference(prob.objective, mpi, rel_step=inputs.finite_difference_rel_step, abs_step=inputs.finite_difference_abs_step, diff_method="forward") as prob_jacobian:
                if mpi.proc0_world:
                    res = minimize(obj_and_der.fun, dofs, args=(prob_jacobian,{'Nfeval':0},max_mode,oustr_dict_inner), jac=True, method='BFGS', options={'maxiter': inputs.MAXITER_single_stage}, tol=1e-15)
                    with open(inputs.debug_output_file, "a") as myfile:
                        try:
                            myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                            myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                            myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                            myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                        except Exception as e:
                            myfile.write(e)

        # Stage 2 Optimization after single_stage
        if (inputs.stage_2 and inputs.single_stage) or (inputs.stage_2 and not inputs.single_stage and not inputs.stage_1):
            pprint(f'  Performing Stage 2 optimization with {inputs.MAXITER_stage_2+inputs.MAXITER_stage_2_simple} iterations')
            surf = vmec.boundary
            bs.set_points(surf.gamma().reshape((-1, 3)))
            if mpi.proc0_world:
                dofs, bs, JF = inner_coil_loop(mpi, JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist, Jf, J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY, vmec, curves, base_curves, surf, coils_results_path, number_vmec_dofs, bs, max_mode, inputs, figures_results_path)
                with open(inputs.debug_output_file, "a") as myfile:
                    try:
                        myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                        myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                        myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                        myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                    except Exception as e:
                        myfile.write(e)

        if mpi.proc0_world:
            pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
            surf.to_vtk(os.path.join(coils_results_path,inputs.resulting_surface+'max_mode_'+str(max_mode)), extra_data=pointData)
            curves_to_vtk(curves, os.path.join(coils_results_path,inputs.resulting_coils+'max_mode_'+str(max_mode)))
            bs.save(os.path.join(coils_results_path,inputs.resulting_field_json+'max_mode_'+str(max_mode)+'.json'))
            vmec.write_input(os.path.join(vmec_results_path, f'input.{inputs.name}_maxmode{max_mode}'))

        os.chdir(vmec_results_path)
        try:
            pprint(f"Aspect ratio at max_mode {max_mode}: {vmec.aspect()}")
            pprint(f"Mean iota at {max_mode}: {vmec.mean_iota()}")
            pprint(f"Quasisymmetry objective at max_mode {max_mode}: {qs.total()}")
            pprint(f"Squared flux at max_mode {max_mode}: {Jf.J()}")
        except Exception as e:
            pprint(e)
        os.chdir(current_path)
        if inputs.single_stage and mpi.proc0_world:
            try:
                df = pd.DataFrame(oustr_dict_inner)
                df.to_csv(os.path.join(current_path, f'output_max_mode_{max_mode}.csv'), index_label='index')
                ax=df.plot(
                    kind='line',
                    logy=True,
                    y=['J','Jf','J_length','J_CC','J_CURVATURE','J_MSC','J_ALS','J_LENGTH_PENALTY','Jquasisymmetry','Jiota','Jaspect'],#,'C-C-Sep','C-S-Sep'],
                    linewidth=0.8)
                ax.set_ylim(bottom=1e-9, top=None)
                ax.set_xlabel('Number of function evaluations')
                ax.set_ylabel('Objective function')
                plt.legend(loc=3, prop={'size': 6})
                plt.tight_layout()
                plt.savefig(os.path.join(figures_results_path, f'optimization_stage3_max_mode_{max_mode}.pdf'), bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            except Exception as e:
                pprint(e)
        previous_max_mode = max_mode
        oustr_dict_outer.append(oustr_dict_inner)

    if mpi.proc0_world:
        pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path,inputs.resulting_surface), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,inputs.resulting_coils))
        bs.save(os.path.join(coils_results_path,inputs.resulting_field_json))
        vmec.write_input(os.path.join(current_path, f'input.{inputs.name}_final'))
        if inputs.single_stage:
            try:
                df = pd.DataFrame(oustr_dict_outer)
                df.to_csv(os.path.join(current_path, f'output_{inputs.name}_final.csv'), index_label='index')
                ax=df.plot(kind='line',
                    logy=True,
                    y=['J','Jf','J_length','J_CC','J_CURVATURE','J_MSC','J_ALS','J_LENGTH_PENALTY','Jquasisymmetry', 'Jiota','Jaspect'],#,'C-C-Sep','C-S-Sep'],
                    linewidth=0.8)
                ax.set_ylim(bottom=1e-9, top=None)
                plt.legend(loc=3, prop={'size': 6})
                ax.set_xlabel('Number of function evaluations')
                ax.set_ylabel('Objective function')
                plt.tight_layout()
                plt.savefig(os.path.join(figures_results_path, f'optimization_stage3_final.pdf'), bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            except Exception as e:
                pprint(e)
    os.chdir(vmec_results_path)
    try:
        pprint(f"Aspect ratio after optimization: {vmec.aspect()}")
        pprint(f"Mean iota after optimization: {vmec.mean_iota()}")
        pprint(f"Quasisymmetry objective after optimization: {qs.total()}")
        pprint(f"Squared flux after optimization: {Jf.J()}")
    except Exception as e:
        pprint(e)

os.chdir(current_path)
if inputs.create_wout_final:
    try:
        vmec_final = Vmec(os.path.join(current_path, f'input.{inputs.name}_final'))
        vmec_final.indata.ns_array[:3]    = [  16,    51,    101]
        vmec_final.indata.niter_array[:3] = [ 2000,  3000,  8000]
        vmec_final.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
        vmec_final.run()
        if mpi.proc0_world:
            shutil.move(os.path.join(current_path, f"wout_{inputs.name}_final_000_000000.nc"), os.path.join(current_path, f"wout_{inputs.name}_final.nc"))
            os.remove(os.path.join(current_path, f'input.{inputs.name}_final_000_000000'))
    except Exception as e:
        pprint('Exception when creating final vmec file:')
        pprint(e)

## Create results figures
if os.path.isfile(os.path.join(current_path, f"wout_{inputs.name}_final.nc")):
    pprint('Found final vmec file')
    sys.path.insert(1, os.path.join(parent_path, inputs.plotting_folder))
    if mpi.proc0_world:
        if inputs.vmec_plot_result:
            pprint("Plot VMEC result")
            import vmecPlot2
            vmecPlot2.main(file=os.path.join(current_path, f"wout_{inputs.name}_final.nc"), name=inputs.name, figures_folder=figures_results_path, coils_curves=curves)

    if inputs.booz_xform_plot_result:
        pprint('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec_final, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,inputs.boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        if mpi.proc0_world:
            b1.bx.write_boozmn(os.path.join(vmec_results_path,"boozmn_"+inputs.name+".nc"))
            pprint("Plot BOOZ_XFORM")
            fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_1_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=int(inputs.boozxform_nsurfaces/2), fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_2_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=inputs.boozxform_nsurfaces-1, fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_3_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            if inputs.name[0:2] == 'QH':
                helical_detail = True
            else:
                helical_detail = False
            fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_symplot_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
            plt.savefig(os.path.join(figures_results_path, "Boozxform_modeplot_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()

    if inputs.find_QFM_surface:
        pprint("Obtain QFM surface")
        import field_from_coils
        R_qfm, Z_qfm, Raxis_qfm, Zaxis_qfm, vmec_qfm = field_from_coils.main(folder=current_path, OUT_DIR=figures_results_path, coils_folder=inputs.coils_folder,
                            vmec_folder=vmec_results_path, mpi=mpi, nzeta=inputs.nzeta_Poincare, nradius=inputs.nradius_Poincare, tol_qfm = inputs.tol_qfm,
                            maxiter_qfm = inputs.maxiter_qfm, constraint_weight = inputs.constraint_weight_qfm, ntheta=inputs.ntheta_vmec_qfm, name_manual=inputs.name,
                            mpol_qfm = inputs.mpol_qfm, ntor_qfm = inputs.ntor_qfm, nphi_qfm = inputs.nphi_qfm, ntheta_qfm = inputs.ntheta_qfm, diagnose_QFM=False,
                            # qfm_poincare_plot=False)
                            qfm_poincare_plot=True, tmax_fl = inputs.tmax_fl_Poincare, degree = inputs.degree_Interpolated_field, tol_tracing = inputs.tol_tracing_Poincare)
        if np.sum(R_qfm) != 0:
            if inputs.vmec_plot_QFM_result:
                import vmecPlot2
                vmecPlot2.main(file=os.path.join(current_path, f"wout_{inputs.name}_qfm.nc"), name=inputs.name+'_qfm', figures_folder=figures_results_path, coils_curves=curves)

            if inputs.booz_xform_plot_QFM_result:
                pprint('Creating Boozer class for vmec_qfm')
                b1_qfm = Boozer(vmec_qfm, mpol=64, ntor=64)
                pprint('Defining surfaces where to compute Boozer coordinates')
                booz_surfaces_qfm = np.linspace(0,1,inputs.boozxform_nsurfaces,endpoint=False)
                pprint(f' booz_surfaces={booz_surfaces_qfm}')
                b1_qfm.register(booz_surfaces_qfm)
                pprint('Running BOOZ_XFORM')
                b1_qfm.run()
                if mpi.proc0_world:
                    b1_qfm.bx.write_boozmn(os.path.join(vmec_results_path,"boozmn_"+inputs.name+"_qfm.nc"))
                    pprint("Plot BOOZ_XFORM")
                    fig = plt.figure(); bx.surfplot(b1_qfm.bx, js=1,  fill=False, ncontours=35)
                    plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_1_"+inputs.name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
                    fig = plt.figure(); bx.surfplot(b1_qfm.bx, js=int(inputs.boozxform_nsurfaces/2), fill=False, ncontours=35)
                    plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_2_"+inputs.name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
                    fig = plt.figure(); bx.surfplot(b1_qfm.bx, js=inputs.boozxform_nsurfaces-1, fill=False, ncontours=35)
                    plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_3_"+inputs.name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
                    if inputs.name[0:2] == 'QH':
                        helical_detail = True
                    else:
                        helical_detail = False
                    fig = plt.figure(); bx.symplot(b1_qfm.bx, helical_detail = helical_detail, sqrts=True)
                    plt.savefig(os.path.join(figures_results_path, "Boozxform_symplot_"+inputs.name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
                    fig = plt.figure(); bx.modeplot(b1_qfm.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
                    plt.savefig(os.path.join(figures_results_path, "Boozxform_modeplot_"+inputs.name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()

os.chdir(parent_path)

# Stop the timer
stop = time.time()

pprint("============================================")
pprint("End of single stage optimization")
pprint(f"Took {stop-start} seconds")
pprint("============================================")
