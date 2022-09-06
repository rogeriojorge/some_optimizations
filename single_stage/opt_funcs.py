import os
import numpy as np
from pathlib import Path
from math import ceil, sqrt
import matplotlib.pyplot as plt
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart
from simsopt.field import coils_via_symmetries

# Define a print function that only prints on one processor
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    def pprint(*args, **kwargs):
        if comm.rank == 0:
            print(*args, **kwargs)
except ImportError:
    comm = None
    pprint = print

def create_results_folders(inputs):
    Path(inputs.coils_folder).mkdir(parents=True, exist_ok=True)
    coils_results_path = str(Path(inputs.coils_folder).resolve())
    Path(inputs.vmec_folder).mkdir(parents=True, exist_ok=True)
    vmec_results_path = str(Path(inputs.vmec_folder).resolve())
    Path(inputs.figures_folder).mkdir(parents=True, exist_ok=True)
    figures_results_path = str(Path(inputs.figures_folder).resolve())
    if inputs.remove_previous_debug_output and inputs.single_stage:
        try:
            os.remove(inputs.debug_output_file)
        except OSError:
            pass
    return coils_results_path, vmec_results_path, figures_results_path

def create_initial_coils(base_curves, base_currents, nfp, surf, coils_results_path, inputs):
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, os.path.join(coils_results_path, inputs.initial_coils))
    pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, inputs.initial_surface), extra_data=pointData)
    return bs, coils, curves

def plot_qfm_poincare(phis, fieldlines_phi_hits, R, Z, OUT_DIR, name):
    if comm is None or comm.rank == 0:
        nradius = len(fieldlines_phi_hits)
        r = []
        z = []
        # Obtain Poincare plot
        for izeta in range(len(phis)):
            r_2D = []
            z_2D = []
            for iradius in range(len(fieldlines_phi_hits)):
                lost = fieldlines_phi_hits[iradius][-1, 1] < 0
                data_this_phi = fieldlines_phi_hits[iradius][np.where(fieldlines_phi_hits[iradius][:, 1] == izeta)[0], :]
                if data_this_phi.size == 0:
                    pprint(f'No Poincare data for iradius={iradius} and izeta={izeta}')
                    continue
                r_2D.append(np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2))
                z_2D.append(data_this_phi[:, 4])
            r.append(r_2D)
            z.append(z_2D)
        r = np.array(r, dtype=object)
        z = np.array(z, dtype=object)

        # Plot figure
        nrowcol = ceil(sqrt(len(phis)))
        fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(12, 8))
        for i in range(len(phis)):
            row = i//nrowcol
            col = i % nrowcol
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
            axs[row, col].set_xlabel("$R$")
            axs[row, col].set_ylabel("$Z$")
            axs[row, col].set_aspect('equal')
            axs[row, col].tick_params(direction="in")
            for j in range(nradius):
                if j== 0 and i == 0:
                    legend1 = 'Poincare plot'
                    legend2 = 'VMEC QFM'
                else:
                    legend1 = legend2 = '_nolegend_'
                try: axs[row, col].scatter(r[i][j], z[i][j], marker='o', s=0.7, linewidths=0, c='b', label = legend1)
                except Exception as e: pprint(e, i, j)
                axs[row, col].scatter(R[i,j], Z[i,j], marker='o', s=0.7, linewidths=0, c='r', label = legend2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_poincare_VMEC_fieldline_all.pdf'), bbox_inches = 'tight', pad_inches = 0)

        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        axs.set_title(f"$\\phi = {phis[0]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
        axs.set_xlabel("$Z$")
        axs.set_ylabel("$R$")
        axs.set_aspect('equal')
        axs.tick_params(direction="in")
        for j in range(nradius):
            if j== 0:
                legend1 = 'Poincare plot'
                legend2 = 'VMEC QFM'
            else:
                legend1 = legend2 = '_nolegend_'
            try: axs.scatter(r[0][j], z[0][j], marker='o', s=1.5, linewidths=0.5, c='b', label = legend1)
            except Exception as e: pprint(e, 0, j)
            axs.scatter(R[0,j], Z[0,j], marker='o', s=1.5, linewidths=0.5, c='r', label = legend2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_poincare_VMEC_fieldline_0.pdf'), bbox_inches = 'tight', pad_inches = 0)