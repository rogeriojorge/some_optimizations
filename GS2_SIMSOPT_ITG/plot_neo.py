#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import warnings
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)

this_path = Path(__file__).parent.resolve()

out_dirs = ([['output_MAXITER350_least_squares_nfp2_QA_QA',
              'output_MAXITER350_least_squares_nfp2_QA_QA_onlyQS'],
             ['output_MAXITER350_least_squares_nfp4_QH_QH',
              'output_MAXITER350_least_squares_nfp4_QH_QH_onlyQS']])
w7x_neo = '/Users/rogeriojorge/local/vmec_equilibria/W7-X/Standard_fixed_boundary/neo_out.W7-X_standard_configuration'
hsx_neo = '/Users/rogeriojorge/local/vmec_equilibria/HSX/QHS_vac_ns201_fixed/neo_out.HSX_QHS_vacuum_ns101'

for i, out_dir_qaorqh in enumerate(out_dirs):
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111)
    # HSX
    token_HSX = open(hsx_neo,'r')
    linestoken=token_HSX.readlines()
    eps_eff=[]
    s_radial=[]
    for x in linestoken:
        s_radial.append(float(x.split()[0])/100)
        eps_eff.append(float(x.split()[1])**(2/3))
    token_HSX.close()
    s_radial = np.array(s_radial)
    eps_eff = np.array(eps_eff)
    s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
    eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
    plt.plot(s_radial,eps_eff, label='HSX', linewidth=2.0)
    # W7X
    token_W7X = open(w7x_neo,'r')
    linestoken=token_W7X.readlines()
    eps_eff=[]
    s_radial=[]
    for x in linestoken:
        s_radial.append(float(x.split()[0])/150)
        eps_eff.append(float(x.split()[1])**(2/3))
    token_W7X.close()
    s_radial = np.array(s_radial)
    eps_eff = np.array(eps_eff)
    s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
    eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
    plt.plot(s_radial,eps_eff, label='W7-X', linewidth=2.0)
    for j, out_dir in enumerate(out_dir_qaorqh):
        os.chdir(os.path.join(this_path,out_dir,'see_min'))
        token = open('neo_out.out','r')
        linestoken=token.readlines()
        eps_eff=[]
        s_radial=[]
        for x in linestoken:
            s_radial.append(float(x.split()[0])/100)
            eps_eff.append(float(x.split()[1])**(2/3))
        token.close()
        s_radial = np.array(s_radial)
        eps_eff = np.array(eps_eff)
        s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
        eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
        plt.plot(s_radial,eps_eff, ('--' if j==0 else ':'), label=('QA' if i==0 else 'QH')+(' + ITG' if j==0 else ' only'), linewidth=2.0)
        os.chdir(this_path)
    #
    ax.set_yscale('log')
    plt.xlabel(r'$s=\psi/\psi_b$', fontsize=22)
    plt.ylabel(r'$\epsilon_{eff}$', fontsize=22)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=18)
    plt.tight_layout()
    fig.savefig('neo_out_'+('QA' if i==0 else 'QH')+'.pdf', dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
# plt.tight_layout()
# fig.savefig('neo_out.pdf', dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)
# plt.close()

# fig = plt.figure(figsize=(8, 6), dpi=200)
# ax = plt.subplot(111)
for i, out_dir_qaorqh in enumerate(out_dirs):
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = plt.subplot(111)
    for j, out_dir in enumerate(out_dir_qaorqh):
        os.chdir(os.path.join(this_path,out_dir,'see_min'))
        loss_fractions = np.loadtxt('loss_history.dat')
        loss_fractions_time = loss_fractions[:,0]
        loss_fractions = loss_fractions[:,1]
        plt.plot(loss_fractions_time,loss_fractions,label=('QA' if i==0 else 'QH')+(' + ITG' if j==0 else ' only'), linewidth=2.0)
        os.chdir(this_path)
    plt.ylabel('Loss fraction', fontsize=22)
    plt.xlabel('Time (s)', fontsize=22)
    plt.legend(fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    plt.savefig('loss_fractions_'+('QA' if i==0 else 'QH')+'.pdf')
    plt.close()
# plt.tight_layout()
# plt.savefig('loss_fractions.pdf')
# plt.close()