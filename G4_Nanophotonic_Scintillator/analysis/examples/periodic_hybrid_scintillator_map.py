import sys
sys.path.insert(0, '../')
import numpy as np
from utils.utils import *
import random
import scipy
from design.hybrid_scintillator import *
import torch

def k_to_absorption_coeff(k):
    lambda_0 = 430e-6
    abs_coeff = 4 * np.pi * k / lambda_0
    return abs_coeff

def main():
    global theory_N_scint
    # Constants
    pairs = 5
    substrateThickness = 0.0001e-3
    scintillatorThicknesses = np.logspace(-4, -1, 30*(multiprocessing.cpu_count() - 1))
    slThicknesses = np.logspace(-4, -1, 30*(multiprocessing.cpu_count() - 1))
    lightAbsorption = 0.015

    mu_Ti = 1.107e2
    mu_Cl = 5.725e1
    mu_C  = 2.373
    mu_O  = 5.952
    TiFractions = 0.3923
    OFractions = 0.4066
    ClFractions = 0.0719
    CFractions = 0.1290
    density_sl = 1.3 + TiFractions/0.4 #g/cm^3
    mu_gamma_sl = density_sl * 1e-4 * (mu_Ti * TiFractions + mu_O * OFractions + mu_Cl * ClFractions + mu_C * CFractions)
    mu_gamma_scint = 1e-4 * 1.02 * (2.562e-1)
    mu_electron = 10.7
    mu_electron_scint = 8.
    mu_sl_l = k_to_absorption_coeff(lightAbsorption)*1e-3
    mu_l_scint = 0.
    C_sl = 1.
    C_scint = 1.

    def theory_N(scintillatorThickness, slThickness, k):
        theoryPhotons = N_l_total(torch.Tensor(k * [scintillatorThickness]), torch.Tensor(k * [slThickness]), mu_gamma_scint, mu_electron_scint, mu_l_scint, mu_gamma_sl, mu_electron, mu_sl_l, C_sl, C_scint)
        return theoryPhotons

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    theory_photons_N_path = '/home/vboxuser/Projects/G4NS/G4_Nanophotonic_Scintillator/build/theory_map.npy'
    generate_new = False
    if generate_new:
        theory_photons_N = []
        for scint_t in scintillatorThicknesses:
            def theory_N_scint(sl_th): 
                return theory_N(scint_t*1e3, sl_th*1e3, pairs)
            theory_photons_N.append(np.array(parallel_for_loop(theory_N_scint, slThicknesses)))
        theory_photons_N = np.array(theory_photons_N)
        np.save(theory_photons_N_path, theory_photons_N)
    else:
        theory_photons_N = np.load(theory_photons_N_path)

    theory_photons_N /= theory_photons_N[0][0]
    plot = ax.matshow(theory_photons_N, interpolation='bilinear', cmap='pink')

    ax.set_xlabel('Stopping layer thickness [um]')
    ax.set_ylabel('Scintillator layer thickness [um]')
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    c_b = plt.colorbar(plot)
    line_width = 3
    ax.legend()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(line_width)
    c_b.outline.set_linewidth(line_width)

    # increase tick width
    ax.tick_params(width=line_width)
    plt.show()


if __name__ == '__main__':
    main()