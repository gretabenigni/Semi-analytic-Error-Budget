#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.io import fits

from src.Functions import (
    compute_andes_optical_gain,
    PSD_aliasing,
    load_parameters,
    read_sigma_slopes,
    double_interpolation_sigma_slope,
    extract_propagation_coefficients
)

def verify_aliasing_energy():
    print("\n" + "="*80)
    print(" VERIFY ALIASING ENERGY CONSERVATION")
    print(" (Variance from Slopes vs PSD Integral SA vs PSD Integral PASSATA)")
    print("="*80 + "\n")

    # 1. Load Parameters
    param = load_parameters('params_mod4.yaml')
    D = param['telescope']['telescope_diam']
    r0 = param['atmosphere']['Fried_par']
    wvl_ref = 500e-9
    seeing = 0.98 * wvl_ref / r0 * (3600 * 180 / np.pi)
    modulation_radius = param['loop parameters']['Coeff_Modulation_Radius']
    alpha = param['coefficients']['Alpha']
    wind_speed = param['atmosphere']['Wind_Speed']
    max_rad_ord = param['loop parameters']['Maximum_Radial_Order_Corrected']
    n_actuators = 10

    file_mod0 = param['files']['file_optg'][0]
    file_mod4 = param['files']['file_optg'][1]
    file_reconstructor = param['files']['file_path_reconstruction_matrix1']

    frame_rate = param['pixel_params']['frm_rate']
    temporal_freqs = np.logspace(-3, np.log10(frame_rate / 2.0), 1000)
    omega_sa = 2 * np.pi * temporal_freqs

    # 2. Compute Optical Gain
    c_optg = compute_andes_optical_gain(file_mod0, file_mod4, seeing, modulation_radius)

    # --------------------------------------------------------------------------------
    # METHOD 1: Direct target variance
    # ---------------------------------------------------------------------------
    data_slopes = read_sigma_slopes() # No arguments, uses internal path
    seeing_vals = data_slopes[0,0,:]                                           
    modal_radius_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0])

    sigma_slope_alias = double_interpolation_sigma_slope(
        modal_radius_vals, seeing_vals, data_slopes,
        modulation_radius, seeing
    )

    p_coeff = extract_propagation_coefficients(file_reconstructor)
    p_coeff = np.squeeze(p_coeff)

    var_from_slopes = (sigma_slope_alias**2) * (c_optg**2) * p_coeff

    # --------------------------------------------------------------------------------
    # METHOD 2: Numerical integral of SA PSD
    # --------------------------------------------------------------------------------
    psd_sa = PSD_aliasing(
        n_actuators, omega_sa, alpha, D, seeing, modulation_radius, 
        wind_speed, max_rad_ord, file_reconstructor, c_optg
    )
    var_from_psd = integrate.simpson(psd_sa, omega_sa, axis=-1)

    # --------------------------------------------------------------------------------
    # METHOD 3: Integral of empirical PASSATA PSD
    # --------------------------------------------------------------------------------
    file_psd_passata = "src/file_fits/ANDES/modal_psd_aliasing.fits"
    try:
        with fits.open(file_psd_passata) as hdul:
            psd_passata = hdul[0].data

        freq_passata = np.linspace(0, frame_rate / 2.0, psd_passata.shape[1])
        omega_passata = 2 * np.pi * freq_passata
        var_passata = integrate.simpson(psd_passata, omega_passata, axis=-1)
        passata_available = True
    except FileNotFoundError:
        print(f"[!] File {file_psd_passata} not found. Skipping End-to-End comparison.")
        passata_available = False

    # --------------------------------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------------------------------
    print(f"shape var_from_psd: {var_from_psd.shape}, shape omega_sa: {omega_sa.shape}")
    print(f"{'Mode':<6} | {'Var from Slopes (SA)':<22} | {'Var PSD Integral (SA)':<22}"
          f" | {'Var PASSATA Integral':<22}")
    print("-" * 80)

    for i in range(min(10, len(var_from_slopes))):
        pass_val = f"{var_passata[i]:.4e}" if passata_available else "N/A"
        print(f"{i:<6} | {var_from_slopes[i]:<22.4e} | {var_from_psd[i]:<22.4e} | {pass_val:<22}")

    print("-" * 80)
    sum_passata = f"{np.sum(var_passata):.4e}" if passata_available else "N/A"
    print(f"{'TOTAL':<6} | {np.sum(var_from_slopes):<22.4e} | {np.sum(var_from_psd):<22.4e}"
          f" | {sum_passata:<22}\n")

if __name__ == "__main__":
    verify_aliasing_energy()
