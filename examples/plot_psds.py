#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from src.Functions import (
    load_parameters, 
    turbulence_psd, 
    temporal_variance,
    aliasing_variance,
    measure_variance,
    build_transfer_function,
    compute_andes_optical_gain,
    funct_d2
)

def plot_system_psds(mode_index=0, plot_inputs=False):
    print(f"\nGenerating PSD plot for Mode {mode_index}...")

    # 1. Load parameters
    param = load_parameters('params_mod4.yaml')

    n_actuators = param['telescope']['N_act']
    D = param['telescope']['telescope_diam']
    aperture_radius = param['telescope']['apert_radius']
    aperture_center = param['telescope']['apert_center']

    r0 = param['atmosphere']['Fried_par']
    L0 = param['atmosphere']['Outer_scale']
    layers_altitude = param['atmosphere']['lay_altitude']
    wind_direction = param['atmosphere']['wind_dir']
    wind_speed = param['atmosphere']['Wind_Speed']
    seeing = param['atmosphere']['Seeing']

    F_excess_noise = np.sqrt(param['wavefront_sensor']['value_for_F_excess_noise'])
    sky_background = param['wavefront_sensor']['sky_backgr']
    dark_current = param['wavefront_sensor']['dark_curr']
    readout_noise = param['wavefront_sensor']['noise_readout']

    file_path_R1 = param['files']['file_path_reconstruction_matrix1']
    file_mod0 = param['files']['file_optg'][0]
    file_mod4 = param['files']['file_optg'][1]

    d1 = param['polynomial_coefficients_array']['d_1']
    d3 = param['polynomial_coefficients_array']['d_3']
    n1 = param['polynomial_coefficients_array']['n_1']
    n2 = param['polynomial_coefficients_array']['n_2']
    n3 = param['polynomial_coefficients_array']['n_3']

    frame_rate = param['pixel_params']['frm_rate']
    temporal_freqs = np.logspace(-3, np.log10(frame_rate / 2.0), 1000)
    omega = 2 * np.pi * temporal_freqs

    spatial_freqs_min = param['frequency_ranges']['spatial_freqs_min']
    spatial_freqs_max = param['frequency_ranges']['spatial_freqs_max']
    spatial_freqs_n = param['frequency_ranges']['spatial_freqs_n']
    spatial_freqs = np.logspace(spatial_freqs_min, spatial_freqs_max, spatial_freqs_n)

    t_0 = param['loop parameters']['sampling_time']
    T_tot = param['loop parameters']['total_delay']
    Modulation_Radius = param['loop parameters']['Coeff_Modulation_Radius']
    Maximum_Rad_Ord_Corr = param['loop parameters']['Maximum_Radial_Order_Corrected']
    alpha_ = param['coefficients']['Alpha']

    phot_flux = float(param['pixel_params']['flux_photons'])                 
    Magnitudo = param['pixel_params']['magn']
    n_subapert = param['pixel_params']['number_of_sub']
    CollectingArea = param['pixel_params']['collect_area']
    x_pixel = param['pixel_params']['pixel_position']

    gain_mapping = {1: 2.0, 2: 1.0, 3: 0.6, 4: 0.4}
    gain_max = gain_mapping.get(T_tot, 0.5)
    gain_array = np.full(n_actuators, gain_max)

    d2 = funct_d2(T_tot)

    # 2. Generate Atmospheric Input
    PSD_atmosf = turbulence_psd(0, 0, aperture_radius, aperture_center, r0, L0, layers_altitude,
                                wind_speed, wind_direction, spatial_freqs, temporal_freqs, n_modes=n_actuators)
    PSD_vibration_zeros = np.zeros_like(PSD_atmosf)

    # 3. Build Transfer Functions
    H_r = build_transfer_function(gain_array, omega, t_0, n_actuators, n1, n2, n3, d1, d2, d3, "H_r")
    H_n = build_transfer_function(gain_array, omega, t_0, n_actuators, n1, n2, n3, d1, d2, d3, "H_n")

    # 4. Compute Optical Gain (needed for aliasing)
    c_optg = compute_andes_optical_gain(file_mod0, file_mod4, seeing, Modulation_Radius)

    # 5. Extract PSDs from separate functions
    _, PSD_out_temp, PSD_in_temp = temporal_variance(
        PSD_atmosf, PSD_vibration_zeros, H_r, n_actuators, omega
    )

    _, PSD_out_alias, PSD_in_alias = aliasing_variance(
        H_n, n_actuators, omega, alpha_, D, seeing, Modulation_Radius, wind_speed,
        Maximum_Rad_Ord_Corr, file_path_R1, c_optg
    )

    _, PSD_out_meas, PSD_in_meas = measure_variance(
        F_excess_noise, x_pixel, sky_background, dark_current, readout_noise,
        phot_flux, D, frame_rate, Magnitudo, n_subapert, CollectingArea,
        file_path_R1, omega, H_n, n_actuators
    )

    PSD_in_temp = np.real(PSD_in_temp)
    PSD_in_alias = np.real(PSD_in_alias)
    PSD_in_meas = np.real(PSD_in_meas)
    PSD_out_temp = np.real(PSD_out_temp)
    PSD_out_alias = np.real(PSD_out_alias)
    PSD_out_meas = np.real(PSD_out_meas)

    # Variance calculation
    var_temp_in = integrate.simpson(PSD_in_temp[mode_index, :], temporal_freqs)
    var_alias_in = integrate.simpson(PSD_in_alias[mode_index, :], temporal_freqs)
    var_meas_in = integrate.simpson(PSD_in_meas[mode_index, :], temporal_freqs)
    var_temp_out = integrate.simpson(PSD_out_temp[mode_index, :], temporal_freqs)
    var_alias_out = integrate.simpson(PSD_out_alias[mode_index, :], temporal_freqs)
    var_meas_out = integrate.simpson(PSD_out_meas[mode_index, :], temporal_freqs)

    # Print variances for debug
    print(f"Turbulence Variance (Input):    {var_temp_in:.2e} nm²")
    print(f"Aliasing Variance (Input):      {var_alias_in:.2e} nm²")
    print(f"Noise Variance (Input):         {var_meas_in:.2e} nm²")
    print(f"Turbulence Variance (Output):   {var_temp_out:.2e} nm²")
    print(f"Aliasing Variance (Output):     {var_alias_out:.2e} nm²")
    print(f"Noise Variance (Output):        {var_meas_out:.2e} nm²")

    # 6. Create Plot
    plt.figure(figsize=(12, 7))
    freqs = temporal_freqs

    if plot_inputs:
        plt.loglog(freqs, PSD_in_temp[mode_index, :], label='Turbulence (Open Loop)',
                   color='tab:blue', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_alias[mode_index, :], label='Aliasing (Open Loop)',
                   color='tab:orange', linestyle='--', alpha=0.6)
        plt.loglog(freqs, PSD_in_meas[mode_index, :], label='Noise (Open Loop)',
                   color='tab:green', linestyle='--', alpha=0.6)

    plt.loglog(freqs, PSD_out_temp[mode_index, :], label='Turbulence Residual (Servo-lag)',
               color='tab:blue', linewidth=2.5)
    plt.loglog(freqs, PSD_out_alias[mode_index, :], label='Aliasing Residual',
               color='tab:orange', linewidth=2.5)
    plt.loglog(freqs, PSD_out_meas[mode_index, :], label='Noise Residual',
               color='tab:green', linewidth=2.5)

    psd_total = (PSD_out_temp[mode_index, :] +
                 PSD_out_alias[mode_index, :] +
                 PSD_out_meas[mode_index, :])

    plt.loglog(freqs, psd_total, label='Total PSD (Sum)',
               color='black', linewidth=3, linestyle='-.')

    plt.title(f"Spectral Analysis (PSD) in Closed Loop - Zernike Mode {mode_index}"
              f"\nLoop Gain = {gain_max:.2f}", fontsize=14)
    plt.xlabel("Temporal Frequency [Hz]", fontsize=12)
    plt.ylabel("Power Spectral Density [nm² / Hz]", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_system_psds(mode_index=0, plot_inputs=True)
