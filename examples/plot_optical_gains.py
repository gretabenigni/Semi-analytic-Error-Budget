#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to plot Modal Optical Gains.
Generates plots similar to Figure 7 in Agapito et al. (2023)
and additional Optical Gain vs Seeing curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.Functions import compute_optical_gain, load_parameters

def plot_modal_optical_gains():
    # 1. Load configuration parameters
    param = load_parameters('params_ANDES.yaml')
    file_mod0 = param['data']['optical_gain_models'][0]
    file_mod4 = param['data']['optical_gain_models'][1]
    n_modes = param['control']['n_modes']

    # 2. Define the physical grid to explore
    seeing_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    
    # Modulation radii values for separate figures
    modulation_radii = [0.0, 3.0] 

    # X-axis (Mode index, starting from 1 to match the paper's 1-based indexing in plots)
    mode_axis = np.arange(1, n_modes + 1)

    # Consistent colors for the seeing curves (avoiding yellow for better visibility)
    colors_seeing = ['black', 'blue', 'cyan', 'green', 'orange', 'red']

    # =========================================================================
    # PLOT TYPE 1: Optical Gain vs Mode Number (for discrete seeing values)
    # =========================================================================
    for r_mod in modulation_radii:
        plt.figure(figsize=(9, 6))

        for s_idx, seeing in enumerate(seeing_values):
            
            # Extract the column vector of optical gains
            c_optg = compute_optical_gain(
                file_mod0=file_mod0,
                file_mod1=file_mod4,
                seeing=seeing,
                modulation_radius=r_mod,
                actuators_number=n_modes
            )

            # Flatten to 1D for plotting
            c_optg_1d = c_optg.flatten()

            plt.plot(
                mode_axis,
                c_optg_1d,
                color=colors_seeing[s_idx % len(colors_seeing)],
                label=f'seeing={seeing:.2f}"'
            )

        # Formatting with logarithmic scale on X-axis
        plt.xscale('log')
        plt.ylim(0, 1.0)
        
        plt.xlabel('Mode number')
        plt.ylabel('Optical gains')
        plt.title(f'Modal Optical Gains vs Modes - Modulation Radius: {r_mod} $\\lambda/D$')
        
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()

    # =========================================================================
    # PLOT TYPE 2: Optical Gain vs Seeing (for specific modes)
    # =========================================================================
    # Target modes to plot (0-indexed)
    desired_modes = [0, 10, 100, 1000, -1]  # -1 will be handled as the last mode
    
    # Check that requested modes exist in the currently loaded configuration
    valid_modes = [m for m in desired_modes if m < n_modes]
    
    # Denser seeing grid for a smooth curve
    dense_seeing = np.linspace(0.4, 1.4, 50)
    colors_modes = ['black', 'red', 'limegreen', 'blue', 'orange']  # Consistent colors for modes

    for r_mod in modulation_radii:
        plt.figure(figsize=(9, 6))

        # Pre-allocate array to store gains: shape (len(dense_seeing), n_modes)
        gains_over_seeing = np.zeros((len(dense_seeing), n_modes))

        # Evaluate the optical gains over the dense seeing grid
        for i, s in enumerate(dense_seeing):
            c_optg = compute_optical_gain(
                file_mod0=file_mod0,
                file_mod1=file_mod4,
                seeing=s,
                modulation_radius=r_mod,
                actuators_number=n_modes
            )
            gains_over_seeing[i, :] = c_optg.flatten()

        # Plot the curves for the validated modes
        for m_idx, mode in enumerate(valid_modes):
            plt.plot(
                dense_seeing,
                gains_over_seeing[:, mode],
                color=colors_modes[m_idx % len(colors_modes)],
                linewidth=2,
                label=f'Mode {mode}'
            )

        plt.xlim(0.4, 1.4)
        plt.ylim(0, 1.0)
        
        plt.xlabel('Seeing [arcsec]')
        plt.ylabel('Optical gains')
        plt.title(f'Optical Gain vs Seeing - Modulation Radius: {r_mod} $\\lambda/D$')
        
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()

    # Show all generated figures
    plt.show()

if __name__ == '__main__':
    plot_modal_optical_gains()