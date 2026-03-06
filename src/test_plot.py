#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:11:30 2026

@author: greta
"""

# pylint: disable=C


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.io import fits                                                    

from src.Functions import total_variance
from src.Functions import compute_andes_optical_gain
#from src.Functions import compute_soul_optical_gain
from src.Functions import extract_propagation_coefficients
from src.Functions import PSD_aliasing
from src.Functions import variance
from src.Functions import double_interpolation_sigma_slope
from src.Functions import read_sigma_slopes


# Function to compute the total residual variance for a set of gain values,
# by combining fitting, temporal, aliasing and measurement error contributions 
  
def variance_total_for_test(number_of_actuators, gain_value_, omega_temp_freq_interval, t_0, num1, num2, 
                            num3, den1, den2, den3, telescope_diameter, Fried_parameter, F_excess, 
                            sky_bkg, dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
                            n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, 
                            seeing, modulation_radius, windspeed, maximum_radial_order_corrected,
                            file_path_reconstruction_matrix, file_optg, PSD_turbolence, PSD_vibration_wind):
    
                                                            
    tot_variance = np.zeros_like(gain_value_, dtype=float)
    
    for i in range(len(gain_value_)): 
        
        g = gain_value_[i]                                                     
        gain_val = np.array([g])                                               

        variance_fit = variance(omega_temp_freq_interval, t_0, gain_val, num1, num2, 
                                num3, den1, den2, den3, "fitting", number_of_actuators, 
                                telescope_diameter, Fried_parameter, F_excess, sky_bkg, 
                                dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
                                n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, 
                                seeing,modulation_radius, windspeed, maximum_radial_order_corrected, 
                                None, file_optg, PSD_tur=None, PSD_vib=None, file_path_matrix_R=None)
    
        variance_temporal,_ , _, _ = variance(omega_temp_freq_interval, t_0, gain_val, 
                                              num1, num2, num3, den1, den2, den3, "temp", 
                                              number_of_actuators, telescope_diameter, Fried_parameter, 
                                              F_excess, sky_bkg, dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
                                              n_subaperture, collecting_area, pixel_pos, 
                                              fitting_coeff, alpha, seeing, modulation_radius, 
                                              windspeed, maximum_radial_order_corrected,'H_r', file_optg,
                                              PSD_tur=PSD_turbolence, PSD_vib=PSD_vibration_wind, file_path_matrix_R=None)
    
        variance_aliasing, _, _, _ = variance(omega_temp_freq_interval, t_0, gain_val, num1, num2, 
                                              num3, den1, den2, den3, "alias", number_of_actuators, 
                                              telescope_diameter, Fried_parameter, F_excess, sky_bkg,
                                              dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
                                              n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, seeing, 
                                              modulation_radius, windspeed, maximum_radial_order_corrected,
                                              'H_n', file_optg, PSD_tur=None, PSD_vib=None, file_path_matrix_R=file_path_reconstruction_matrix)
    
        variance_measurement, _, _, _ = variance(omega_temp_freq_interval, t_0, gain_val, num1, num2, 
                                                 num3, den1, den2, den3, "meas", number_of_actuators, 
                                                 telescope_diameter, Fried_parameter, F_excess, sky_bkg, 
                                                 dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
                                                 n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, seeing, 
                                                 modulation_radius, windspeed, maximum_radial_order_corrected,
                                                 'H_n', file_optg, PSD_tur=None, PSD_vib=None, file_path_matrix_R=file_path_reconstruction_matrix)
        
        tot_variance[i] = total_variance(np.real(variance_fit), np.real(variance_temporal), 
                                         np.real(variance_measurement), np.real(variance_aliasing))            
    
    return tot_variance


# Function to plot the total residual variance of the system as a function 
# of the gain, considering only the first mode.

def test(gain_min, gain_max, omega_temp_freq_interval, t_0, num1, num2, 
         num3, den1, den2, den3, telescope_diameter, Fried_parameter, F_excess, 
         sky_bkg, dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
         n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, seeing, 
         modulation_radius, windspeed, maximum_radial_order_corrected,
         file_path_reconstruction_matrix, file_optg, PSD_turbolence, PSD_vibration_wind): 
   
    do_test = input("Do you want to do the test? (y/n):") 

    if do_test == "y":
       
        print ('TEST') 
     
        actuators_number = 1                                                  
     
        gain_value = np.arange (gain_min, gain_max, 0.1)
        variance_total = variance_total_for_test(actuators_number, gain_value, omega_temp_freq_interval,
                                                 t_0, num1, num2, num3, den1, den2, den3, telescope_diameter, 
                                                 Fried_parameter, F_excess, sky_bkg, dark_curr,read_out_noise, 
                                                 photon_flux, frame_rate, magnitudo, n_subaperture, collecting_area, 
                                                 pixel_pos, fitting_coeff, alpha, seeing, 
                                                 modulation_radius, windspeed, maximum_radial_order_corrected,
                                                 file_path_reconstruction_matrix, file_optg,
                                                 PSD_turbolence, PSD_vibration_wind)
        
        plt.plot(gain_value, variance_total, marker='o')    
        plt.xlabel('Gain')
        plt.ylabel('Total variance')
        plt.title('Total variance as a function of the gain')
        plt.grid()
        plt.show()
       
        
# Defines a function that allows, when needed, to plot PSD_in, PSD_out, and the transfer 
# function for the variances (temp, alias, meas)

def plot(f, H_r_t, H_n_m, H_n_a, PSD_in_t, PSD_out_t, PSD_in_m, PSD_out_m, PSD_in_a, PSD_out_a):
    
    do_plot = input("Do you want to plot PSD_in, PSD_out, and the transfer function? (y/n):")

    if do_plot == "y":
      
        PSD_in = [PSD_in_t, PSD_in_m, PSD_in_a]                  
        PSD_out = [PSD_out_t, PSD_out_m, PSD_out_a]              
        H = [H_r_t, H_n_m, H_n_a]
       
        labels_PSD = ["temp", "meas","alias"]
        labels_H = ["r", "n", "n"]
       
        for i in range(len(PSD_in)):                                                           
          
            plt.loglog(f, PSD_in[i][0, :], label=f"PSD_in_{labels_PSD[i]} (mode 0)")       
            plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels_PSD[i]} (mode 0)")
            plt.xlabel("Frequency [rad/s]")
            plt.ylabel("PSD")
            plt.title(f"PSD {labels_PSD[i]} (modo 0)")
            plt.legend()
            plt.grid()
            plt.show()
           
            plt.loglog(f, np.abs(H[i][0, :])**2)     
            plt.xlabel("Frequency [rad/s]")
            plt.ylabel(f"|H_{labels_H[i]}|^2")
            plt.title(f"Transfert function H_{labels_H[i]} (mode 0)")
            plt.grid()
            plt.show()
       
        
# Function to plot the three output PSDs (temporal, measurement and aliasing)
# for mode 0 on the same graph.
       
def plot_all_PSD(f, PSD_out_t, PSD_out_m, PSD_out_a):
    
    do_plot = input("Do you want to plot all the PSD_out (mode 0)? (y/n):")

    if do_plot == "y":

        PSD_out = [PSD_out_t, PSD_out_m, PSD_out_a]

        labels = ["temp", "meas", "alias"]

        for i in range(len(PSD_out)):

            plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels[i]} (mode 0)")

        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("PSD")
        plt.title(f"PSD {labels[i]} – mode 0")
        plt.legend()
        plt.grid()

        plt.show()


# Function to verify the consistency of the aliasing variance calculation.
# It computes the analytical aliasing variance (summing the selected modes)
# and compares it with the variance obtained by integrating the corresponding 
# aliasing PSD. 
# The function also print the variance of the first mode alone in both cases.
    
def check(file_path_matrix_R, telescope_diameter, seeing, modulation_radius,
          actuators_number, alpha, omega_temp_freq_interval, windspeed, 
          maximum_radial_order_corrected,file_optg, system="ANDES"):
    
    if system == "ANDES":
        c_optg = compute_andes_optical_gain(file_optg[0], file_optg[1], seeing, modulation_radius)
    # TODO not supported yet
    #elif system == "SOUL":
    #    gain = compute_soul_optical_gain(file_optg, mod_modes, binning, magnitude)
    
    p_coefficient = extract_propagation_coefficients(file_path_matrix_R)
    
    if p_coefficient is None:   
                                              
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.")
    
    data_slopes = read_sigma_slopes()  
    seeing_vals = data_slopes[0,0,:]                                           
  
    modal_radius_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0]) 
    
    
    
    sigma_slope_alias = double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
                                                         modulation_radius, seeing)
  
    sigma_alias_2_two_modes = 0.0
    
    for i in range (actuators_number):
        
        sigma_alias_2 = p_coefficient[i] * (sigma_slope_alias ** 2) / c_optg ** 2
        
        sigma_alias_2_two_modes += sigma_alias_2
        
    print("ALIASING VARIANCE:", sigma_alias_2_two_modes)
    
    PSD_al = PSD_aliasing (actuators_number, omega_temp_freq_interval, alpha, 
                           telescope_diameter, seeing, modulation_radius, windspeed,
                           maximum_radial_order_corrected,
                           file_path_matrix_R, c_optg)
    
    
    integral_per_mode = integrate.simpson(PSD_al, omega_temp_freq_interval)
    sigma_alias_2_PSD_total = np.sum(integral_per_mode)
    
    print("ALIASING VARIANCE FROM PSD:", sigma_alias_2_PSD_total)
    
    sigma_alias_2_one_mode = p_coefficient * (sigma_slope_alias ** 2) / c_optg ** 2
    
    print("ALIASING VARIANCE ONE MODE:", sigma_alias_2_one_mode[0])
    
    print("ALIASING VARIANCE FROM PSD ONE MODE:", integral_per_mode[0])
    
    
def plot_PSD_alias_mode_0(actuators_number, omega_temp_freq_interval, alpha, telescope_diameter,
                          seeing, modulation_radius, windspeed, maximum_radial_order_corrected,
                          file_path_matrix_R, file_optg, system="ANDES"):
    
    with fits.open("src/file_fits/ANDES/modal_psd_aliasing.fits") as hdul:
        data = hdul[0].data 
        
        freq_hz = data[:, 0]
        mode_0 = data[:, 1]
        
        freq_rad_s = 2 * np.pi * freq_hz
        if system == "ANDES":
            c_optg = compute_andes_optical_gain(file_optg[0], file_optg[1], seeing, modulation_radius)
        # TODO not supported yet
        #elif system == "SOUL":
        #    gain = compute_soul_optical_gain(file_optg, mod_modes, binning, magnitude)
        PSD_aliasing_mode0_given = mode_0 / (c_optg ** 2 * 2 * np.pi)             
        
    PSD_alising_mine = PSD_aliasing (actuators_number, omega_temp_freq_interval, alpha,  
                                     telescope_diameter, seeing, modulation_radius, windspeed, 
                                      maximum_radial_order_corrected,file_path_matrix_R, c_optg)
    
    PSD_alising_mine_mode0 = PSD_alising_mine[0,:]
    
    plt.loglog(omega_temp_freq_interval, PSD_alising_mine_mode0, label="PSD alias mine mode 0")
    plt.loglog(freq_rad_s, PSD_aliasing_mode0_given, label="PSD alias data mode 0")
    

    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("PSD")  
    plt.legend()
    plt.grid()
    plt.show()

    # N = len(freq_rad_s)   # = 501

    # omega_cut = omega_temp_freq_interval[:N]
    # PSD_mine_cut = PSD_alising_mine_mode0[:N]

    # ratio = PSD_mine_cut / PSD_aliasing_mode0_given
    # plt.figure()

    # plt.semilogx(omega_cut, ratio)
    # plt.xlabel("Frequency [rad/s]")
    # plt.ylabel("Mine / Given")
    # plt.grid()
    # plt.title("PSD ratio")

    # plt.show()
    
    
    
    
    
  

    
        
        
    
    

    

    
    
    
          
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
