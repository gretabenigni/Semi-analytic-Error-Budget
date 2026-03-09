#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:17:28 2025

@author: greta
"""

# pylint: disable=C

import numpy as np 

from src.Functions import turbulence_psd
from src.Functions import variance
from src.Functions import funct_d2
from src.Functions import total_variance
from src.Functions import interpolate_and_normalize_psd
from src.Functions import load_parameters
from src.Functions import load_PSD_windshake

from src.plots import plot_total_variance_mode_0
from src.plots import plot_all_PSD
from src.plots import check
from src.plots import plot_PSD_alias_mode_0
from src.plots import plot

param = load_parameters('params_mod4.yaml')

if param is None: 
    
    raise RuntimeError("Parameters not loaded")                                                                  
      
print("Parameters loaded successfully.")
  
n_actuators = param['telescope']['N_act']
Telescope_diameter = param['telescope']['telescope_diam']
aperture_radius = param['telescope']['apert_radius']
aperture_center = param['telescope']['apert_center']
  
r0 = param['atmosphere']['Fried_parameter_ls']
L0 = param['atmosphere']['Outer_scale']
layers_altitude = param['atmosphere']['lay_altitude']
wind_speed = param['atmosphere']['wind_sp']       # wind speed list
wind_direction = param ['atmosphere']['wind_dir']
Fried_param = param ['atmosphere']['Fried_par']
WindSpeed = param ['atmosphere']['Wind_Speed']    # wind speed not list
seeing_ = param ['atmosphere']['Seeing']
 
rho = param['source']['radial_distance']
theta = param['source']['deg']
  
value_F_excess_noise = param['wavefront_sensor']['value_for_F_excess_noise']
F_excess_noise = np.sqrt(value_F_excess_noise)
sky_background = param['wavefront_sensor']['sky_backgr']
dark_current = param['wavefront_sensor']['dark_curr']
readout_noise = param['wavefront_sensor']['noise_readout']
  
file_path_R1 = param['files']['file_path_reconstruction_matrix1']
file_path_wind1 = param['files']['file_path_PSD_windshake1']

d1 = param['polynomial_coefficients_array']['d_1']
d3 = param['polynomial_coefficients_array']['d_3']
n1 = param['polynomial_coefficients_array']['n_1']
n2 = param['polynomial_coefficients_array']['n_2']
n3 = param['polynomial_coefficients_array']['n_3']
  
spatial_freqs_minimum = param['frequency_ranges']['spatial_freqs_min']
spatial_freqs_maximum = param['frequency_ranges']['spatial_freqs_max']
spatial_freqs_number = param['frequency_ranges']['spatial_freqs_n']
spatial_freqs = np.logspace(spatial_freqs_minimum, spatial_freqs_maximum, spatial_freqs_number)
temporal_freqs_minimum = param['frequency_ranges']['temporal_freqs_min']
temporal_freqs_maximum = param['frequency_ranges']['temporal_freqs_max']
temporal_freqs_number = param['frequency_ranges']['temporal_freqs_n']
temporal_freqs = np.logspace(temporal_freqs_minimum, temporal_freqs_maximum, temporal_freqs_number)
omega_temporal_freqs = 2 * np.pi * temporal_freqs
  
t_0 = param['loop parameters']['sampling_time']
rho_sens = param['loop parameters']['sensor_sensitivity']
T_tot = param['loop parameters']['total_delay']
gain_minimum = param['loop parameters']['gain_min']
g_maximum_mapping = {                                                          
    1: 2.0,                                                                    
    2: 1.0,
    3: 0.6,
    4: 0.4
}
gain_maximum = g_maximum_mapping.get(T_tot)
gain_number = param['loop parameters']['gain_n']
gain_ = np.linspace(gain_minimum, gain_maximum, gain_number)
coeff_for_modulation_radius = param['loop parameters']['Coeff_Modulation_Radius']
Modulation_Radius = coeff_for_modulation_radius
Maximum_Rad_Ord_Corr = param['loop parameters']['Maximum_Radial_Order_Corrected']

fitting_coeff = param['coefficients']['fitting_coefficient']
alpha_ = param['coefficients']['Alpha']
  
phot_flux = float(param['pixel_params']['flux_photons'])                       
FrameRate = param['pixel_params']['frm_rate']
Magnitudo = param['pixel_params']['magn']
n_subapert = param ['pixel_params']['number_of_sub']
CollectingArea = param ['pixel_params']['collect_area']
x_pixel = param['pixel_params']['pixel_position']

file_optg = param['files']['file_optg']


display = True


freq, PSD_wind_vib = load_PSD_windshake(file_path_wind1)

if (freq is None and PSD_wind_vib is None) or (freq is None or PSD_wind_vib is None):                                     
    
    raise RuntimeError("PSD windshake or corresponding frequencies not loaded") 

print("PSD windshake and corresponding frequencies loaded successfully.")

PSD_atmosf = turbulence_psd(rho, theta, aperture_radius, aperture_center, r0, L0, layers_altitude, 
                            wind_speed, wind_direction, spatial_freqs, temporal_freqs)

d2 = funct_d2 (T_tot)


var_fit = variance(omega_temporal_freqs, t_0, gain_, n1, n2, n3,
                   d1, d2, d3, "fitting", n_actuators, Telescope_diameter, Fried_param,
                   F_excess_noise, sky_background, dark_current, readout_noise, phot_flux,            
                   FrameRate, Magnitudo, n_subapert, CollectingArea,  x_pixel, fitting_coeff,
                   alpha_, seeing_, Modulation_Radius, WindSpeed, Maximum_Rad_Ord_Corr,
                   None, file_optg, PSD_tur=None, PSD_vib=None, file_path_matrix_R=None) 


if np.array_equal(temporal_freqs, freq): 
    
    var_temp, PSD_out_temp, PSD_in_temp, H_r_temp = variance(omega_temporal_freqs, t_0,gain_, n1, n2, n3, d1, d2, d3, 
                                                             "temp",n_actuators, Telescope_diameter, Fried_param, 
                                                             F_excess_noise, sky_background, dark_current, 
                                                             readout_noise, phot_flux, FrameRate, Magnitudo, 
                                                             n_subapert, CollectingArea, x_pixel, None, alpha_, seeing_, 
                                                             Modulation_Radius, WindSpeed, Maximum_Rad_Ord_Corr, 'H_r', file_optg,
                                                             PSD_tur=PSD_atmosf,  PSD_vib=PSD_wind_vib, file_path_matrix_R=None) 
    
else:
    
    PSD_wind_vib_interp_norm = interpolate_and_normalize_psd(temporal_freqs, freq, PSD_wind_vib, n_actuators)
    #print (PSD_wind_vib_interp_norm)
    var_temp, PSD_out_temp, PSD_in_temp, H_r_temp = variance(omega_temporal_freqs, t_0, gain_, n1, n2, n3, 
                                                             d1, d2, d3, "temp",n_actuators, Telescope_diameter, 
                                                             Fried_param, F_excess_noise, sky_background, dark_current, 
                                                             readout_noise, phot_flux, FrameRate, Magnitudo, 
                                                             n_subapert, CollectingArea, x_pixel, None, alpha_, seeing_, 
                                                             Modulation_Radius, WindSpeed, Maximum_Rad_Ord_Corr, 'H_r', file_optg,
                                                             PSD_tur=PSD_atmosf, PSD_vib=PSD_wind_vib_interp_norm, 
                                                             file_path_matrix_R=None) 
  




    
var_alias, PSD_out_alias, PSD_in_alias, H_n_alias = variance(omega_temporal_freqs, t_0, gain_,n1, n2, n3, 
                                                             d1, d2, d3, "alias", n_actuators, Telescope_diameter, 
                                                             Fried_param, F_excess_noise, sky_background, dark_current, 
                                                             readout_noise, phot_flux, FrameRate, Magnitudo, 
                                                             n_subapert, CollectingArea, x_pixel, None, alpha_, seeing_, 
                                                             Modulation_Radius, WindSpeed, Maximum_Rad_Ord_Corr, 'H_n', file_optg,
                                                             PSD_tur=None, PSD_vib=None, file_path_matrix_R=file_path_R1)
   

        
var_meas, PSD_out_meas, PSD_in_meas, H_n_meas = variance(omega_temporal_freqs, t_0,
                                                         gain_, n1, n2, n3, d1, d2, d3, "meas", 
                                                         n_actuators, Telescope_diameter, Fried_param, F_excess_noise,
                                                         sky_background, dark_current, readout_noise, phot_flux, FrameRate, Magnitudo, 
                                                         n_subapert, CollectingArea, x_pixel, None, alpha_, seeing_, 
                                                         Modulation_Radius, WindSpeed, Maximum_Rad_Ord_Corr, 'H_n', file_optg,
                                                         PSD_tur=None, PSD_vib=None, file_path_matrix_R=file_path_R1) 


var_total = total_variance(var_fit, var_temp, var_alias, var_meas)

if display:

    plot_total_variance_mode_0(gain_minimum, gain_maximum, omega_temporal_freqs, t_0, n1, n2, 
                           n3, d1, d2, d3, Telescope_diameter, Fried_param, F_excess_noise, 
                           sky_background, dark_current, readout_noise, phot_flux, FrameRate, Magnitudo, 
                           n_subapert, CollectingArea, x_pixel, fitting_coeff, alpha_, seeing_, 
                           Modulation_Radius, WindSpeed, Maximum_Rad_Ord_Corr, file_path_R1, file_optg,
                           PSD_atmosf, PSD_wind_vib)


    plot(omega_temporal_freqs, H_r_temp, H_n_meas, H_n_alias, PSD_in_temp, PSD_out_temp,
         PSD_in_meas, PSD_out_meas, PSD_in_alias, PSD_out_alias)
       

    plot_all_PSD(omega_temporal_freqs, PSD_out_temp, PSD_out_meas, PSD_out_alias)


    check(file_path_R1, Telescope_diameter, seeing_, Modulation_Radius, 
          n_actuators, alpha_, omega_temporal_freqs, WindSpeed, Maximum_Rad_Ord_Corr,
          file_optg, system="ANDES")


    plot_PSD_alias_mode_0(n_actuators, omega_temporal_freqs, alpha_, Telescope_diameter,
                          seeing_, Modulation_Radius, WindSpeed, Maximum_Rad_Ord_Corr,
                          file_path_R1, file_optg, system="ANDES")







