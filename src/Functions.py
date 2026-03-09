#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:43:33 2025

@author: greta
"""
# pylint: disable=C

import yaml
import numpy as np
from scipy import integrate
from astropy.io import fits                                                    
from functools import reduce                                                   
import sympy as sp                                                             
from scipy.interpolate import RegularGridInterpolator                        

from arte.types.guide_source import GuideSource 
from arte.types.aperture import CircularOpticalAperture 
from arte.atmo.von_karman_covariance_calculator import VonKarmanSpatioTemporalCovariance 
from arte.atmo.cn2_profile import Cn2Profile                                                                                     


# Reads the YAML file (where parameters are listed) and returns a dictionary.

def load_parameters(yaml_file):
    
    with open(yaml_file, 'r', encoding='utf-8') as stream:                     
                                                                               
        try: 
            
            parameters = yaml.safe_load(stream)                                
            return parameters
        
        except yaml.YAMLError as exc:                                          
            
            print(exc)
            return None
        

# Function to define the d2 array, whose length depends on the value of T_total.

def funct_d2 (T_total):
    
    d2 = np.zeros(T_total + 1)
    d2[0] = 1
    
    return d2


# Function that returns the numerator and denominator of the transfer function C, 
# expressed as polynomials in Z. The function also returns the numeric definition of Z.
# For the moment, we are considering the control function C as defined below,
# which means that we are using an integral control.

def funct_C (gain, omega_temp_freq_interval, t_0):    
    
    Z_symbolic = sp.symbols('Z')                                                   
    C = (Z_symbolic * gain) / (Z_symbolic - 1)                                    
    
    # Estraggo numeratore e denominatore
    num, den = sp.fraction(C)                                                      
    
    # Ottengo i coefficienti come array NumPy
    n4 = np.array(sp.Poly(num, Z_symbolic).all_coeffs(), dtype=complex)          
    d4 = np.array(sp.Poly(den, Z_symbolic).all_coeffs(), dtype=complex)        
    
    Z_numeric = np.exp(1j * omega_temp_freq_interval * t_0)
    
    return n4, d4, Z_numeric
  
  
# Function to compute the numerator and denominator polynomials of the transfer functions H_r and H_n
# using np.polymul, np.polyadd and polyval (to evaluate them at Z).
# The expressions written below were derived from Equations (4) and (5) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna, 2019)
# through appropriate algebraic steps, allowing us to use np.polymul, np.polyadd,
# and np.polyval to construct the numerator and denominator of H_r and H_n.
# WFS = n1 / d1, Reconstructor and delay = n2 / d2, deformable Mirror = n3 / d3 and Controller = n4 / d4.

def transfer_funct(n1, n2, n3, n4, d1, d2, d3, d4, Z, transfer_function_type):    
   
    H_r_coeff_num = reduce(np.polymul, [d1, d2, d3, d4])                       
    H_r_coeff_den = np.polyadd(reduce(np.polymul, [d1, d2, d3, d4]), reduce(np.polymul, [n1, n2, n3, n4]))
    H_n_coeff_num = reduce(np.polymul, [n2, n3, n4, d1])
    H_n_coeff_den = H_r_coeff_den                                              
    
    H_r_num = np.polyval (H_r_coeff_num, Z)                                    
    H_r_den = np.polyval (H_r_coeff_den, Z)
    H_n_num = np.polyval (H_n_coeff_num, Z)
    H_n_den = np.polyval (H_n_coeff_den, Z)
    
    
    if transfer_function_type == "H_r":
        
        H_r = H_r_num/ H_r_den
        return H_r

    if transfer_function_type == "H_n":
        
        H_n = H_n_num/ H_n_den
        return H_n
    
    else: 
        raise ValueError("Transfer_function_type must be one of 'H_r' o 'H_n'")


# Function to obtain the atmospheric PSD for n_modes Zernike modes starting from tip (j=2).
# Returns a 2D array of shape (n_modes, len(tempor_freqs)).
# Default n_modes=2 reproduces the original tip-and-tilt behaviour.

def turbulence_psd(rho, theta, aperture_radius, aperture_center, r0, L0, layers_altitude,
                   wind_speed, wind_direction, space_freqs, tempor_freqs, n_modes=2):

    source = GuideSource((rho, theta), np.inf)
    aperture = CircularOpticalAperture(aperture_radius, aperture_center)
    cn2_profile = Cn2Profile.from_r0s(r0, L0, layers_altitude, wind_speed, wind_direction)

    vk = VonKarmanSpatioTemporalCovariance(
        source1=source, source2=source, aperture1=aperture, aperture2=aperture,
        cn2_profile=cn2_profile, spat_freqs=space_freqs)

    # Zernike indices: tip=2, tilt=3, focus=4, ...
    modes = list(range(2, 2 + n_modes))
    modes_psd = vk.getGeneralZernikeCPSD(j=modes, k=modes, temp_freqs=tempor_freqs)

    # Diagonal elements give the temporal PSD of each mode
    PSD_atmo = np.array([modes_psd[i, i, :] for i in range(n_modes)])

    return PSD_atmo
  

# Function to calculate the Fitting Error, see Equation (7) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna, 2019)

def fitting_variance(fitting_coeff, actuators_number, telescope_diameter, Fried_parameter):
    
    var_fitting = fitting_coeff * actuators_number ** (-0.9) * (telescope_diameter/Fried_parameter) ** (5/3)
    
    return var_fitting


# Function to compute the numerator and denominator coefficients (n4 and d4) of the transfer function C 
# for all actuators, and to return the numeric Z vector from funct_C.

def compute_n4_d4(gain, omega_temp_freq_interval, t_0, actuators_number):
    
    n4_array_example, d4_array_example, _ = funct_C(gain[0], omega_temp_freq_interval, t_0)        
 
    len_n4 =  len(n4_array_example)
    len_d4 =  len(d4_array_example)
 
    n4_array = np.zeros((actuators_number, len_n4), dtype=complex)      
    d4_array = np.zeros((actuators_number, len_d4), dtype=complex)      
       
    for i in range (actuators_number):                                   
         
        n4_array[i, :], d4_array[i, :], Z_num = funct_C (gain[i], omega_temp_freq_interval, t_0)     
         
    return n4_array, d4_array, Z_num
  

# Function to compute the transfer function H.
# The 'transfer_function_type' argument selects between two different types of transfer functions: "H_r" or "H_n".

def compute_H(actuators_number, omega_temp_freq_interval, num1, num2, num3, num4, den1, den2,
              den3, den4, Z, transfer_function_type):
    
    H = np.zeros((actuators_number, len(omega_temp_freq_interval)), dtype=complex)
    
    for i in range(actuators_number):
        
        H[i, :] = transfer_funct(num1, num2, num3, num4[i, :], den1, den2, den3, den4[i, :], Z,
                                transfer_function_type)
            
    return H
    

# Funtion to compute the output PSD by multiplying the squared modulus of the transfert function with the
# input PSD.
# This term appears as the integrand in Equations (8), (10), and (15) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def func_out(function, function_input):
    
    funtion_output = (np.abs(function)) ** 2 * (function_input)   
    
    return funtion_output
    

# Function to compute the numerical integral of the output PSD over the interval of 
# the considered frequencies using Simpson's function.

def integrate_function(integrand_function, integr_interval): 
    
    result = integrate.simpson(integrand_function, integr_interval)
    
    return result
    

# Function that allows us to load the windshake PSD and their corresponding frequencies

def load_PSD_windshake(file_path_wind): 
    
    with fits.open(file_path_wind) as hdul: 
        
        try: 
            
            data = hdul[0].data                                                # pylint: disable=E1101 
            frequencies = data[0, :]                                           # first row: frequencies 
            PSD_windshake = data[1:, :]                                        # second and third rows: PSD tip and tilt

            return frequencies, PSD_windshake

        except Exception as exc: 
        
            
            print(exc)
            return None, None
    

# Function to resize the vibration PSD to match the size of the atmospheric turbulence PSD 
# by filling with zeros, if needed.

def resize_psd_like(PSD_atmo_turb, PSD_vibration):
    
    PSD_vib_1= np.zeros_like(PSD_atmo_turb)  
    m = PSD_vibration.shape[0]  
    PSD_vib_1[:m, :] = PSD_vibration
    
    return PSD_vib_1


# Computes the output PSD by applying the corresponding transfer function to the input PSD 
# (using the function "func_out"). Than, the function integrates each modal output PSD over 
# the given frequency range, as described in Equation (8), (10) and (15) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_output_PSD_and_integrate(actuators_number, transf_funct, PSD_input, omega_temp_freq_interval):
    
    variance_ = 0 
    
    PSD_out = np.zeros_like(PSD_input)
    
    for i in range (actuators_number):                                  
         
        PSD_out[i, :] = func_out(transf_funct[i, :], PSD_input[i, :])
        integral = integrate_function(PSD_out[i, :], omega_temp_freq_interval)
        variance_ += integral
         
    return variance_, PSD_out


# Computes the temporal variance by resizing the vibration PSD, summing atmospheric turbulence PSD 
# and vibration PSD, applying the transfer function, and integrating over the frequency interval.
# See Equation (8) (in "Semianalytical error budget for adaptive optics systems with pyramid wavefront sensors", 
# Agapito and Pinna, 2019)

def temporal_variance (PSD_atmo_turb, PSD_vibration, transf_funct, actuators_number, omega_temp_freq_interval): 
    
    PSD_vib = resize_psd_like(PSD_atmo_turb, PSD_vibration)
    
    PSD_input = PSD_atmo_turb + PSD_vib

    variance_temp, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, PSD_input, omega_temp_freq_interval)
   
    return variance_temp, PSD_output, PSD_input 


# Function to extract the noise propagation coefficients from the reconstruction matrix.

def extract_propagation_coefficients(file_path_matrix_R): 
    
    with fits.open(file_path_matrix_R) as hdul: 
        
        try: 

            if len(hdul) > 1:
                h = hdul[1]
            else:
                h = hdul[0]
            R = np.array(h.data.copy())                # pylint: disable=E1101
            print("Shape of R:", R.shape)
            if R.ndim == 2:    
                return np.diag(R @ R.T)
            else:
                # If the data is already a 1D array, we assume it's the diagonal of R @ R.T
                # and return it directly
                return R

        except Exception as exc:
            
            print(exc) 
            return None


# =============================================================================
# OPTICAL GAIN MODULE
# =============================================================================

# Function to build the optical gain grid from FITS files.
# It reads the last-mode optical gain values for two modulation radii and arranges them 
# into a 2D grid indexed by modulation radius and seeing.

def _load_andes_gain_grid(file_mod0, file_mod4):
    """
    Loads and stacks the ANDES optical gain data from two separate FITS files.
    """
    with fits.open(file_mod0) as hdul:
        gain_mod0 = hdul[0].data[:, -1]
        
    with fits.open(file_mod4) as hdul:
        gain_mod4 = hdul[0].data[:, -1]
        
    return np.vstack([gain_mod0, gain_mod4])


# Function to compute the optical gain for a given modulation radius and seeing.
# Uses an optical gain grid from ANDES_og_mod0.fits and ANDES_og_mod4.fits and performs
# a 2D interpolation to estimate the gain for the given modulation radius and seeing

def compute_andes_optical_gain(file_mod0, file_mod4,
                                seeing, modulation_radius):
    """
    Computes the optical gain for the ANDES system using 2D interpolation.
    Axes: modulation radius, seeing.
    """
    gain_grid = _load_andes_gain_grid(file_mod0, file_mod4)
    
    # Define the grid axes for ANDES
    modal_radius_vals = np.array([0.0, 4.0])
    seeing_vals = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    
    # 2D Interpolation
    interp_optical_gain = RegularGridInterpolator(
        (modal_radius_vals, seeing_vals), 
        gain_grid, 
        bounds_error=False, 
        fill_value=None
    )
    
    interpolated_gain = float(interp_optical_gain((modulation_radius, seeing)))
    return interpolated_gain


# SOUL optical gain data is stored in a single FITS file as a 3D data cube,
# with axes for modulated modes, binning, and magnitudes.

def _load_soul_gain_cube(filepath):
    """
    Loads the SOUL optical gain data cube and its axes from a single FITS file.
    """
    with fits.open(filepath) as hdul:
        magnitudes = hdul[1].data     # Shape: (11,)
        binning = hdul[2].data        # Shape: (4,)
        mod_modes = hdul[3].data      # Shape: (6,)
        gain_cube = hdul[4].data      # Shape: (6, 4, 11)
        
    return mod_modes, binning, magnitudes, gain_cube


# SOUL optical gain is computed by performing a 3D interpolation over the modulated modes,
# binning, and magnitudes.

def compute_soul_optical_gain(filepath, target_mod_modes, target_binning, target_magnitude):
    """
    Computes the optical gain for the SOUL system using 3D interpolation.
    Axes order must match the data cube shape: (mod_modes, binning, magnitudes).
    """
    mod_modes, binning, magnitudes, gain_cube = _load_soul_gain_cube(filepath)
    
    # 3D Interpolation
    interp_optical_gain = RegularGridInterpolator(
        (mod_modes, binning, magnitudes), 
        gain_cube, 
        bounds_error=False, 
        fill_value=None
    )
    
    interpolated_gain = float(interp_optical_gain((target_mod_modes, target_binning, target_magnitude)))
    return interpolated_gain


# Function to perform a 2D interpolation of the optical gain over modulation 
# radius and seeing using RegularGridInterpolator

def double_interpolation_optical_gain(modal_radius_val, seeing_val, optical_gain_grid, modulation_radius, seeing):
   
    interp_optical_gain = RegularGridInterpolator((modal_radius_val, seeing_val), optical_gain_grid, bounds_error=False, fill_value=None)  
     
    last_opt_gain = float(interp_optical_gain((modulation_radius, seeing)))
    
    return last_opt_gain


# Function to read and return the sigma slopes data from the FITS file.

def read_sigma_slopes():
    
    with fits.open("src/file_fits/ANDES/slopes_rms_time_avg_all.fits") as hdul:
        data = hdul[0].data                                              # pylint: disable=E1101   
        
        return data


# Function to perform 2D interpolation of the sigma slopes data for a given modulation
# radius and seeing.

def double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
                                     modulation_radius, seeing):
    
    interp_sigma = RegularGridInterpolator((modal_radius_vals, seeing_vals), data_slopes[1,:,:], 
                                           bounds_error=False, fill_value=None) 
    sigma_slope_aliasing = float(interp_sigma((modulation_radius, seeing)))       
    
    return sigma_slope_aliasing


# Function to compute omega_0 (w_0)

def omega_0(telescope_diameter, windspeed, maximum_radial_order_corrected):
    
    omega0 = 2 * np.pi * (maximum_radial_order_corrected + 1) * windspeed/telescope_diameter
    
    return omega0


# Function to compute the preliminary aliasing coefficient k' based on sigma slope, 
# temporal frequency interval and system parameters. 

def compute_k_prime(omega_temp_freq_interval, alpha, sigma_slope_alias, c, telescope_diameter, 
                    windspeed, maximum_radial_order_corrected):
    
    w_0 = omega_0(telescope_diameter, windspeed, maximum_radial_order_corrected)
    
    omega_min = np.min(omega_temp_freq_interval) 
    omega_max = np.max(omega_temp_freq_interval) 
    integral_result = w_0 ** alpha * (w_0 - omega_min) + (omega_max ** (alpha + 1)
                                                                  - w_0 ** (alpha + 1)) / (alpha + 1)
    k_prime = (sigma_slope_alias ** 2) / ((c ** 2) * integral_result)    
    
    return k_prime
    
    
# Function to compute the aliasing modal coefficients k based on a given modulation
# radius and seeing.
# It uses sigma slope data from FITS files and applies 2D interpolation over 
# modulation radius and seeing.

def k_coeff_aliasing(modulation_radius, seeing, c, alpha, telescope_diameter, 
                     omega_temp_freq_interval, file_path_matrix_R, windspeed,
                     maximum_radial_order_corrected):
    
    data_slopes = read_sigma_slopes()  
    
    seeing_vals = data_slopes[0,0,:]                                           
    modal_radius_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0]) 
    
    sigma_slope_alias = double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
                                                         modulation_radius, seeing)
    
    k_pr = compute_k_prime(omega_temp_freq_interval, alpha, sigma_slope_alias, c, telescope_diameter, 
                           windspeed, maximum_radial_order_corrected)
    
    p_coefficient = extract_propagation_coefficients(file_path_matrix_R)
    
    if p_coefficient is None:   
                                              
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.")
   
    k_coeff = p_coefficient * k_pr
    
    return k_coeff


# Function to compute the aliasing PSD matrix from the modal coefficients k and the optical gain.

def aliasing_psd_from_coeffs(actuators_number, omega_temp_freq_interval, c, k, 
                             alpha, telescope_diameter, windspeed, maximum_radial_order_corrected):
    
    w_0 = omega_0(telescope_diameter, windspeed, maximum_radial_order_corrected)
    
    PSDaliasing = np.zeros((actuators_number, len(omega_temp_freq_interval)))
    
    for j in range(len(omega_temp_freq_interval)): 
        
        if omega_temp_freq_interval[j] <= w_0:
            
            for i in range (actuators_number): 
                
                PSDaliasing[i, j] = k[i] * (w_0 ** alpha)  
            
        else:  
            
            for i in range (actuators_number):   
            
                PSDaliasing[i, j] = k[i] * (omega_temp_freq_interval[j] ** alpha)
    
    return PSDaliasing  
    

# Function to compute the optical gain c, the modal aliasing coefficients k 
# and, then, to build the corresponding aliasing PSD matrix.

def PSD_aliasing (actuators_number, omega_temp_freq_interval, alpha,  
                  telescope_diameter, seeing, modulation_radius, windspeed,
                  maximum_radial_order_corrected, file_path_matrix_R, c_optg):
    
    k = k_coeff_aliasing(modulation_radius, seeing, c_optg, alpha, telescope_diameter,
                         omega_temp_freq_interval, file_path_matrix_R, windspeed,
                         maximum_radial_order_corrected)
    PSD_alias = aliasing_psd_from_coeffs(actuators_number, omega_temp_freq_interval, 
                                         c_optg, k, alpha, telescope_diameter, 
                                         windspeed, maximum_radial_order_corrected)

    return PSD_alias  
    

# Computes the aliasing variance by applying the transfer function to the aliasing PSD 
# and integrating the output over the specified frequency interval.
# See Equation (15) (in "Semianalytical error budget for adaptive optics systems
# with pyramid wavefront sensors", Agapito and Pinna, 2019).  

def aliasing_variance (transf_funct, actuators_number, omega_temp_freq_interval, 
                       alpha, telescope_diameter, seeing, modulation_radius, windspeed, 
                       maximum_radial_order_corrected, file_path_matrix_R, c_optg):
    
    PSD_input = PSD_aliasing(actuators_number, omega_temp_freq_interval, alpha, telescope_diameter,
                             seeing, modulation_radius, windspeed, maximum_radial_order_corrected,
                             file_path_matrix_R, c_optg)
    
    variance_alias, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct,
                                                                  PSD_input, omega_temp_freq_interval)
    
    return variance_alias, PSD_output, PSD_input 
    

# Function to compute the photon flux per frame per pixel, accounting for frame rate, 
# sub-aperture geometry, and source magnitude.

def flux_for_frame_for_pixel(photon_flux, telescope_diameter, frame_rate, magnitudo,
                             n_subaperture, collecting_area):
    
    flux_per_frame = photon_flux / frame_rate
    
    sub_aperture_radius = (telescope_diameter / n_subaperture)/2
    
    sub_aperture_area = np.pi * (sub_aperture_radius) ** 2

    flux_per_frame_per_sub_aperture = flux_per_frame * (sub_aperture_area / collecting_area)
    
    flux_per_frame_per_sub_aperture_magnitudo = flux_per_frame_per_sub_aperture * 10 ** (- magnitudo / 2.5)
    
    flux_per_frame_per_sub_aperture_magnitudo_per_pixel = flux_per_frame_per_sub_aperture_magnitudo / 4

    return flux_per_frame_per_sub_aperture_magnitudo_per_pixel


# Computes the pixel variance, as described in Equation (14) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019). 
# Than computes the slope noise variance, using the pixel variance and weighting 
# pixel noise by their positions, as described in Equation (12) in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_slope_noise_variance(F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
                                 photon_flux, telescope_diameter,frame_rate, magnitudo, 
                                 n_subaperture, collecting_area):
    
    n_phot_pix = flux_for_frame_for_pixel(photon_flux, telescope_diameter, frame_rate, magnitudo,
                                          n_subaperture, collecting_area)
    
    pixel_pos = np.array(pixel_pos)                                            
    pixel_variance = F_excess ** 2 * (n_phot_pix + sky_bkg + dark_curr) + read_out_noise
    pix_intensity = n_phot_pix
    slope_variance = np.sum((pixel_pos ** 2) * pixel_variance) / (4 * (pix_intensity)) ** 2   
   
    return slope_variance
 

# Function which returns the noise PSD (PSD_w), assuming that the noise w has a flat temporal
# PSD over the entire frequency range, as stated in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019).

def compute_noise_PSD(p_coefficient, omega_temp_freq_interval, actuators_number, sigma2_w):
    
    PSD_w = np.zeros((len(p_coefficient), len(omega_temp_freq_interval)))   
  
    omega_interval_min = np.min(omega_temp_freq_interval)                 
    omega_interval_max = np.max(omega_temp_freq_interval)
   
    for i in range (actuators_number):  

        PSD_w[i, :] = sigma2_w[i] /(omega_interval_max -  omega_interval_min) 

    return PSD_w 


# Calculates the slope noise variance, extracts the propagation coefficients from the FITS
# file, and computes the measurement noise variance for each mode. 
# Then, it calculates the corresponding measurement noise PSD, to obtain the total measurement 
# variance, as described in Equation (10) (in "Semianalytical error budget 
# for adaptive optics systems with pyramid wavefront sensors", Agapito and Pinna (2019). 

def measure_variance (F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
                      photon_flux, telescope_diameter,frame_rate, magnitudo, n_subaperture, 
                      collecting_area, file_path_matrix_R, omega_temp_freq_interval, 
                      transf_funct, actuators_number):
    
    variance_meas = 0
    
    slope_noise_variance = compute_slope_noise_variance(F_excess, pixel_pos, sky_bkg, dark_curr, 
                                                        read_out_noise, photon_flux, telescope_diameter,
                                                        frame_rate, magnitudo, n_subaperture, collecting_area)
   
    p_coefficient = extract_propagation_coefficients(file_path_matrix_R)
    
    if p_coefficient is None:                                                
        
        raise RuntimeError("Propagation coefficients not loaded") 
   
    print("Propagation coefficients loaded successfully.")
    
    sigma2_w = p_coefficient * slope_noise_variance
    
    PSD_input =  compute_noise_PSD (p_coefficient, omega_temp_freq_interval, actuators_number, sigma2_w)
    
    variance_meas, PSD_output = compute_output_PSD_and_integrate(actuators_number, transf_funct, 
                                                                 PSD_input, omega_temp_freq_interval)
    
    return variance_meas, PSD_output, PSD_input 
    

# Function to compute the transfer function H; it internally computes n4 and d4 
# (numerator and denominator of function C) and then builds H using these polynomials.

def build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                            den2, den3, transfer_function_type):   
    
    num4, den4, Z = compute_n4_d4(gain, omega_temp_freq_interval, t_0, actuators_number)

    transfer_function = compute_H (actuators_number, omega_temp_freq_interval, num1, num2, num3, num4, den1, 
                                den2, den3, den4, Z, transfer_function_type)
       
    return transfer_function


# Computes the variance of a given type ('fitting', 'temp', 'alias', or 'meas') for the system.

def variance(omega_temp_freq_interval, t_0, gain, num1, num2, num3, den1, den2, den3, 
             variance_type, actuators_number, telescope_diameter, Fried_parameter, F_excess, 
             sky_bkg, dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
             n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, seeing, 
             modulation_radius, windspeed, maximum_radial_order_corrected, transfer_function_type,
             file_optg, PSD_tur=None, PSD_vib=None, file_path_matrix_R=None, system="ANDES"):
    
   
    valid_types = ("fitting", "temp", "alias", "meas")
    
    # optical gains are needed for both aliasing and measurement variance,
    # so we compute them at the beginning
    if system == "ANDES":
        c_optg = compute_andes_optical_gain(file_optg[0], file_optg[1], seeing, modulation_radius)
    # TODO not supported yet
    #elif system == "SOUL":
    #    gain = compute_soul_optical_gain(file_optg, mod_modes, binning, magnitude)
    
    if variance_type not in valid_types: 
        raise ValueError("Variance_type must be one of: 'fitting', 'temp', 'alias', or 'meas'")
       
       
    if variance_type == "fitting" and PSD_tur is None and PSD_vib is None:
            
        sigma2_fit = fitting_variance(fitting_coeff, actuators_number, telescope_diameter, Fried_parameter) 
       
        print("Fitting:", sigma2_fit)   
        return sigma2_fit
   
    
    if variance_type == "temp" and PSD_tur is not None and PSD_vib is not None:
    
        H = build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                                    den2, den3, transfer_function_type)
    
        sigma2_temp, PSD_out, PSD_in = temporal_variance (PSD_tur, PSD_vib, H, actuators_number, omega_temp_freq_interval)
         
        print("Temporal:", sigma2_temp)       
        return sigma2_temp, PSD_out, PSD_in, H
   
    
    if variance_type == "alias" and PSD_tur is None and PSD_vib is None and file_path_matrix_R is not None:
        
        H = build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                                    den2, den3,  transfer_function_type)
 
        sigma2_alias, PSD_out, PSD_in = aliasing_variance(H, actuators_number, omega_temp_freq_interval, 
                               alpha, telescope_diameter, seeing, modulation_radius, windspeed, 
                               maximum_radial_order_corrected, file_path_matrix_R, c_optg)
       
        print("Aliasing:", sigma2_alias)       
        return sigma2_alias, PSD_out, PSD_in, H
   
    
   
    if variance_type == "meas" and PSD_tur is None and PSD_vib is None and file_path_matrix_R is not None:
       
        H = build_transfer_function(gain, omega_temp_freq_interval, t_0, actuators_number, num1, num2, num3, den1, 
                                    den2, den3, transfer_function_type)  
      
        sigma2_meas, PSD_out, PSD_in  = measure_variance (F_excess, pixel_pos, sky_bkg, dark_curr, read_out_noise,
                                                          photon_flux, telescope_diameter, frame_rate, magnitudo, n_subaperture, 
                                                          collecting_area, file_path_matrix_R, omega_temp_freq_interval, 
                                                          H, actuators_number)
      
      
        print("Measure:", sigma2_meas)       
        return sigma2_meas, PSD_out, PSD_in, H
      
      

# Function to interpolate a 1D vector to a new set of points, setting values outside the original range to 0.

def interpolate_vector(x_interpolation, x_original, vector_original):
    
    vector_interpolated = np.interp (x_interpolation, x_original, vector_original, left = 0, right = 0) 
    
    return vector_interpolated


# Function to interpolate and normalize PSD to a new frequency interval.

def interpolate_and_normalize_psd(freqs_interpolation, freqs_original, PSD_original, actuators_number):
    
    PSD_interpolated = np.zeros_like(PSD_original)
    PSD_interpolated_normalized = np.zeros_like(PSD_original)
    sigma2 = np.zeros(actuators_number)
    sigma2_interp = np.zeros(actuators_number)
    
    Omega_freqs_interpolation = 2 * np.pi * freqs_interpolation  
    Omega_freqs_original = 2 * np.pi * freqs_original
    
    for i in range (actuators_number):
        
        PSD_interpolated[i, :]=interpolate_vector(freqs_interpolation, freqs_original, PSD_original[i, :])

        sigma2 [i] = integrate.simpson(PSD_original[i, :], Omega_freqs_original)
        sigma2_interp[i] = integrate.simpson(PSD_interpolated[i, :], Omega_freqs_interpolation)
   
        PSD_interpolated_normalized [i, :]= (PSD_interpolated[i, :]) * (sigma2[i])/(sigma2_interp[i])

    return PSD_interpolated_normalized
         

# Function to compute the total variance by summing fitting variance, temporal variance, and meas variance contributions.

def total_variance(fit_err, temp_err, alias_err, meas_err):
    var_tot = np.real(fit_err) + np.real(temp_err) + np.real(meas_err) + np.real(alias_err)
    print ("Total variance:", var_tot)
    return var_tot 


# # Function to compute the total residual variance for a set of gain values,
# # by combining fitting, temporal, aliasing and measurement error contributions 
  
# def variance_total_for_test(number_of_actuators, gain_value_, omega_temp_freq_interval, t_0, num1, num2, 
#                             num3, den1, den2, den3, telescope_diameter, Fried_parameter, F_excess, 
#                             sky_bkg, dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
#                             n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha,
#                             seeing, modulation_radius, windspeed, maximum_radial_order_corrected,
#                             file_path_reconstruction_matrix, PSD_turbolence, PSD_vibration_wind):
    
                                                            
#     tot_variance = np.zeros_like(gain_value_, dtype=float)
    
#     for i in range(len(gain_value_)): 
        
#         g = gain_value_[i]                                                     
#         gain_val = np.array([g])                                               

#         variance_fit = variance(omega_temp_freq_interval, t_0, gain_val, num1, num2, 
#                                 num3, den1, den2, den3, "fitting", number_of_actuators, 
#                                 telescope_diameter, Fried_parameter, F_excess, sky_bkg, 
#                                 dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
#                                 n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, 
#                                 seeing, modulation_radius, windspeed, maximum_radial_order_corrected, 
#                                 None, PSD_tur=None, PSD_vib=None, file_path_matrix_R=None)
    
#         variance_temporal,_ , _, _ = variance(omega_temp_freq_interval, t_0, gain_val, 
#                                               num1, num2, num3, den1, den2, den3, "temp", 
#                                               number_of_actuators, telescope_diameter, Fried_parameter, 
#                                               F_excess, sky_bkg, dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
#                                               n_subaperture, collecting_area, pixel_pos, 
#                                               fitting_coeff, alpha, seeing, modulation_radius, 
#                                               windspeed, maximum_radial_order_corrected,'H_r', PSD_tur=PSD_turbolence, 
#                                               PSD_vib=PSD_vibration_wind, file_path_matrix_R=None)
    
#         variance_aliasing, _, _, _ = variance(omega_temp_freq_interval, t_0, gain_val, num1, num2, 
#                                               num3, den1, den2, den3, "alias", number_of_actuators, 
#                                               telescope_diameter, Fried_parameter, F_excess, sky_bkg,
#                                               dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
#                                               n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, seeing, 
#                                               modulation_radius, windspeed, maximum_radial_order_corrected,
#                                               'H_n', PSD_tur=None, PSD_vib=None, file_path_matrix_R=file_path_reconstruction_matrix)
    
#         variance_measurement, _, _, _ = variance(omega_temp_freq_interval, t_0, gain_val, num1, num2, 
#                                                  num3, den1, den2, den3, "meas", number_of_actuators, 
#                                                  telescope_diameter, Fried_parameter, F_excess, sky_bkg, 
#                                                  dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
#                                                  n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, seeing, 
#                                                  modulation_radius, windspeed, maximum_radial_order_corrected,
#                                                  'H_n', PSD_tur=None, PSD_vib=None, file_path_matrix_R=file_path_reconstruction_matrix)
        
#         tot_variance[i] = total_variance(np.real(variance_fit), np.real(variance_temporal), 
#                                          np.real(variance_measurement), np.real(variance_aliasing))            
    
#     return tot_variance


# # Function to plot the total residual variance of the system as a function 
# # of the gain, considering only the first mode.

# def test(gain_min, gain_max, omega_temp_freq_interval, t_0, num1, num2, 
#          num3, den1, den2, den3, telescope_diameter, Fried_parameter, F_excess, 
#          sky_bkg, dark_curr, read_out_noise, photon_flux, frame_rate, magnitudo, 
#          n_subaperture, collecting_area, pixel_pos, fitting_coeff, alpha, seeing, 
#          modulation_radius, windspeed, maximum_radial_order_corrected,
#          file_path_reconstruction_matrix, PSD_turbolence, PSD_vibration_wind): 
   
#     do_test = input("Do you want to do the test? (y/n):") 

#     if do_test == "y":
       
#         print ('TEST') 
     
#         actuators_number = 1                                                  
     
#         gain_value = np.arange (gain_min, gain_max, 0.1)
#         variance_total = variance_total_for_test(actuators_number, gain_value, omega_temp_freq_interval,
#                                                  t_0, num1, num2, num3, den1, den2, den3, telescope_diameter, 
#                                                  Fried_parameter, F_excess, sky_bkg, dark_curr,read_out_noise, 
#                                                  photon_flux, frame_rate, magnitudo, n_subaperture, collecting_area, 
#                                                  pixel_pos, fitting_coeff, alpha, seeing, 
#                                                  modulation_radius, windspeed, maximum_radial_order_corrected,
#                                                  file_path_reconstruction_matrix, 
#                                                  PSD_turbolence, PSD_vibration_wind)
        
#         plt.plot(gain_value, variance_total, marker='o')    
#         plt.xlabel('Gain')
#         plt.ylabel('Total variance')
#         plt.title('Total variance as a function of the gain')
#         plt.grid()
#         plt.show()
       
        
# # Defines a function that allows, when needed, to plot PSD_in, PSD_out, and the transfer 
# # function for the variances (temp, alias, meas)

# def plot(f, H_r_t, H_n_m, H_n_a, PSD_in_t, PSD_out_t, PSD_in_m, PSD_out_m, PSD_in_a, PSD_out_a):
    
#     do_plot = input("Do you want to plot PSD_in, PSD_out, and the transfer function? (y/n):")

#     if do_plot == "y":
      
#         PSD_in = [PSD_in_t, PSD_in_m, PSD_in_a]                  
#         PSD_out = [PSD_out_t, PSD_out_m, PSD_out_a]              
#         H = [H_r_t, H_n_m, H_n_a]
       
#         labels_PSD = ["temp", "meas","alias"]
#         labels_H = ["r", "n", "n"]
       
#         for i in range(len(PSD_in)):                                                           
          
#             plt.loglog(f, PSD_in[i][0, :], label=f"PSD_in_{labels_PSD[i]} (mode 0)")       
#             plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels_PSD[i]} (mode 0)")
#             plt.xlabel("Frequency [rad/s]")
#             plt.ylabel("PSD")
#             plt.title(f"PSD {labels_PSD[i]} (modo 0)")
#             plt.legend()
#             plt.grid()
#             plt.show()
           
#             plt.loglog(f, np.abs(H[i][0, :])**2)     
#             plt.xlabel("Frequency [rad/s]")
#             plt.ylabel(f"|H_{labels_H[i]}|^2")
#             plt.title(f"Transfert function H_{labels_H[i]} (mode 0)")
#             plt.grid()
#             plt.show()
       
        
# # Function to plot the three output PSDs (temporal, measurement and aliasing)
# # for mode 0 on the same graph.
       
# def plot_all_PSD(f, PSD_out_t, PSD_out_m, PSD_out_a):
    
#     do_plot = input("Do you want to plot all the PSD_out (mode 0)? (y/n):")

#     if do_plot == "y":

#         PSD_out = [PSD_out_t, PSD_out_m, PSD_out_a]

#         labels = ["temp", "meas", "alias"]

#         for i in range(len(PSD_out)):

#             plt.loglog(f, PSD_out[i][0, :], label=f"PSD_out_{labels[i]} (mode 0)")

#         plt.xlabel("Frequency [rad/s]")
#         plt.ylabel("PSD")
#         plt.title(f"PSD {labels[i]} – mode 0")
#         plt.legend()
#         plt.grid()

#         plt.show()


# # Function to verify the consistency of the aliasing variance calculation.
# # It computes the analytical aliasing variance (summing the selected modes)
# # and compares it with the variance obtained by integrating the corresponding 
# # aliasing PSD. 
# # The function also print the variance of the first mode alone in both cases.
    
# def check(file_path_matrix_R, telescope_diameter, seeing, modulation_radius,
#           actuators_number, alpha, omega_temp_freq_interval, windspeed, 
#           maximum_radial_order_corrected):
    
#     c = c_optical_gain (seeing, modulation_radius)
    
#     p_coefficient = extract_propagation_coefficients(file_path_matrix_R)
    
#     if p_coefficient is None:   
                                              
#         raise RuntimeError("Propagation coefficients not loaded") 
   
#     print("Propagation coefficients loaded successfully.")
    
#     data_slopes = read_sigma_slopes()  
#     seeing_vals = data_slopes[0,0,:]                                           
  
#     modal_radius_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0]) 
    
    
    
#     sigma_slope_alias = double_interpolation_sigma_slope(modal_radius_vals, seeing_vals, data_slopes, 
#                                                          modulation_radius, seeing)
  
#     sigma_alias_2_two_modes = 0.0
    
#     for i in range (actuators_number):
        
#         sigma_alias_2 = p_coefficient[i] * (sigma_slope_alias ** 2) / c ** 2
        
#         sigma_alias_2_two_modes += sigma_alias_2
        
#     print("ALIASING VARIANCE:", sigma_alias_2_two_modes)
    
#     PSD_al = PSD_aliasing (actuators_number, omega_temp_freq_interval, alpha,  
#                            telescope_diameter, seeing, modulation_radius, windspeed, maximum_radial_order_corrected,
#                            file_path_matrix_R)
    
    
#     integral_per_mode = integrate.simpson(PSD_al, omega_temp_freq_interval)
#     sigma_alias_2_PSD_total = np.sum(integral_per_mode)
    
#     print("ALIASING VARIANCE FROM PSD:", sigma_alias_2_PSD_total)
    
#     sigma_alias_2_one_mode = p_coefficient * (sigma_slope_alias ** 2) / c ** 2
    
#     print("ALIASING VARIANCE ONE MODE:", sigma_alias_2_one_mode[0])
    
#     print("ALIASING VARIANCE FROM PSD ONE MODE:", integral_per_mode[0])
    
    
# def plot_PSD_alias_mode_0(actuators_number, omega_temp_freq_interval, alpha, telescope_diameter,
#                           seeing, modulation_radius, windspeed, maximum_radial_order_corrected,
#                           file_path_matrix_R):
    
#     with fits.open("modal_psd_aliasing.fits") as hdul:
#         data = hdul[0].data 
        
#         freq_hz = data[:, 0]
#         mode_0 = data[:, 1]
        
#         freq_rad_s = 2 * np.pi * freq_hz
#         c = c_optical_gain (seeing, modulation_radius)
#         PSD_aliasing_mode0_given = mode_0 / (c ** 2 * 2 * np.pi)             
        
#     PSD_alising_mine = PSD_aliasing (actuators_number, omega_temp_freq_interval, alpha,  
#                                            telescope_diameter, seeing, modulation_radius, windspeed, 
#                                            maximum_radial_order_corrected,file_path_matrix_R)
    
#     PSD_alising_mine_mode0 = PSD_alising_mine[0,:]
    
#     plt.loglog(omega_temp_freq_interval, PSD_alising_mine_mode0, label="PSD alias mine mode 0")
#     plt.loglog(freq_rad_s, PSD_aliasing_mode0_given, label="PSD alias data mode 0")
    

#     plt.xlabel("Frequency [rad/s]")
#     plt.ylabel("PSD")  
#     plt.legend()
#     plt.grid()
#     plt.show()

#     # N = len(freq_rad_s)   # = 501

#     # omega_cut = omega_temp_freq_interval[:N]
#     # PSD_mine_cut = PSD_alising_mine_mode0[:N]

#     # ratio = PSD_mine_cut / PSD_aliasing_mode0_given
#     # plt.figure()

#     # plt.semilogx(omega_cut, ratio)
#     # plt.xlabel("Frequency [rad/s]")
#     # plt.ylabel("Mine / Given")
#     # plt.grid()
#     # plt.title("PSD ratio")

#     # plt.show()
    
    
    
    
    
  

    
        
        
    
    

    

    
    
    
          
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
