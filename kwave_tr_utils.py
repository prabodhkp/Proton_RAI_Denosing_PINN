# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 10:18:59 2025

@author: TRUE Lab
"""

# kwave_tr_utils.py

import numpy as np
import scipy.io as sio
#from kwave import kWaveGrid, kWaveMedium, kSensor, kSource, kspaceFirstOrder3D
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from scipy.interpolate import interp1d
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from my_visualization import plot_mip


def setup_kwave_grid(Lx, Ly, Lz, dx, dy, dz, c0):
    """
    Setup the k-Wave grid based on the given domain parameters.
    
    Parameters:
    Lx, Ly, Lz : float
        Domain size in meters.
    dx, dy, dz : float
        Grid spacing in meters.
    c0 : float
        Sound speed in m/s.
    
    Returns:
    kgrid : kWaveGrid object
        The k-Wave grid object.
    Nx, Ny, Nz : int
        Number of grid points in each dimension.
    """
    Nx = int(np.round(Lx / dx))
    Ny = int(np.round(Ly / dy))
    Nz = int(np.round(Lz / dz))
    
    # Create k-Wave grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
    dt = 0.3 * min(dx, dy, dz) / c0  # Time step, following stability criterion
    kgrid.makeTime(c0, dt)  # Initialize grid with time information
    
    return kgrid, Nx, Ny, Nz


def create_medium(c0, rho0):
    """
    Create a k-Wave medium with given sound speed and density.
    
    Parameters:
    c0 : float
        Sound speed in m/s.
    rho0 : float
        Density in kg/m^3.
    
    Returns:
    medium : kWaveMedium object
        The k-Wave medium object with the given properties.
    """
    medium = kWaveMedium(sound_speed=c0, density=rho0)
    return medium


def load_and_setup_detectors(detector_file, Lx, Ly, Lz, dx, dy, dz, Nx, Ny, Nz, fix_z):
    """
    Load the detector coordinates from a .mat file and return a sensor mask.
    
    Parameters:
    detector_file : str
        Path to the file with detector coordinates.
    Lx, Ly, Lz : float
        Domain size in meters.
    dx, dy, dz : float
        Grid spacing in meters.
    Nx, Ny, Nz : int
        Grid dimensions.
    fix_z : float
        Fixed z-coordinate for detectors.
    
    Returns:
    sensor_mask : np.ndarray
        A boolean array representing detector positions on the grid.
    det_z_idx : np.ndarray
        Z-index positions of detectors on the grid.
    """
    mat = sio.loadmat(detector_file)
    det_cords = mat["det_cords"]
    
    # Fix z-coordinate as specified
    det_cords[:, 2] = fix_z  # Set fixed z-position for detectors
    
    # Convert coordinates to grid indices
    det_x_ind = np.round((det_cords[:, 0] + Lx / 2) / dx).astype(int)
    det_y_ind = np.round((det_cords[:, 1] + Ly / 2) / dy).astype(int)
    det_z_ind = np.round((det_cords[:, 2] + Lz / 2) / dz).astype(int)
    
    # Clip indices to be within bounds of the grid
    det_x_ind = np.clip(det_x_ind, 0, Nx-1)
    det_y_ind = np.clip(det_y_ind, 0, Ny-1)
    det_z_ind = np.clip(det_z_ind, 0, Nz-1)
    
    # Create a mask for the detectors
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    for i in range(len(det_x_ind)):
        sensor_mask[det_x_ind[i], det_y_ind[i], det_z_ind[i]] = True
    
    return sensor_mask, det_z_ind


def batch_time_reversal_reconstruction(results, kgrid, sensor_mask, medium, Fs_exp, det_z_idx, 
                                      Nx_predict, Ny_predict, Nz_predict, use_gpu, verbose):
    """
    Perform time reversal reconstruction for each frame in the results.
    
    Parameters:
    results : np.ndarray
        Sensor data (N_det, Nt, Nframes).
    kgrid : kWaveGrid object
        The k-Wave grid object.
    sensor_mask : np.ndarray
        A boolean array representing detector positions on the grid.
    medium : kWaveMedium object
        The k-Wave medium object.
    Fs_exp : float
        Sampling frequency in Hz.
    det_z_idx : np.ndarray
        Z-index positions of detectors on the grid.
    Nx_predict, Ny_predict, Nz_predict : int
        The output crop size for the reconstruction.
    use_gpu : bool
        Whether to use GPU acceleration.
    verbose : int
        Verbosity level (0=quiet, 1=normal, 2=detailed).
    
    Returns:
    PB_rec_exp : np.ndarray
        Reconstructed pressure field (Nframes, Nx_predict, Ny_predict, Nz_predict).
    BP_loc_shotwise : np.ndarray
        Brightest point locations (Nframes, 3).
    TR_max : np.ndarray
        Maximum pressure values before normalization (Nframes, 1).
    """
    Nframes = results.shape[2]
    BP_loc_shotwise = np.zeros((Nframes, 3), dtype=int)
    TR_max = np.zeros((Nframes, 1), dtype=int)
    PB_rec_exp = np.zeros((Nframes, Nx_predict, Ny_predict, Nz_predict), dtype=float)
    
    for frame_ind in range(Nframes):
        sensor_data = results[:, :, frame_ind]  # Shape: (N_det, Nt)
        
        # Reverse the sensor data for time reversal
        sensor_data_tr = sensor_data[:, ::-1].astype(np.float32)  # Reverse the time axis
        
        # Update the time parameters
        dt_data = 1.0 / Fs_exp
        t_end = (sensor_data.shape[1] - 1) * dt_data
        kgrid.setTime(sensor_data.shape[1], dt_data)
        
        # Set up the sensor for time reversal
        sensor = kSensor()
        sensor.mask = sensor_mask
        sensor.time_reversal_boundary_data = sensor_data.astype(np.float32) 
        
        # Empty source for time reversal
        source = kSource()
        source.p_mask =0*sensor_mask # np.zeros((Nx, Ny, Nz), dtype=bool)

        
        # Simulation options
        sim_opts = SimulationOptions()
        sim_opts.pml_inside = False
        sim_opts.pml_size = [10, 10, 10]
        sim_opts.data_cast = 'single'
        sim_opts.save_to_disk = True
        sim_opts.record_movie = False
        
        # Execution options
        exec_opts = SimulationExecutionOptions()
        exec_opts.is_gpu_simulation = use_gpu
        exec_opts.verbose_level = verbose
        
        # Perform the time reversal reconstruction
        recon_result = kspaceFirstOrder3D(kgrid, source, sensor, medium, sim_opts, exec_opts)
        
        # Extract the reconstructed pressure field and crop it
        pressure_field = recon_result['p_final']
        pressure_field = np.transpose(pressure_field, (2, 1, 0))  # Transpose to (Nx, Ny, Nz)
        pressure_field[pressure_field < 0] = 0  # Set negative values to zero
        pressure_field[:, :, 154:] = 0
        plot_mip(pressure_field, h=1e-3)
        
        # Store the cropped reconstructed pressure field
        PB_rec_exp[frame_ind] = pressure_field[:Nx_predict, :Ny_predict, :Nz_predict]
        
        # Find the brightest point in the reconstruction
        BP_loc = np.unravel_index(np.argmax(pressure_field), pressure_field.shape)
        BP_loc_shotwise[frame_ind] = BP_loc
        TR_max[frame_ind] = np.max(pressure_field)
    
    return PB_rec_exp, BP_loc_shotwise, TR_max
