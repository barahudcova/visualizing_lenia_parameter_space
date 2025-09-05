import numpy as np
import pickle
import os
from tqdm import tqdm

import cProfile, pstats, io
from pstats import SortKey
import time

import torch
from lenia_jax_fft import MultiLeniaJAX
import cv2
import pickle as pk
import numpy as np, os, random
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp



TIME_BOUND = 200 # in seconds

def classify(phases):
    if phases == {"order"}:
        return "order"
    elif phases == {"chaos"}: #or phases == {"chaos", "max"}:
        return "chaos"
    elif phases == {"order", "chaos", "max"}:
        return "max"
    elif phases == {"order", "max"}:
        return "max"
    elif phases == {"order", "chaos"}:
        return "trans"
    elif phases:
        return "no phase"
    else:
        return "TBA"
    
def check_mass(xmass_history, ymass_history, std, window_size, current_time):
    if len(xmass_history[:current_time]) < window_size:
        return False
    
    # Extract relevant window of history
    xmasses = xmass_history[current_time - window_size:current_time]
    ymasses = ymass_history[current_time - window_size:current_time]
    
    # Calculate means
    xmju = jnp.mean(xmasses)
    ymju = jnp.mean(ymasses)
    
    # Check for chaos
    chaos = True
    
    # Check x masses
    for x in xmasses:
        diff = jnp.abs(xmju - x)
        if diff > std:
            chaos = False
    
    # Check y masses
    for y in ymasses:
        diff = jnp.abs(ymju - y)
        if diff > std:
            chaos = False
    
    return chaos

def get_batch_phases(auto, mass_centers_x, mass_centers_y, total_masses, std, window_size):
    phases = [None] * auto.batch

    
    for b in range(auto.batch):
        # Check if the latest values match any previous values
        same_mass = jnp.any(total_masses[-1, b] == total_masses[:-1, b])
        same_cent_x = jnp.any(mass_centers_x[-1, b] == mass_centers_x[:-1, b])
        same_cent_y = jnp.any(mass_centers_y[-1, b] == mass_centers_y[:-1, b])
        
        if same_mass and same_cent_x and same_cent_y:
            phases[b] = "order"
            continue
        
        # Calculate stability metrics
        cmx_mean = jnp.mean(mass_centers_x[-window_size:, b], axis=0)
        cmy_mean = jnp.mean(mass_centers_y[-window_size:, b], axis=0)
        
        cmx_diffs = jnp.abs(cmx_mean - mass_centers_x[-window_size:, b])
        cmy_diffs = jnp.abs(cmy_mean - mass_centers_y[-window_size:, b])
        
        cmx_stable = jnp.all(jnp.max(cmx_diffs) < std)
        cmy_stable = jnp.all(jnp.max(cmy_diffs) < std)
        
        if cmx_stable and cmy_stable:
            phases[b] = "chaos"
        else:
            phases[b] = "max"
    return phases

# @timeout(TIME_BOUND)
def get_approx_trans_extended(auto, std, window_size):
    """Extended version with T_MAX = int(np.round(1000*np.log2(array_size)))"""
    config = auto.state
    array_size = config.shape[-1]
    T_MAX = int(np.round(1000*np.log2(array_size)))  # Extended time

    mass_centers_x = np.zeros((T_MAX, auto.batch))
    mass_centers_y = np.zeros((T_MAX, auto.batch))
    total_masses = np.zeros((T_MAX, auto.batch))

    for t in range(T_MAX):
        auto.step()
        cm, tm = auto.get_batch_mass_center(auto.state)
        mass_centers_x[t]=cm[0]
        mass_centers_y[t]=cm[1]
        total_masses[t]=tm

    phases = get_batch_phases(auto, mass_centers_x, mass_centers_y, total_masses, std, window_size)

    return phases

def get_approx_trans(auto, std, window_size):
    """Original version with T_MAX = int(np.round(100*np.log2(array_size)))"""
    config = auto.state
    array_size = config.shape[-1]
    T_MAX = int(np.round(100*np.log2(array_size)))

    mass_centers_x = np.zeros((T_MAX, auto.batch))
    mass_centers_y = np.zeros((T_MAX, auto.batch))
    total_masses = np.zeros((T_MAX, auto.batch))

    for t in range(T_MAX):
        auto.step()
        cm, tm = auto.get_batch_mass_center(auto.state)
        mass_centers_x[t]=cm[0]
        mass_centers_y[t]=cm[1]
        total_masses[t]=tm

    phases = get_batch_phases(auto, mass_centers_x, mass_centers_y, total_masses, std, window_size)

    return phases

def reprocess_max_phases(folder_name, array_size, polygon_size_range, samples, params, batch_size):
    """
    Reprocess existing data where individual samples are classified as 'max' with extended simulation time
    """
    g_mju = np.round(params["mu"].item(), 5)
    g_sig = np.round(params["sigma"].item(), 5)
    
    path = f"{folder_name}/data/{g_mju}_{g_sig}.pickle"
    
    print(f"Reprocessing data for mu={g_mju}, sigma={g_sig}")
    
    # Load existing data
    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
    except FileNotFoundError:
        print(f"No existing data found at {path}")
        return
    
    if str(array_size) not in data:
        print(f"No data for array_size {array_size}")
        return
    
    std = 3
    window_size = 200
    
    # Check each polygon size
    for polygon_size in polygon_size_range:
        if str(polygon_size) not in data[str(array_size)]:
            #print(f"No data for polygon_size {polygon_size}, skipping")
            continue
            
        polygon_data = data[str(array_size)][str(polygon_size)]
        
        # Check if this polygon size has any samples classified as "max"
        individual_phases = polygon_data["phase"]
        has_max_sample = "max" in individual_phases
        
        if not has_max_sample:
            #print(f"Polygon size {polygon_size}: no 'max' samples found, skipping")
            continue
            
        print(f"Reprocessing polygon size {polygon_size} (contains {individual_phases.count('max')} 'max' samples)...")
        
        # Get the existing seeds and samples to reproduce exact same conditions
        existing_seeds = polygon_data["seed"]
        existing_samples = polygon_data["sample"]
        existing_phases = polygon_data["phase"]
        
        if len(existing_seeds) != len(existing_samples) or len(existing_seeds) != len(existing_phases):
            print(f"Data inconsistency for polygon_size {polygon_size}, skipping")
            continue
        
        # Create automaton
        auto = MultiLeniaJAX((array_size, array_size), batch=batch_size, num_channels=1, dt=0.1, params=params)
        
        # Clear old data and prepare for new results
        new_phases = []
        new_seeds = []
        new_samples = []
        
        # Process in batches using the exact same seeds
        num_existing = len(existing_seeds)
        for batch_start in range(0, num_existing, batch_size):
            batch_end = min(batch_start + batch_size, num_existing)
            current_batch_size = batch_end - batch_start
            
            # Get seeds for this batch
            batch_seeds = existing_seeds[batch_start:batch_end]
            batch_samples = existing_samples[batch_start:batch_end]
            
            # If we have fewer seeds than batch_size, we need to adjust
            if current_batch_size < batch_size:
                # Create a new automaton with the correct batch size
                auto_batch = MultiLeniaJAX((array_size, array_size), batch=current_batch_size, num_channels=1, dt=0.1, params=params)
            else:
                auto_batch = auto
            
            # Set the exact same initial conditions using the stored seeds
            auto_batch.seeds = batch_seeds
            auto_batch.set_init_voronoi_batch(polygon_size, init_polygon_index=0, seeds=batch_seeds)
            
            # Run extended simulation
            new_batch_phases = get_approx_trans_extended(auto_batch, std, window_size)
            
            # Store results
            new_phases.extend(new_batch_phases)
            new_seeds.extend(batch_seeds)
            new_samples.extend(batch_samples)
            
            print(f"  Processed batch {batch_start//batch_size + 1}/{(num_existing + batch_size - 1)//batch_size}")
        
        # Update data with new results
        data[str(array_size)][str(polygon_size)]["phase"] = new_phases
        data[str(array_size)][str(polygon_size)]["seed"] = new_seeds
        data[str(array_size)][str(polygon_size)]["sample"] = new_samples

        
        # Save updated data
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        old_classification = classify(set(existing_phases))
        new_classification = classify(set(new_phases))
        print(f"  Updated classification: {old_classification} -> {new_classification}")
        print(f"  Phase distribution: {dict(zip(*np.unique(new_phases, return_counts=True)))}")
        print()

def scan_and_reprocess_all_data(base_folder, array_size, polygon_size_range, samples, batch_size):
    """
    Scan all existing data files and reprocess those with 'max' phases
    """
    data_folder = f"{base_folder}/data"
    g_mju_range = [np.round(i, 5) for i in np.arange(0.1, 0.5, 0.005)]
    g_sig_range = [np.round(i, 5) for i in np.arange(0.001, 0.1, 0.001)]
    
    if not os.path.exists(data_folder):
        print(f"Data folder {data_folder} does not exist")
        return
    
    # Get all pickle files in the data folder
    pickle_files = [f for f in os.listdir(data_folder) if f.endswith('.pickle')]
    
    print(f"Found {len(pickle_files)} data files to check")
    
    for pickle_file in tqdm(pickle_files):
        # Extract mu and sigma from filename
        try:
            base_name = pickle_file.replace('.pickle', '')
            mu_str, sigma_str = base_name.split('_')
            g_mju = np.round(float(mu_str), 5)
            g_sig = np.round(float(sigma_str), 5)
            if (g_mju not in g_mju_range) or (g_sig not in g_sig_range):
                print(f"Skipping file {pickle_file} with unsupported mu/sigma: {g_mju}, {g_sig}")
                continue
        except ValueError:
            #print(f"Could not parse filename {pickle_file}, skipping")
            continue
        
        # Create params for this mu/sigma combination
        
        
        params = {
            'k_size': base_params['k_size'], 
            'mu': jnp.array([[[g_mju]]]), 
            'sigma': jnp.array([[[g_sig]]]), 
            'beta': base_params['beta'], 
            'mu_k': base_params['mu_k'], 
            'sigma_k': base_params['sigma_k'], 
            'weights': base_params['weights'],
            'func_k': base_params['func_k'],
        }
        
        print(f"\nChecking file: {pickle_file} (mu={g_mju}, sigma={g_sig})")
        reprocess_max_phases(base_folder, array_size, polygon_size_range, samples, params, batch_size)

#============================== PARAMETERS ================================
W,H = 200,200 # Size of the automaton
array_size = W
dt = 0.1 # Time step size
num_channels= 1
device = 0    #index of device available

print(jax.devices())

jax.config.update("jax_default_device", jax.devices()[device])

samples = 64  # total number of data samples per polygon size
B = 64  # batch size

polygon_size_range = list(range(100))

# Base parameters (will be overridden for each specific mu/sigma combination)
beta = [1.0, 0.5]
k_mju = [0.5]
k_sig = [0.15]

base_params = {
    'k_size': 57, 
    'mu': jnp.array([[[0.5]]]), 
    'sigma': jnp.array([[[0.15]]]), 
    'beta': jnp.array([[[beta]]]), 
    'mu_k': jnp.array([[[k_mju]]]), 
    'sigma_k': jnp.array([[[k_sig]]]), 
    'weights': jnp.array([[[1.0]]]),
    'func_k': 'quad4',
} 


#======================================================================

if __name__ == "__main__":
    start = time.time()
    
    # Create a dummy automaton to get the folder structure
    auto = MultiLeniaJAX((W, H), batch=B, num_channels=1, dt=0.1, params=base_params, device=device)
    folder_name = auto.kernel_path
    
    print(f"Scanning and reprocessing data in folder: {folder_name}")
    
    # Scan all data and reprocess files with 'max' phases
    scan_and_reprocess_all_data(folder_name, array_size, polygon_size_range, samples, B)
    
    print(f"\nTotal processing time: {time.time()-start:.2f} seconds")