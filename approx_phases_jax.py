import numpy as np
import pickle
import os

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


def classify(phases):
    # order correspond to the stable phase
    # chaos corresponds to metastable phase
    # max corresponds to solitons

    if phases == {"order"}:
        return "order"
    elif phases == {"chaos"}:
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
    """
    Determine a phase (order ~ stable, chaos ~ metastable, max ~ soliton) for each configuration in the batch based on the stability of their mass centers and total masses.
    1. If the latest total mass and mass centers match any previous values, we hypothesize the system has enetered a loop and classify as "order" ~ stable phase.
    2. If the system did not enter a loop, but the mass centers are stable within a defined threshold (std) over a specified window size, classify as "chaos" ~ metastable phase.
    3. Otherwise, classify as "max" ~ solitons.
    """


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


def iterate_and_get_approx_phases(auto, std, window_size, t_max):
    """
    Iterate whole batch of Lenia init configurations, stores mass centers and total masses,
    and computes approximate dynamical phases based on their stability.
    """

    mass_centers_x = np.zeros((t_max, auto.batch))
    mass_centers_y = np.zeros((t_max, auto.batch))
    total_masses = np.zeros((t_max, auto.batch))

    for t in range(t_max):
        auto.step()
        cm, tm = auto.get_batch_mass_center(auto.state)
        mass_centers_x[t]=cm[0]
        mass_centers_y[t]=cm[1]
        total_masses[t]=tm

    phases = get_batch_phases(auto, mass_centers_x, mass_centers_y, total_masses, std, window_size)

    return phases



def get_approx_data(auto, polygon_size_range, array_size, samples, params):
    """
    Generates approximate dynamical phase data for the Lenia system given by params.
    Hyperparameters:
    t_max : maximal number of iterations for each initial configuration (this is increased 10 times if a max phase is detected)
    std : threshold for stability of mass center 
    window_size : number of iterations over which stability of mass center is checked
    """
    std = 3
    window_size = 200
    t_max = int(np.round(100*np.log2(array_size)))


    batch_size = auto.batch
    folder_name = auto.kernel_path

    print("folder name: ", folder_name)

    g_mju = np.round(params["mu"].item(), 4)
    g_sig = np.round(params["sigma"].item(), 4)

    print("rounded: ", g_mju, g_sig)

    path = f"{folder_name}/data/{g_mju}_{g_sig}.pickle"



    PH_TUP = []
    PH = []

    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
    except:
        data = {"params": params} 

    try:
        _ = data[str(array_size)]
    except:
        data[str(array_size)] = {}

    
    for polygon_size in polygon_size_range:
        start = time.time()
        print("polygon size: ", polygon_size)

        try:
            ph = [t for t in data[str(array_size)][str(polygon_size)]["phase"]]
            num_samples = len(ph)
            print(polygon_size, num_samples, "already computed")
        except:
            num_samples = 0
            data[str(array_size)][str(polygon_size)] = {}
            data[str(array_size)][str(polygon_size)]["phase"] = []
            data[str(array_size)][str(polygon_size)]["seed"] = []
            data[str(array_size)][str(polygon_size)]["sample"] = []

        if num_samples >= samples:
            PH_TUP.append((polygon_size, classify(set(data[str(array_size)][str(polygon_size)]["phase"]))))
            PH.append(classify(set(data[str(array_size)][str(polygon_size)]["phase"]))) 
            print(f"{num_samples} computed, skipping polygon_size {polygon_size}")
            continue

        if samples-num_samples < batch_size:
            print("yes")
            samples = num_samples+batch_size

        for batch_index in range(num_samples, samples, batch_size):
            print("batch index: ", batch_index)

            auto.set_init_voronoi_batch(polygon_size, batch_index)
            #auto.plot_voronoi_batch()
            seeds = auto.seeds

            phases = iterate_and_get_approx_phases(auto, std, window_size, t_max=t_max)

            if "max" in phases:
                auto.set_init_voronoi_batch(polygon_size, batch_index)
                seeds = auto.seeds

                print(f"{g_mju}, {g_sig} - {polygon_size} - {t_max}, found max, recomputing phases")
                print("old phases: ", set(phases))

                t_max = int(np.round(1000*np.log2(array_size)))
                phases = iterate_and_get_approx_phases(auto, std, window_size, t_max=t_max)
                print("new phases: ", set(phases))



            data[str(array_size)][str(polygon_size)]["phase"]+=phases
            data[str(array_size)][str(polygon_size)]["seed"]+=seeds
            indices = list(range(batch_index, batch_index+batch_size))
            data[str(array_size)][str(polygon_size)]["sample"]+=indices


            with open(path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


            print("phase: ", classify(set(data[str(array_size)][str(polygon_size)]["phase"])))
        
        PH_TUP.append((polygon_size, classify(set(data[str(array_size)][str(polygon_size)]["phase"]))))
        PH.append(classify(set(data[str(array_size)][str(polygon_size)]["phase"]))) 
                
        print("time: ", time.time() - start)
        print()

    return PH_TUP, PH



            

#============================== PARAMETERS ================================
array_size = 100  # Size of the Lenia world
dt = 0.1 # Time step size
num_channels= 1 # Number of channels in the Lenia world, we only use 1 channel
device = 0    #index of device available

print(jax.devices())

jax.config.update("jax_default_device", jax.devices()[device])

samples = 64  # total number of data samples per polygon size
B = 64  # batch size

polygon_size_range = [10,20,30,40,50,60,70,80,90] # range of polygon sizes to be explored

k_func = "gauss"     # kernel function type: gaussian bump, other types include "exp" and "quad4"


beta = [1.0]        # beta parameter determining the number of kernel "bumps" and their weight
k_mju = [0.5]       # k_mju parameter determining the center of the kernel, only applicable for func_ = "gauss", other kernel types do not depend on this value
k_sig = [0.15]      # k_sig parameter determining the width of the kernel, only applicable for func_ = "gauss", other kernel types do not depend on this value

k_radius = 13       # kernel radius  

g_mu = 0.15        # growth mu parameter
g_sig = 0.015       # growth sigma parameter

params = {
    'k_size': 2*k_radius+1, 
    'mu': jnp.array([[[g_mu]]]), 
    'sigma': jnp.array([[[g_sig]]]), 
    'beta': jnp.array([[[beta]]]), 
    'mu_k': jnp.array([[[k_mju]]]), 
    'sigma_k': jnp.array([[[k_sig]]]), 
    'weights': jnp.array([[[1.0]]]),
    'func_k': k_func,
} 


#============================= GENERATING DATA ====================================
# Generating dynamical phase data for the Lenia system given by params
# !! Note that the file polygons_{array_size}.pickle should already exist in the utils folder, if it does not exist, generate it with the voronoi_polygons.py script !!

# The data generation is computationally heavy and will run a few days on a gpu
# The results are stored in unif_random_voronoi/kernel_folder/data file

#======================================================================

start = time.time()

auto = MultiLeniaJAX((array_size, array_size), batch=B, num_channels=1, dt=0.1, params=params, device=device)
auto.plot_kernel()

for g_mju in np.arange(0.1, 0.5, 0.005):
    g_mju = np.round(g_mju, 5)
    for g_sig in np.arange(0.001,0.1, 0.001):
        g_sig = np.round(g_sig, 5)

        params["mu"] = jnp.put(params["mu"], jnp.array([0,0,0]), g_mju, inplace=False)
        params["sigma"] = jnp.put(params["sigma"], jnp.array([0,0,0]), g_sig, inplace=False)

        print(params)

        auto = MultiLeniaJAX((array_size, array_size), batch=B, num_channels=1, dt=0.1, params=params, device=device)
        ph_tup, ph = get_approx_data(auto, polygon_size_range, array_size, samples, params) 

        # in case finer polygon sizes are wanted:
        if not "max" in ph:
            if "trans" in ph:
                r_min = min([k for (k, v) in ph_tup if v=="trans"])
                r_max = min(r_min+10, max(polygon_size_range))
                new_polygon_size_range = np.arange(r_min, r_max, 1)
                print(new_polygon_size_range)
                ph_tup, ph = get_approx_data(auto, new_polygon_size_range, array_size, samples, params)
                print(ph)
            elif ("order" in ph) and ("chaos" in ph):
                r_min = max([k for (k, v) in ph_tup if v == "order"])
                r_max = min(r_min+10, max(polygon_size_range))
                new_polygon_size_range = np.arange(r_min, r_max, 1)
                print(new_polygon_size_range)
                ph_tup, ph = get_approx_data(auto, new_polygon_size_range, array_size, samples, params)
                print(ph) 

                
print(time.time()-start)