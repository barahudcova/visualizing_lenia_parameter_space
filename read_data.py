import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from math import *
import matplotlib.colors as mcolors
import json
import shutil
import io

import cProfile, pstats, io
from pstats import SortKey
import jax.numpy as jnp
from lenia_jax_fft import MultiLeniaJAX


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)



def classify(phases):
    if phases == {"order"}:
        return "order"
    elif phases == {"chaos"} or phases == {"chaos", "max"}:
        return "chaos"
    elif phases == {"order", "chaos", "max"}:
        return "max"
    elif phases == {"order", "max"}:
        return "trans"
    elif phases == {"order", "chaos"}:
        return "trans"
    elif phases:
        return "no phase"
    else:
        return "TBA"

def get_phase(k_mju, k_sig, g_mju, g_sig, random_size, mode="unif_random_voronoi", array_size=100):
    path = f'{mode}/{k_mju}_{k_sig}/data/{g_mju}_{g_sig}.pickle'
    file = open(path, "rb")
    data = pickle.load(file)
    return classify(set(data[str(array_size)][str(random_size)]["phase"]))

def get_phase_data(random_size, g_mju, g_sig, folder_name, mode="unif_random_voronoi", array_size=100):
    path = f'{mode}/{folder_name}/data/{g_mju}_{g_sig}.pickle'
    with open(path, "rb") as f:
        data = pickle.load(f)
    #trans = [t for i, t in enumerate(data[str(array_size)][str(random_size)]["trans"]) if not data[str(array_size)][str(random_size)]["phase"][i]=="max"]
    return data[str(array_size)][str(random_size)]["phase"]

def get_global_phase(folder_name, g_mju, g_sig, mode="unif_random_voronoi", array_size=100,):
    phases = set()
    path = f'{mode}/{folder_name}/data/{g_mju}_{g_sig}.pickle'
    with open(path, "rb") as f:
        data = pickle.load(f)[str(array_size)]
    for random_size in np.arange(10, 90, 1):
        try:
            phase = data[str(random_size)]["phase"]
            phases = phases.union(set(phase))
        except:
            continue
    return classify(phases)

def get_phase_video(folder_name, phase, g_mju, g_sig, mode="unif_random_voronoi", array_size=100, time_steps=100, device='cuda'):
    path = f'{mode}/{folder_name}/data/{g_mju}_{g_sig}.pickle'
    file = open(path, "rb")
    data = pickle.load(file)[str(array_size)]

    params = {'k_size': 27, 
          'mu': torch.tensor([[[0.1]]], device='cuda:0'), 
          'sigma': torch.tensor([[[0.04]]], device='cuda:0'), 
          'beta': torch.tensor([[[[1]]]], device='cuda:0'), 
          'mu_k': torch.tensor([[[[0.5]]]], device='cuda:0'), 
          'sigma_k': torch.tensor([[[[0.15]]]], device='cuda:0'), 
          'weights': torch.tensor([[[1]]], device='cuda:0')}
    
    params["mu"][0][0][0] = g_mju
    params["sigma"][0][0][0] = g_sig

    auto = BatchLeniaMC((1,array_size,array_size), dt=0.1, params=params, num_channels=1, device=device)
    auto.to(device)


    polygon_size_range = np.arange(10, 100, 1)

    for polygon_size in polygon_size_range:
        try:
            for sample, (seed, ph) in enumerate(zip(data[str(polygon_size)]["seed"], data[str(polygon_size)]["phase"])):
                if ph == phase:
                    video_path=f"{mode}/{folder_name}/videos/{g_mju}_{g_sig}_{phase}_{polygon_size}_{sample}_{seed}.gif"
                    if os.path.exists(video_path):
                        print(f"polygon size {polygon_size} video already exists")
                    else:
                        print(f"polygon size {polygon_size} generating video")
                        print(polygon_size, sample)
                        auto.set_init_voronoi(polygon_size, sample, seed)
                        auto.make_video(time_steps = time_steps, step=1, config = auto.state, video_path=video_path)
                    break
        except:
            continue
        
def get_all_phase_videos(folder_name, global_phase, local_phase, g_mju_range, g_sig_range, k_mju, k_sig, beta, func_k, mode="unif_random_voronoi", array_size=100):    
    for g_mju in g_mju_range:
        for g_sig in g_sig_range:
            g_mju = np.round(g_mju, 4)
            g_sig = np.round(g_sig, 4)
            try: 
                path = f'{mode}/{folder_name}/data/{g_mju}_{g_sig}.pickle'
                print("path: ", path)
                gphase = get_global_phase(folder_name, g_mju, g_sig)
                with open(path, "rb") as f:
                    data = pickle.load(f)[str(array_size)]

                print(gphase)


                params = {
                    'k_size': 37, 
                    'mu': jnp.array([[[g_mju]]]), 
                    'sigma': jnp.array([[[g_sig]]]), 
                    'beta': jnp.array([[[beta]]]), 
                    'mu_k': jnp.array([[[k_mju]]]), 
                    'sigma_k': jnp.array([[[k_sig]]]), 
                    'weights': jnp.array([[[1.0]]]),
                    'func_k': func_k,
                } 


                auto = MultiLeniaJAX((100, 100), batch=1, num_channels=1, dt=0.1, params=params)


                polygon_size_range = np.arange(10, 100, 1)
                found = False

                if gphase == global_phase:
                    print("global phase: ", gphase)
                    for polygon_size in polygon_size_range:
                        if not found:
                            try:
                                for sample, (seed, ph) in enumerate(zip(data[str(polygon_size)]["seed"], data[str(polygon_size)]["phase"])):
                                    if ph == local_phase:
                                        found = True
                                        video_path=f"{mode}/{folder_name}/videos/{g_mju}_{g_sig}_{local_phase}_{polygon_size}_{sample}_{seed}.gif"
                                        print("video path: ", video_path)
                                        if os.path.exists(video_path):
                                            print(f"gmju: {g_mju}, gsig: {g_sig}, polygon size {polygon_size} video already exists")
                                        else:
                                            print(f"gmju: {g_mju}, gsig: {g_sig}, polygon size {polygon_size} generating video")
                                            seeds = [seed]
                                            auto.make_video(seeds=seeds, polygon_size=polygon_size, init_polygon_index=sample, sim_time=400, step_size=4, phase=ph, save_path=video_path)
                                        break
                            except:
                                continue
            except:
                print("error", g_mju, g_sig)
                continue 

def plot_phase_proportion(folder_name, mode="unif_random_voronoi", array_size=100):
    for g_mju in np.arange(0.1, 0.5, 0.005):
        g_mju = np.round(g_mju, 4)
        for g_sig in np.arange(0.001,0.1, 0.001):
            g_sig = np.round(g_sig, 4)
            
 
            path = f'{mode}/{folder_name}/data/{g_mju}_{g_sig}.pickle'

            T_order = []
            T_max = []
            T_chaos = []


            R_order = []
            R_max = []
            R_chaos = []

            file = open(path, "rb")
            D = pickle.load(file)

            phase = get_global_phase(folder_name, g_mju, g_sig)
            if phase == "max":

                print(g_mju, g_sig)
                random_range = range(0, 100)

                for random_size in random_range:
                    try:
                        data = D[str(array_size)][str(random_size)]
                        phase = data["phase"]
                        order_trans = [1 for v in phase if v == "order"]
                        max_trans = [1 for v in phase if v == "max"]
                        chaos_trans = [1 for v in phase if v == "chaos"]
                        # print("random_size: ", random_size, "# samples: ", len(trans))
                        if order_trans:
                            T_order.append(len(order_trans)/len(phase))
                            R_order.append(random_size)

                        if max_trans:
                            T_max.append(len(max_trans)/len(phase))
                            R_max.append(random_size)

                        if chaos_trans:
                            T_chaos.append(len(chaos_trans)/len(phase))
                            R_chaos.append(random_size)
                        
                    except:
                        continue

                size = 8

                fig, ax = plt.subplots(figsize=(2, 2))

                # Add minor gridlines
                # ax.minorticks_on()

                # Set major and minor ticks
                ax.set_xticks([i  for i in range(0, 91, 20)])  # Major ticks every 10
                ax.set_ylim(0-0.025, 1+0.025)
                ax.set_xlim(0, 90)
                ax.set_yticks([0, 1])
                #ax.set_xticks([i  for i in range(0, 90, 1)], minor=True)  # Minor ticks every 1

                plt.scatter(R_order, T_order, color='#718fdc', label='order', s=size)
                plt.scatter(R_max, T_max, color='#ff9585', label='solitons', s=size)
                plt.scatter(R_chaos, T_chaos, color='#91d4c4', label='chaos', s=size)

                # Add a legend
                # plt.legend()
                # plt.title(mode)

                plt.savefig(f"{mode}/{folder_name}/prop_plots/{g_mju}_{g_sig}.png")
                plt.close()
         
def print_graph(folder_name, mju_lim=0.5, sig_lim=0.1, mode="unif_random_voronoi", array_size=100):
    D = {"order": {"x": [], "y": []}, "chaos": {"x": [], "y": []}, "max": {"x": [], "y": []},  "no phase": {"x": [], "y": []}, "trans": {"x": [], "y": []}, "TBA": {"x": [], "y": []}}



    for g_mju in np.arange(0.1, mju_lim, 0.005):
        g_mju = np.round(g_mju, 4)
        for g_sig in np.arange(0.001,sig_lim, 0.001):
            g_sig = np.round(g_sig, 4)
            try:
                phase = get_global_phase(folder_name, g_mju, g_sig, mode, array_size=array_size)
                print(g_mju, g_sig, phase)
                D[phase]["x"].append(g_sig)
                D[phase]["y"].append(g_mju)
        
            except:
                #print("error", g_mju, g_sig)
                continue 


    fig, ax = plt.subplots(figsize=(8, 6))

    # Set limits for x and y axes (similar to the image)
    ax.set_xlim(0, sig_lim)
    ax.set_ylim(0.1, mju_lim)

    # Add major gridlines
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.7)

    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)

    # Set major and minor ticks
    ax.set_xticks([i / 100 for i in range(0, int(100*sig_lim)+1, 2)])  # Major ticks every 0.02
    ax.set_yticks([i / 10 for i in range(1, int(10*mju_lim)+1)])  # Major ticks every 0.1
    ax.set_xticks([i / 100 for i in range(0, int(100*sig_lim)+1, 1)], minor=True)  # Minor ticks every 0.01
    ax.set_yticks([i / 100 for i in range(10, int(100*mju_lim)+1, 5)], minor=True)  # Minor ticks every 0.05

    # Set labels for axes (σ and μ)
    ax.set_xlabel(r'$\sigma$', fontsize=14)
    ax.set_ylabel(r'$\mu$', fontsize=14)
    size=3

    # Scatter plot with labels
    ax.scatter(D["order"]["x"], D["order"]["y"], color='#718fdc', label='order', s=size)
    ax.scatter(D["chaos"]["x"], D["chaos"]["y"], color='#91d4c4', label='chaos', s=size)
    ax.scatter(D["trans"]["x"], D["trans"]["y"], color='#f4c454', label='trans', s=size)
    ax.scatter(D["max"]["x"], D["max"]["y"], color='#ff9585', label='solitons', s=size)
    ax.scatter(D["no phase"]["x"], D["no phase"]["y"], color='red', label='no phase', s=size)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.savefig(f"{mode}/{folder_name}/phases_data.png")

def rename_and_move_videos(folder_name):
    for file in os.listdir(f"unif_random_voronoi/{folder_name}/videos/."):
        if file.endswith(".gif"):
            src = f"unif_random_voronoi/{folder_name}/videos/"+file

            parsed_file=file.split("_")
            new_file = "_".join(parsed_file[:-2])+".gif"
            print(new_file)

            new_file_path = f"unif_random_voronoi/{folder_name}/git_videos/"+new_file
            print(new_file_path)
            
            shutil.copyfile(src, new_file_path) 


def get_one_huge_data_file(folder_name, array_size=100):
    D = {array_size: []}
    folder_path = f"{folder_name}/data/"
    for file in sorted(os.listdir(folder_path)):
        g_mju, g_sig = file[:-7].split("_")
        data = pickle.load(open(folder_path+file, "rb"))

        g_mju = np.round(float(g_mju), 4)
        g_sig = np.round(float(g_sig), 4)

        params = data["params"]

        for key in params.keys():
            try:
                params[key] = params[key].tolist()
            except:
                continue

        phases = set()
        for random_size in np.arange(10, 90, 1):
            try:
                phase = data[str(array_size)][str(random_size)]["phase"]
                phases = phases.union(set(phase))
            except:
                continue
    

        d = {"x": g_sig, "y": g_mju, "params": params, "phase": classify(phases)}
        d["data"] = data[str(array_size)]
        D[array_size].append(d)

    json_path = f"{folder_name}/all_data.json"
    pickle_path = f"{folder_name}/all_data.pickle"
    json.dump(D, open(json_path, "w"))
    pickle.dump(D, open(pickle_path, "wb"))

   

def get_graph_file(folder_name, mode="unif_random_voronoi"):
    D = []
    gif_prefix=f"https://raw.githubusercontent.com/barahudcova/cont_lenia/refs/heads/main/{folder_name}/git_videos/"
    img_prefix=f"https://raw.githubusercontent.com/barahudcova/cont_lenia/refs/heads/main/{folder_name}/prop_plots/"
    i=0

    for g_mju in np.arange(0.1, 0.5, 0.005):
        g_mju = np.round(g_mju, 4)
        for g_sig in np.arange(0.001,0.1, 0.001):
            g_sig = np.round(g_sig, 4)
            print(g_mju, g_sig)
            try:
                
                file_list = [f for f in os.listdir(f"{mode}/{folder_name}/git_videos") if f.startswith(f"{g_mju}_{g_sig}_")]
                rand_sizes = [int((f.split("_")[-1]).split(".")[0]) for f in file_list]
                gifs = [gif_prefix+f for f in file_list]
                tups = zip(rand_sizes, gifs)
                tups = sorted(tups, key=lambda x: x[0])

                rand_sizes = [t[0] for t in tups]
                gifs = [t[1] for t in tups]

                if len(rand_sizes)>20:
                    rand_sizes = rand_sizes[::5]
                    gifs = gifs[::5]
                prop_img = img_prefix+f"{g_mju}_{g_sig}.png"

               
                phase = get_global_phase(folder_name, g_mju, g_sig, mode)
                if phase != "TBA":
                    D.append({"x": g_sig, "y": g_mju, "phase": phase, "rand_sizes": rand_sizes, "gifs": gifs, "id": i, "prop_img": prop_img})
                
                i+=1
            except:
                print("error", g_mju, g_sig)
                continue


    path = f"{mode}/{folder_name}/phases_data.json"
    json.dump(D, open(path, "w"))


def profile_time(f, top_time):
    pr = cProfile.Profile()
    pr.enable()
    f()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(top_time)
    print(s.getvalue())

#============================== PARAMETERS ================================
W,H = 100,100 # Size of the automaton
array_size = W
dt = 0.1 # Time step size
num_channels= 1
mode = "unif_random_voronoi"



beta = [1.0, 0.5]
k_mju = [0.5]
k_sig = [0.15]

func_k = 'quad4'

params = {
    'k_size': 57, 
    'mu': jnp.array([[[0.15]]]), 
    'sigma': jnp.array([[[0.015]]]), 
    'beta': jnp.array([[[beta]]]), 
    'mu_k': jnp.array([[[k_mju]]]), 
    'sigma_k': jnp.array([[[k_sig]]]), 
    'weights': jnp.array([[[1.0]]]),
    'func_k': func_k,
}



lenia = MultiLeniaJAX((W, H), batch=1, num_channels=1, dt=0.1, params=params)
folder_name = lenia.kernel_folder
kernel_path = lenia.kernel_path


#================================================================================

print_graph(folder_name, array_size=200)

global_phase = "max"
local_phase = "max"

g_mju_range = np.arange(0.1, 0.5, 0.005)
g_sig_range = np.arange(0.001,0.1, 0.001)


#get_one_huge_data_file(kernel_path)

#get_all_phase_videos(folder_name, global_phase, local_phase, g_mju_range, g_sig_range, k_mju, k_sig, beta, func_k, mode="unif_random_voronoi", array_size=200)