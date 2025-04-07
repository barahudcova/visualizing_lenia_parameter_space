import torch
import numpy as np, math

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.path import Path
import pickle
from tqdm import tqdm
# Multiprocessing imports removed
from functools import partial

def load_pattern(pattern, array_sizes):
    """take a pattern and embed it in a larger array of zeros, placing it roughly in the middle"""
    array_mids = [size // 2 for size in array_sizes]
    array = np.zeros([1] + array_sizes)
    pattern = np.asarray(pattern)
    _, w, h = pattern.shape
    x1 = array_mids[0] - w//2;  x2 = x1 + w
    y1 = array_mids[1] - h//2;  y2 = y1 + h
    array[:, x1:x2, y1:y2] = pattern
    return array

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def crop_mask(a):
    """crop an array to the smallest rectangle that contains all the non-zero elements"""
    try:
        coords = np.argwhere(a)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        cropped = a[x_min:x_max + 1, y_min:y_max + 1]
    except:
        return np.zeros(a.shape)
    return cropped

def generate_random_polygons(array_size, rand_sizes, samples, seed=42):
    # Single-process version
    np.random.seed(seed)

    rand_sizes = list(rand_sizes)
    AREA = {}
    
    # Initialize result dictionary
    for rand_size in rand_sizes:
        AREA[rand_size] = []

    AREA['seed'] = seed
    
    # precompute k = 1 canvas
    x1, y1 = np.meshgrid(np.arange(array_size), np.arange(array_size))
    x1, y1 = x1.flatten(), y1.flatten()
    points_1 = np.vstack((x1, y1)).T
    rand_points_range_1 = range(4, 50)

    # precompute k = 2 canvas 
    x2, y2 = np.meshgrid(np.arange(2 * array_size), np.arange(2 * array_size))
    x2, y2 = x2.flatten(), y2.flatten()
    points_2 = np.vstack((x2, y2)).T
    rand_points_range_2 = range(3, 8)
    
    for rand_size in tqdm(rand_sizes, desc="Processing sizes"):
        L = 0
        
        while L < samples:
            if rand_size < array_size/2:
                k = 1
                points = points_1
                rand_points_range = rand_points_range_1
            else:
                k = 2
                points = points_2
                rand_points_range = rand_points_range_2
      
            for rand_points in rand_points_range:
                # Continue collecting samples only if needed
                if len(AREA[rand_size]) >= samples:
                    break
                    
                rpoints = np.random.uniform(0, k*array_size, (rand_points, 2))
                vor = Voronoi(rpoints)
                regions, vertices = voronoi_finite_polygons_2d(vor)
                rand_vertices = [[vertices[i] for i in region] for region in regions]
                ps = [Path(rv) for rv in rand_vertices]
                grids = [p.contains_points(points) for p in ps]
                masks = [np.asarray(grid.reshape(k*array_size, k*array_size)) for grid in grids]
                masks = [crop_mask(m) for m in masks]

                masks = [m for m in masks if max(m.shape)<array_size+1]
                areas = [int(np.round(np.sqrt(np.sum(mask)))) for mask in masks]
    
                for i, area in enumerate(areas):
                    try:
                        if area in AREA and len(AREA[area]) < samples:
                            AREA[area].append(masks[i])
                    except:
                        continue
        
            L = len(AREA[rand_size])
    
    with open(f'utils/polygons{array_size}.pickle', 'wb') as handle:
        pickle.dump(AREA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return AREA

def plot_voronoi(array_size, rand_points):
    points = np.random.uniform(0, array_size, (rand_points, 2))
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    index = np.random.randint(0, len(regions))
    rand_region = regions[index]
    rand_vertices = [vertices[i] for i in rand_region]

    x, y = np.meshgrid(np.arange(array_size), np.arange(array_size))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    #
    p = Path(rand_vertices)  # make a polygon
    grid = p.contains_points(points)
    mask = np.transpose(np.asarray(grid.reshape(array_size, array_size), dtype=int))
    area = np.sum(mask)
    print("area: ", area, "sqrt area:",  int(np.round(np.sqrt(area))))

    plotx_points = []
    ploty_points = []

    nplotx_points = []
    nploty_points = []
    for x in range(array_size):
        for y in range(array_size):
            if mask[x, y]:
                plotx_points.append(x)
                ploty_points.append(y)
            else:
                nplotx_points.append(x)
                nploty_points.append(y)

    plt.scatter(plotx_points, ploty_points, color="darkred", s = 1)
    plt.scatter(nplotx_points, nploty_points, color="darkgray", s = 1)
    plt.savefig("voronoi.png")

def plot_voronoi_sample_from_file(array_size, rand_size, sample):
    with open(f'utils/polygons{array_size}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    mask = np.asarray(data[rand_size][sample], dtype=int)
    init_config = load_pattern(mask.reshape(1, *mask.shape), [array_size, array_size]).reshape(array_size, array_size)
    plt.matshow(init_config)
    plt.savefig(f'polygons_area{rand_size}_sample{sample}.png')

def plot_all_voronoi_samples_from_file(array_size, rand_size):
    with open(f'polygons{array_size}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    samples = data[rand_size]
    num_samples = len(samples)
    
    # Calculate grid dimensions for subplots
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure and subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f'All Samples for Size {rand_size}', fontsize=16)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Plot each sample
    for i, sample_mask in enumerate(samples):
        if i < num_samples:
            mask = np.asarray(sample_mask, dtype=int)
            init_config = load_pattern(mask.reshape(1, *mask.shape), 
                                     [array_size, array_size]).reshape(array_size, array_size)
            im = axes[i].matshow(init_config)
            axes[i].set_xticks([])  # Remove x-axis ticks
            axes[i].set_yticks([])  # Remove y-axis ticks
            axes[i].axis('off')  # Turn off axes completely
    
    # Hide unused subplots
    for i in range(num_samples, grid_size * grid_size):
        axes[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(f'utils/polygons_area{rand_size}_all_samples.png', dpi=150)
    plt.close()
    
    print(f"Saved plot with all {num_samples} samples of size {rand_size}")


#============================== EXAMPLE ================================
#generate polygons of all sizes between 10 and 90, with 10 samples each, all fitting in a 100x100 array (we want at least 128 samples, but ideally 1024)

array_size = 100
rand_sizes = range(10, 91)
samples = 256

seed = 42
generate_random_polygons(array_size, rand_sizes, samples, seed)

#rand_size=60
#plot_all_voronoi_samples_from_file(array_size, rand_size)