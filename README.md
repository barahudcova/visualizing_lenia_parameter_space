# Visualizing the Structure of Lenia Parameter Space

This code is used to approximately classify single-channel Lenia dynamics from sampled initial configurations into four dynamical classes we define. You can see the results [here](https://lenia-explorer.vercel.app/).

## How to use:

Run the scripts from the project root.

1. Install the requirements in requirements.txt
2. Decide which size of Lenia world you will use, the standard one is 100 x 100.
3. Generate the file polygon100.pickle (or, generally, polygons{array_size}.pickle): In `utils/voronoi_polygons.py`, disable any active example plotting calls, uncomment the `generate_random_polygons(...)` call, and run that script. Comment out the generation call again before running the simulation, since the utility is imported by it.
4. Use the file approx_batch_phases.py to define the Lenia kernel you are interested in and the range of g_mju anf g_sig defining the growth function. Run the code to generate the appropriate data.
5. The results will be saved in ./unif_random_voronoi/kernel_folder/data
6. You can use read_data.py to analyze the data, for instance, to plot the corresponding phase space

Other files:
lenia_jax_fft.py: Classical implementation of Lenia in jax. 

The data-generation workflow uses JAX with 64-bit mode enabled. Results are sensitive to the exact initial values. To reproduce a saved trajectory, reuse the same polygon file and sample index, noise seed, parameters, time step, and numerical environment (including the device). Fresh initializations choose new noise seeds unless supplied explicitly.

## Classification method
Hyperparameters:
array_size (typically 100)
Tmax = 1000*log_2(array_size)
window_size = 200
std = 3

Given a Lenia system with global update function F, and an initial configuration A^0 (a 2D array of values between 0 and 1), we generate a trajectory A^0, F(A^0), ..., F^Tmax(A^0). After each update, we store the configuration's center of mass (x and y coordinates) and total mass. Each trajectory is assigned one of the following three labels. The `classify` functions aggregate these labels across samples, including a transitional label (`trans`); their aggregation rules differ between `approx_phases_jax.py` and `read_data.py`.

### Stable Phase
The stable label (`order`) approximates looping behavior. The code checks whether the final total mass and each final center-of-mass coordinate each exactly match an earlier recorded value. These matches are checked independently and may occur at different earlier times, so this heuristic does not prove that the full configuration repeats.

### Metastable Phase
The metastable label (`chaos`) is assigned if the trajectory fails the stable test and its center of mass stays near its mean over the final `window_size` recorded steps. Each coordinate must remain strictly less than `std` from its own window mean.

### Unclassified
The trajectory satisfies neither criterion and receives the label `max`.








