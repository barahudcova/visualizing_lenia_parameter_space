# Visualizing the Structure of Lenia Parameter Space

This code is used to automatically attribute to each single-channel Lenia one of four dynamical classes.

## How to use:
1. Install the requirements in requirements.txt
2. Decide which size of Lenia world will you use, the standard one is 100 x 100.
3. Generate the file polygon100.pickle (or, generally, polygons{array_size}.pickle)
4. Use the file approx_batch_phases.py to define the Lenia kernel you are interested in and the range of g_mju anf g_sig defining the growth function. Run the fila to generate the appropriate data.
5. The results will be saved in ./unif_random_voronoi/kernel_folder/data
6. You can use read_data.py to analyze the data, for instance, to plot the corresponding phase space 

Other files:
lenia_jax_fft.py: Classical implementation of Lenia in jax. 



## Classification method
Hyperparameters:
array_size (typically 100)
Tmax = 1000*log_2(array_size)
window_size = 200
std = 3

Given a Lenia system with global update function F, and an initial configuration A^0 which (a 2_d array of values between 0 and 1) we generate a trajectory A^0, F(A^0), ..., F^Tmax(A^0). 


