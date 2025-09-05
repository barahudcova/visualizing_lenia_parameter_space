from lenia_jax_fft import MultiLeniaJAX
from lenia_numpy_fft import MultiLeniaNumPy
import numpy as np


#--------------------------------------NUMPY-JAX-COMPATIBILITY-EXAMPLE------------------------------------------
"""I test a very specific initial configuration of very specific Lenia parameter values which give rise to two gliders interacting in complicated ways. Such an instance is typically very sensitive to small numerical impricisions.
   Luckily, in this case it seems that the NumPy and JAX implementations give similar results."""

beta = [1, 0.5]
k_mju = [0.5]
k_sig = [0.15]

params = {
    'k_size': 37, 
    'mu': np.array([[[0.2]]]), 
    'sigma': np.array([[[0.022]]]), 
    'beta': np.array([[[beta]]]), 
    'mu_k': np.array([[[k_mju]]]), 
    'sigma_k': np.array([[[k_sig]]]), 
    'weights': np.array([[[1.0]]]),
    'func_k': 'quad4',
} 


""" params = {
    'k_size': 27, 
    'mu': np.array([[[0.15]]]), 
    'sigma': np.array([[[0.015]]]), 
    'beta': np.array([[[[1.0]]]]), 
    'mu_k': np.array([[[[0.5]]]]), 
    'sigma_k': np.array([[[[0.15]]]]), 
    'weights': np.array([[[1.0]]]),
    'func_k': 'exp_mu_sig'
} """



polygon_size = 40
array_size = 100

sim_time = 10000


lenia_numpy = MultiLeniaNumPy((array_size, array_size), batch=1, num_channels=1, dt=0.1, params=params)
lenia_jax   = MultiLeniaJAX((array_size, array_size), batch=1, num_channels=1, dt=0.1, params=params)


seeds = [3410421445]
sample = 142

lenia_numpy.set_init_voronoi_batch(polygon_size, seeds=seeds, init_polygon_index=0)
lenia_jax.set_init_voronoi_batch(polygon_size, seeds=seeds, init_polygon_index=0)

numpy_save_path = f"{lenia_numpy.kernel_path}/compatibility/cluster_numpy_{lenia_numpy.g_mu}_{lenia_numpy.g_sig}_{polygon_size}_{seeds[0]}_{sample}_simtime{sim_time}.gif"
jax_save_path = f"{lenia_jax.kernel_path}/compatibility/cluster_jax_{lenia_jax.g_mu}_{lenia_jax.g_sig}_{polygon_size}_{seeds[0]}_{sample}_simtime{sim_time}.gif"

lenia_numpy.make_video(seeds, polygon_size, init_polygon_index=sample, sim_time=sim_time, step_size=4, phase='max', save_path=numpy_save_path)
lenia_jax.make_video(seeds, polygon_size, init_polygon_index=sample, sim_time=sim_time, step_size=4, phase='max', save_path=jax_save_path)