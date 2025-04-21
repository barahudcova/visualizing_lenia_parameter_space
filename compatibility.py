from lenia_jax_fft import MultiLeniaJAX
from lenia_numpy_fft import MultiLeniaNumPy
import numpy as np


#--------------------------------------NUMPY-JAX-COMPATIBILITY-EXAMPLE------------------------------------------
"""I test a very specific initial configuration of very specific Lenia parameter values which give rise to two gliders interacting in complicated ways. Such an instance is typically very sensitive to small numerical impricisions.
   Luckily, in this case it seems that the NumPy and JAX implementations give similar results."""



params = {
    'k_size': 27, 
    'mu': np.array([[[0.15]]]), 
    'sigma': np.array([[[0.015]]]), 
    'beta': np.array([[[[1.0]]]]), 
    'mu_k': np.array([[[[0.5]]]]), 
    'sigma_k': np.array([[[[0.15]]]]), 
    'weights': np.array([[[1.0]]])
}



polygon_size = 50
array_size = 100


lenia_numpy = MultiLeniaNumPy((array_size, array_size), batch=1, num_channels=1, dt=0.1, params=params)
lenia_jax   = MultiLeniaJAX((array_size, array_size), batch=1, num_channels=1, dt=0.1, params=params)


seeds = [3410421445]
sample = 142

lenia_numpy.set_init_voronoi_batch(polygon_size, seeds=seeds, init_polygon_index=0)
lenia_jax.set_init_voronoi_batch(polygon_size, seeds=seeds, init_polygon_index=0)

numpy_save_path = f"{lenia_numpy.kernel_path}/compatibility/cluster_numpy_{lenia_numpy.g_mu}_{lenia_numpy.g_sig}_{polygon_size}_{seeds[0]}_{sample}.gif"
jax_save_path = f"{lenia_jax.kernel_path}/compatibility/cluster_jax_{lenia_jax.g_mu}_{lenia_jax.g_sig}_{polygon_size}_{seeds[0]}_{sample}.gif"

lenia_numpy.make_video(seeds, polygon_size, init_polygon_index=sample, sim_time=1000, step_size=4, phase='max', save_path=numpy_save_path)
lenia_jax.make_video(seeds, polygon_size, init_polygon_index=sample, sim_time=1000, step_size=4, phase='max', save_path=jax_save_path)