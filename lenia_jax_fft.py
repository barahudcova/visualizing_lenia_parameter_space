import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import random
import pygame
import imageio.v2 as imageio
from functools import partial
from jax import jit, vmap
from utils.voronoi_polygons import load_pattern
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

class LeniaParams:
    """JAX version of LeniaParams to store and manage Lenia parameters."""
    def __init__(self, batch_size=1, k_size=25, channels=3, device=None, param_dict=None):
        self.batch_size = batch_size
        self.k_size = k_size
        self.channels = channels
        
        if param_dict is not None:
            self.param_dict = param_dict
            self.mu = jnp.array(param_dict['mu'])
            self.sigma = jnp.array(param_dict['sigma'])
            self.beta = jnp.array(param_dict['beta'])
            self.mu_k = jnp.array(param_dict['mu_k'])
            self.sigma_k = jnp.array(param_dict['sigma_k'])
            self.weights = jnp.array(param_dict['weights'])
            self.k_size = param_dict.get('k_size', k_size)
            self.func_k = param_dict.get('func_k', 'exp_mu_sig')
            self.batch_size = self.weights.shape[0]
        else:
            # Initialize with default values
            self.mu = jnp.array([[[0.1]]])
            self.sigma = jnp.array([[[0.015]]])
            self.beta = jnp.array([[[[1.0]]]])
            self.mu_k = jnp.array([[[[0.5]]]])
            self.sigma_k = jnp.array([[[[0.15]]]])
            self.weights = jnp.array([[[1.0]]])
            self.param_dict = {
                'k_size': self.k_size,
                'mu': self.mu,
                'sigma': self.sigma,
                'beta': self.beta,
                'mu_k': self.mu_k,
                'sigma_k': self.sigma_k,
                'weights': self.weights,
                'func_k': 'exp_mu_sig'
            }
        

    
    def to(self, device):
        # JAX handles device placement differently, so this is a no-op
        # but kept for API compatibility
        pass
    
    def __getitem__(self, key):
        return self.param_dict[key]


class Automaton:
    """Base automaton class."""
    def __init__(self, size):
        self.h, self.w = size
        self._worldmap = None
        self.worldsurface = None
    
    @property
    def worldmap(self):
        return self._worldmap
    
    @worldmap.setter
    def worldmap(self, value):
        self._worldmap = value
        
    def draw(self):
        if self._worldmap is not None:
            # Convert from JAX array to numpy array
            array = np.asarray(self._worldmap.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Create pygame surface from the array
            self.worldsurface = pygame.surfarray.make_surface(array)
            
            return self._worldmap
        return None


class MultiLeniaJAX(Automaton):
    """
    Multi-channel Lenia automaton implemented in JAX.
    A multi-colored GoL-inspired continuous automaton. Originally introduced by Bert Chan.
    """
    def __init__(self, size, batch=1, dt=0.1, num_channels=3, params=None, param_path=None, device=0):
        """
        Initializes automaton.  

        Args:
            size: (H,W) of ints, size of the automaton
            batch: int, batch size for parallel simulations
            dt: float, time-step used when computing the evolution
            num_channels: int, number of channels (C)
            params: LeniaParams class or dict of parameters
            param_path: str, path to folder containing saved parameters
            device: str, device (not used in JAX implementation)
        """
        jax.config.update("jax_default_device", jax.devices()[device])
        super().__init__(size=size)

        self.batch = batch
        self.C = num_channels
        self.device = device  # Not used in JAX, kept for compatibility

        if params is None:
            self.params = LeniaParams(batch_size=self.batch, k_size=25, channels=self.C)
        elif isinstance(params, dict):
            self.params = LeniaParams(param_dict=params)
        else:
            self.params = params

        self.k_size = self.params['k_size']

        if self.params['func_k'] in ['exp', 'quad4']:
            kernel_folder = self.params['func_k']+"_"+"_".join([str(s) for s in self.params["beta"][0,0,0]])+"_"+str(self.k_size)
        else:
            kernel_folder = "_".join([str(s) for s in self.params["mu_k"][0,0,0]])+"_"+"_".join([str(s) for s in self.params["sigma_k"][0,0,0]])+"_"+str(self.k_size)
        
        self.kernel_folder = kernel_folder
        self.kernel_path = "unif_random_voronoi/" + kernel_folder

        if not os.path.exists(self.kernel_path):
            os.mkdir(self.kernel_path)
            os.mkdir(f"{self.kernel_path}/data")


        #print(self.kernel_path)

        self.g_mu = np.round(params["mu"].item(), 4)
        self.g_sig = np.round(params["sigma"].item(), 4)

        self.beta = params["beta"][0][0][0]

        self.mu_k = self.params.mu_k[:, :, :, :, None, None]  # (B,C,C,#cores,1,1)
        self.sigma_k = self.params.sigma_k[:, :, :, :, None, None]  # (B,C,C,#cores,1,1)

        self.data_path = f"unif_random_voronoi/{kernel_folder}/data/{self.g_mu}_{self.g_sig}.pickle"


        # Create a random initial state
        key = jax.random.PRNGKey(0)
        self.state = jax.random.uniform(key, shape=(self.batch, self.C, self.h, self.w))

        #initialize kernel func:
        if self.params['func_k'] == 'exp_mu_sig':
            self.func_k = lambda r: jnp.exp(-((r - self.mu_k) / self.sigma_k)**2 / 2)
        elif self.params['func_k'] == 'exp':
            self.func_k = lambda r: np.exp( 4 - 1 / (r * (1-r)) )
        elif self.params['func_k'] == 'quad4':
            self.func_k = lambda r: (4 * r * (1-r))**4


        # Load polygons for initialization
        try:
            with open(f'utils/polygons{self.h}.pickle', 'rb') as handle:
                self.polygons = pickle.load(handle)
        except:
            print("polygons for this array size not generated yet")
            self.polygons = None

        self.dt = dt

        # Compute normalized weights
        self.normal_weights = self.norm_weights(self.params.weights)
        
        # Compute kernel and its FFT
        self.kernel = self.compute_kernel()
        self.fft_kernel = self.kernel_to_fft(self.kernel)

        ii, jj = jnp.meshgrid(jnp.arange(0, self.w), jnp.arange(0, self.h), indexing='ij')
        # Stack and reshape coordinates
        coords = jnp.stack([jnp.reshape(ii, (-1,)), jnp.reshape(jj, (-1,))], axis=-1)
        coords = coords.astype(jnp.float32)

        # Reshape to (array_size^2, 2, 1)
        self.coords = jnp.reshape(coords, (self.w*self.h, 2, 1))

        # For loading and saving parameters
        self.saved_param_path = param_path
        if self.saved_param_path is not None:
            self.param_files = [file for file in os.listdir(self.saved_param_path) if file.endswith('.pt')]
            self.num_par = len(self.param_files)
            if self.num_par > 0:
                self.cur_par = random.randint(0, self.num_par-1)
            else:
                self.cur_par = None

        self.to_save_param_path = 'SavedParameters/Lenia'
        
        # JIT-compile the step function for performance
        self.jit_step = jit(self._step)

    def to(self, device):
        # JAX handles device placement differently, so this is a no-op
        # but kept for API compatibility
        pass
    
    def update_params(self, params, k_size_override=None):
        """
        Updates parameters of the automaton.
        
        Args:
            params: LeniaParams object
            k_size_override: int, override the kernel size of params
        """
        if k_size_override is not None:
            self.k_size = k_size_override
            if self.k_size % 2 == 0:
                self.k_size += 1
                print(f'Increased even kernel size to {self.k_size} to be odd')
            params.k_size = self.k_size
        
        self.params = LeniaParams(param_dict=params.param_dict)
        self.batch = self.params.batch_size
        
        # Update derived parameters
        self.normal_weights = self.norm_weights(self.params.weights)
        self.kernel = self.compute_kernel()
        self.fft_kernel = self.kernel_to_fft(self.kernel)

    @staticmethod
    def norm_weights(weights):
        """
        Normalizes the relative weight sum of the growth functions.
        
        Args:
            weights: (B,C,C) array of weights
            
        Returns:
            (B,C,C) array of normalized weights
        """
        # Sum weights along the first dimension
        sum_weights = jnp.sum(weights, axis=1, keepdims=True)  # (B,1,C)
        
        # Normalize weights, avoiding division by zero
        return jnp.where(sum_weights > 1e-6, weights / sum_weights, 0)

    def set_init_voronoi_batch(self, polygon_size=60, init_polygon_index=0, seeds=None):
        """
        Initialize state using Voronoi polygons.
        
        Args:
            polygon_size: int, size of polygons
            init_polygon_index: int, starting index for polygons
            seeds: list of ints, random seeds for initialization
        """
        if not seeds:
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]
        elif not len(seeds) == self.batch:
            print("number of seeds does not match batch size, reinitializing seeds")
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]

        self.seeds = seeds

        # Create empty numpy array for states
        states_np = np.empty((self.batch, self.C, self.h, self.w))
        
        for i, seed in enumerate(seeds):
            polygon_index = init_polygon_index + i
            mask = self.polygons[polygon_size][polygon_index % 1024]
            mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

            np.random.seed(seed)
            
            # Generate random state and apply mask
            rand_np = np.random.rand(1, self.C, self.h, self.w)
            pattern = np.asarray(rand_np * mask)
            states_np[i] = pattern[0]
        
        # Convert to JAX array
        self.state = jnp.array(states_np)
        print(self.state.device)

    def plot_voronoi_batch(self, figsize=(15, 10), save_path="inits.png"):
        cmap = "gray"
        """
        Creates and saves a matplotlib figure with all states from a Lenia object.
        
        Args:
            lenia_obj: The Lenia object containing states
            figsize: Tuple (width, height) for the figure size
            save_path: String path where to save the plot
            cmap: Colormap to use for the plots
            title: Main title for the plot
        """
        # Get the states from the Lenia object
        states = self.state  # Shape: (batch, channels, height, width)
        batch_size, channels, height, width = states.shape
        
        # Calculate the grid dimensions for the subplots
        grid_cols = int(np.ceil(np.sqrt(batch_size)))
        grid_rows = int(np.ceil(batch_size / grid_cols))
        
        # Create the figure and subplots
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        
        # Flatten the axes array for easier indexing
        if batch_size > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Handle the case of a single plot
        
        # Plot each state
        for i in range(batch_size):
            # For multichannel data, combine channels
            if channels > 1:
                # Create RGB image if 3 channels, otherwise use first channel
                if channels == 3:
                    state_img = np.transpose(states[i], (1, 2, 0))
                    im = axes[i].imshow(state_img)
                else:
                    im = axes[i].imshow(states[i][0], cmap=cmap)
            else:
                im = axes[i].imshow(states[i][0], cmap=cmap)
            
            axes[i].set_title(f"State {i}")
            axes[i].axis('off')  # Remove axes for cleaner look
        
        # Hide any unused subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')
            
        # Add colorbar if using a colormap
        #if channels == 1 or channels > 3:
        #    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"Plot saved to {save_path}")
        
        return fig, axes


    def kernel_slice(self, r, beta=None):
        """
        Given a distance matrix r, computes the kernel of the automaton.
        
        Args:
            r: (k_size,k_size) array, value of the radius for each pixel of the kernel
            
        Returns:
            (B,C,C,k_size,k_size) array of kernel values
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None, None, :, :]  # (1,1,1,1,k_size,k_size)
        
        # Get number of cores
        num_cores = self.params.mu_k.shape[3]
        
        # Expand r to match batched parameters
        r = jnp.broadcast_to(r, (self.batch, self.C, self.C, num_cores, self.k_size, self.k_size))        

        if beta is None: # implementation with prespecified cores
            beta = self.params.beta[:, :, :, :, None, None]  # (B,C,C,#cores,1,1)
            
        
        K = self.func_k(r)  # (B,C,C,#cores,k_size,k_size)
        K = jnp.sum(beta*K, axis=3)
        
        return K

    def compute_kernel(self):
        """
        Computes the kernel given the current parameters.
        
        Returns:
            (B,C,C,k_size,k_size) array of kernel values
        """
        # Create coordinate grid
        xyrange = jnp.linspace(-1, 1, self.k_size)
        x, y = jnp.meshgrid(xyrange, xyrange, indexing='xy')
        
        # Compute radius values
        r = jnp.sqrt(x**2 + y**2)
        print(r.shape)
        b = len(self.beta)
        beta = None

        if self.params['func_k'] in ['exp', 'quad4']: #making this compatible with Bert's implementation with cores being duplicated according to beta
            r = jnp.where(r > 1, 0, r)
            if b>1:
                Br = b*r
                bs = np.asarray([float(f) for f in self.beta])
                beta = bs[np.minimum(np.floor(Br).astype(int), b-1)]
                r = Br%1
        
        # Compute kernel
        K = self.kernel_slice(r, beta)  # (B,C,C,k_size,k_size)
        
        # Normalize kernel
        summed = jnp.sum(K, axis=(-1, -2), keepdims=True)  # (B,C,C,1,1)
        summed = jnp.where(summed < 1e-6, 1.0, summed)  # Avoid division by zero
        K = K / summed
        
        return K
    
    def plot_kernel(self, save_path=None):
        if not save_path:
            save_path = f"{self.kernel_path}/kernel.png"
        if self.C == 1:
            kernel = self.kernel[0, 0, :]
            #print(kernel.shape)
            knl = load_pattern(kernel, [self.k_size+1, self.k_size+1])
            plt.imshow(1-(knl)[0,:,:], cmap="binary")
            plt.grid(axis='x', color='0.95')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            print("too many channels to plot")

    def kernel_to_fft(self, K):
        """
        Computes the Fourier transform of the kernel.
        
        Args:
            K: (B,C,C,k_size,k_size) array, the kernel
            
        Returns:
            (B,C,C,h,w) array, the FFT of the kernel
        """
        # Create padded kernel
        padded_K = jnp.zeros((self.batch, self.C, self.C, self.h, self.w))
        
        # Place kernel in the center
        k_h, k_w = self.k_size, self.k_size
        start_h, start_w = self.h // 2 - k_h // 2, self.w // 2 - k_w // 2
        
        # Update padded kernel with actual kernel values
        padded_K = padded_K.at[:, :, :, start_h:start_h+k_h, start_w:start_w+k_w].set(K)
        
        # Shift for FFT
        padded_K = jnp.roll(padded_K, [-self.h // 2, -self.w // 2], axis=(-2, -1))
        
        # Compute FFT
        return jnp.fft.fft2(padded_K)

    def growth(self, u):
        """
        Computes the growth function applied to concentrations u.
        
        Args:
            u: (B,C,C,h,w) array of concentrations
            
        Returns:
            (B,C,C,h,w) array of growth values
        """
        # Reshape parameters for broadcasting
        mu = self.params.mu[:, :, :, None, None]  # (B,C,C,1,1)
        sigma = self.params.sigma[:, :, :, None, None]  # (B,C,C,1,1)
        
        # Broadcast to match u's shape
        mu = jnp.broadcast_to(mu, u.shape)
        sigma = jnp.broadcast_to(sigma, u.shape)
        
        # Compute growth function (Gaussian bump)
        return 2 * jnp.exp(-((u - mu)**2 / (sigma)**2) / 2) - 1

    def get_fftconv(self, state):
        """
        Compute convolution of state with kernel using FFT.
        
        Args:
            state: (B,C,h,w) array, the current state
            
        Returns:
            (B,C,C,h,w) array of convolution results
        """
        # Compute FFT of state
        fft_state = jnp.fft.fft2(state)  # (B,C,h,w)
        
        # Reshape for broadcasting with kernel
        fft_state = fft_state[:, :, None, :, :]  # (B,C,1,h,w)
        
        # Multiply in frequency domain
        convolved = fft_state * self.fft_kernel  # (B,C,C,h,w)
        
        # Inverse FFT
        result = jnp.fft.ifft2(convolved)  # (B,C,C,h,w)
        
        return jnp.real(result)

    def _step(self, state):
        """
        Core step function that will be JIT-compiled.
        
        Args:
            state: (B,C,h,w) array, the current state
            
        Returns:
            (B,C,h,w) array, the updated state
        """
        # Compute convolutions
        convs = self.get_fftconv(state)  # (B,C,C,h,w)
        
        # Compute growth
        growths = self.growth(convs)  # (B,C,C,h,w)
        
        # Apply weights
        weights = self.normal_weights[:, :, :, None, None]  # (B,C,C,1,1)
        weights = jnp.broadcast_to(weights, growths.shape)  # (B,C,C,h,w)
        
        # Sum weighted growths
        dx = jnp.sum(growths * weights, axis=1)  # (B,C,h,w)
        
        # Update state
        new_state = jnp.clip(state + self.dt * dx, 0, 1)  # (B,C,h,w)
        
        return new_state  # (B,C,h,w)

    def step(self):
        """
        Steps the automaton state by one iteration.
        """
        self.state = self.jit_step(self.state)

    def mass(self):
        """
        Computes average 'mass' of the automaton for each channel.
        
        Returns:
            (B,C) array, mass of each channel
        """
        return jnp.mean(self.state, axis=(-1, -2))  # (B,C)
    
    def get_batch_mass_center(self, array):
        B, C, H, W = array.shape  # array shape: (B,C,H,W)
        #print("array shape: ", array.shape)
        
        # Reshape array to (H*W, 1, B*C)
        A = jnp.transpose(array, (2, 3, 0, 1)).reshape(H*W, 1, B*C)
        
        # Calculate total mass
        total_mass = jnp.sum(A, axis=0)[-1]  # (B*C)
        #print(total_mass)
        
        # Calculate weighted sum by coordinates
        prod = A * self.coords
        sum_mass = jnp.sum(prod, axis=0)  # (2, B*C)
        
        # Create a mask for non-zero masses
        mask = (total_mass != 0)
        
        # Normalize by total mass where total mass is not zero
        # Using JAX's where function to conditionally update values
        sum_mass = jnp.where(
            mask.reshape(1, -1),  # Reshape mask to match sum_mass dimensions
            sum_mass / jnp.where(mask, total_mass, 1.0),  # Divide by mass where mask is True
            sum_mass  # Keep original values where mask is False
        )
    
        return sum_mass, total_mass
    

    def draw(self):
        """
        Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow = self.state[0]  # (C,h,w)

        if self.C == 1:
            # Expand to 3 channels
            toshow = jnp.broadcast_to(toshow, (3, self.h, self.w))
        elif self.C == 2:
            # Add zero channel
            zeros = jnp.zeros((1, self.h, self.w))
            toshow = jnp.concatenate([toshow, zeros], axis=0)
        else:
            # Use only first 3 channels
            toshow = toshow[:3, :, :]
    
        self._worldmap = toshow
        
        # Create pygame surface
        super().draw()
        
        return toshow

    def make_video(self, seeds=None, polygon_size=60, init_polygon_index=0, sim_time=200, step_size=2, phase=None, save_path=None):
        """
        Create a video from the simulation frames.
        
        Args:
            seeds: list of ints, random seeds for initialization
            polygon_size: int, size of polygons
            save_path: str, path to save the video
        """
        if seeds is None:
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]
        
        self.set_init_voronoi_batch(polygon_size=polygon_size, init_polygon_index=init_polygon_index, seeds=seeds)
        
        # Create directory for frames
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)

        pygame.init()
        pygame.display.set_mode((500, 500))

        frame_count = 0
        
        for t in range(sim_time):
            self.step()
    
            if t % step_size == 0:  # Save every 4th frame to reduce file count
                self.draw()
                pygame.image.save(self.worldsurface, f"{frames_dir}/frame_{frame_count:04d}.png")
                frame_count += 1
        
                # Update display
                pygame.display.get_surface().blit(self.worldsurface, (0, 0))
                pygame.display.flip()

        # Save final state
        self.draw()

        video_dir = f"{self.kernel_path}/videos"
        os.makedirs(video_dir, exist_ok=True)

        if not save_path:
            if not phase:
                save_path = f"{video_dir}/{polygon_size}_{init_polygon_index}_{seeds[0]}_jax.gif"
            else:
                save_path = f"{video_dir}/{polygon_size}_{phase}_{init_polygon_index}_{seeds[0]}_jax.gif"

        # Create video from frames
        frames = [imageio.imread(f"{frames_dir}/frame_{i:04d}.png") for i in range(frame_count)]
        imageio.mimsave(save_path, frames, fps=30)

        pygame.quit()



#---------------------------------EXAMPLE---------------------------------



samples = 64  # total number of data samples per polygon size
beta = [1, 0.5]
k_mju = [0.5]
k_sig = [0.15]


B = 64  # batch size

polygon_size_range = [10,20,30,40,50,60,70,80,90]
#======================================================================


""" params = {
    'k_size': 37, 
    'mu': jnp.array([[[0.2]]]), 
    'sigma': jnp.array([[[0.022]]]), 
    'beta': jnp.array([[[beta]]]), 
    'mu_k': jnp.array([[[k_mju]]]), 
    'sigma_k': jnp.array([[[k_sig]]]), 
    'weights': jnp.array([[[1.0]]]),
    'func_k': 'quad4',
} 



# Create Lenia instance
lenia = MultiLeniaJAX((100, 100), batch=1, num_channels=1, dt=0.1, params=params)
lenia.plot_kernel()

polygon_size = 35
lenia.set_init_voronoi_batch(polygon_size=polygon_size)
#lenia.plot_voronoi_batch(figsize=(15, 10), save_path="inits.png")


seeds=[42]
lenia.make_video(seeds=seeds, polygon_size=polygon_size, sim_time=200, step_size=2)
 """