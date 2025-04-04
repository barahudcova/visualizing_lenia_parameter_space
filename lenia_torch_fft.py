import torch,torch.nn,torch.nn.functional as F
import numpy as np
from utils_torch.batch_params import LeniaParams
from showtens import show_image
from automaton_torch import Automaton
import pygame,os, random
import matplotlib.pyplot as plt
import pickle
import imageio
from utils.voronoi_polygons import load_pattern

class MultiLeniaTorch(Automaton):
    """
        Multi-channel lenia automaton. A multi-colored GoL-inspired continuous automaton. Introduced by Bert Chan.
    """
    def __init__(self, size, batch=1, dt=0.1, num_channels=3, params: LeniaParams | dict=None, param_path=None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (H,W) of ints, size of the automaton
                dt : time-step used when computing the evolution
                num_channels : int, number of channels (C)
                params : LeniaParams class, or dict of parameters containing the following
                    key-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (1,C,C) tensor, mean of growth functions
                    'sigma' : (1,C,C) tensor, standard deviation of the growth functions
                    'beta' :  (1,C,C, # of cores) float, max of the kernel cores 
                    'mu_k' : (1,C,C, # of cores) [0,1.], location of the kernel cores
                    'sigma_k' : (1,C,C, # of cores) float, standard deviation of the kernel cores
                    'weights' : (1,C,C) float, weights for the growth weighted sum
                param_path : path to folder containing saved parameters
                device : str, device 
        """
        super().__init__(size=size)

        self.batch = batch # In principle we can compute many worlds at once, but not useful here.
        self.C = num_channels
        self.device=device

        if(params is None): # Transform to LeniaParams
            self.params = LeniaParams(batch_size=self.batch, k_size=25,channels=self.C, device=device)
        elif(isinstance(params,dict)):
            self.params = LeniaParams(param_dict=params, device=device)
        else:
            self.params = params.to(device)

        self.k_size = self.params['k_size'] # The chosen kernel size

        self.state = torch.rand(self.batch,self.C,self.h,self.w,device=device) # Our world state

        with open(f'utils/polygons{self.h}.pickle', 'rb') as handle:
            polygons = pickle.load(handle)

        self.polygons = polygons
        #except:
        #    print("polygons for this array size not generated yet")


        self.dt = dt

        # Define the kernel, and its fourier transform, used for the convolution
        self.kernel = torch.zeros((self.k_size,self.k_size),device=device)
        self.fft_kernel = torch.zeros((self.batch,self.C,self.C,self.h,self.w),device=device)
        self.normal_weights = torch.zeros((self.batch,self.C,self.C),device=device) # Normalized weights for the growth functions
        
        self.update_params(self.params)

        # For loading and saving parameters
        self.saved_param_path = param_path
        if(self.saved_param_path is not None):
            self.param_files = [file for file in os.listdir(self.saved_param_path) if file.endswith('.pt')]
            self.num_par = len(self.param_files)
            if(self.num_par>0):
                self.cur_par = random.randint(0,self.num_par-1)
            else:
                self.cur_par = None

        self.to_save_param_path = 'SavedParameters/Lenia'

    def to(self,device): # Emulate nn.Module.to
        self.params.to(device)
        self.kernel = self.kernel.to(device)
        self.normal_weights = self.normal_weights.to(device)
    
    def update_params(self, params : LeniaParams, k_size_override = None):
        """
            Updates parameters of the automaton. 
            Changes batch size to match the one of provided params.

            Args:
                params : LeniaParams, prefer the former
                k_size_override : int, override the kernel size of params
        """

        if(k_size_override is not None):
            self.k_size = k_size_override
            if(self.k_size%2==0):
                self.k_size += 1
                print(f'Increased even kernel size to {self.k_size} to be odd')
            params.k_size = self.k_size
        self.params = LeniaParams(param_dict=params.param_dict, device=self.device)

        self.to(self.device)
        self.norm_weights()

        self.batch = self.params.batch_size # update batch size
        self.kernel = self.compute_kernel() # (B,C,C,k_size,k_size)
        self.fft_kernel = self.kernel_to_fft(self.kernel) # (B,C,C,h,w)

    def set_init_voronoi_batch(self, polygon_size=60, init_polygon_index=0, seeds=None):
        if not seeds:
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]
        elif not len(seeds)==self.batch:
            print("number of seeds does not match batch size, reinitializing seeds")
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]

        self.seeds = seeds
        print(seeds)

        states = torch.empty((self.batch, self.C, self.h, self.w)).to(self.device) # (B,C,H,W)
        for i, seed in enumerate(seeds):
            polygon_index = init_polygon_index+i
            mask = self.polygons[polygon_size][polygon_index%1024]
            mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

            np.random.seed(seed)
        
        # Generate random state and apply mask
            rand_np = np.random.rand(self.batch, self.C, self.h, self.w)
            rand = torch.tensor(rand_np, dtype=torch.float32).to(self.device)

            pattern = np.asarray(rand_np * mask)
            state = torch.tensor(np.asarray(pattern)).view(1,self.C,self.h,self.w).float().to(self.device)

            states[i, :, :, :] = state
        
        self.state = states  # (B,C,H,W)

    
    def norm_weights(self):
        """
            Normalizes the relative weight sum of the growth functions
            (A_j(t+dt) = A_j(t) + dt G_{ij}B_ij), here we enforce sum_i B_ij = 1
        """
        # Normalizing the weights
        N = self.params.weights.sum(dim=1, keepdim = True) # (B,1,C)
        self.normal_weights = torch.where(N > 1.e-6, self.params.weights/N, 0)

    def get_params(self) -> LeniaParams:
        """
            Get the LeniaParams which defines the automaton
        """
        return self.params


    def kernel_slice(self, r):
        """
            Given a distance matrix r, computes the kernel of the automaton.
            In other words, compute the kernel 'cross-section' since we always assume
            rotationally symmetric kernels.

            Args :
            r : (k_size,k_size), value of the radius for each pixel of the kernel
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None,None] #(1, 1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,self.C,self.C,self.params.mu_k.shape[3],-1,-1) #(B,C,C,#of cores,k_size,k_size)

        mu_k = self.params.mu_k[..., None, None] # (B,C,C,#of cores,1,1)
        sigma_k = self.params.sigma_k[..., None, None]# (B,C,C,#of cores,1,1)

        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) #(B,C,C,#of cores,k_size,k_size)
        beta = self.params.beta[..., None, None] # (B,C,C,#of cores,1,1)

        K = torch.sum(beta*K, dim = 3) # Sum over the cores with the respective heights (B,C,C,k_size, k_size)

        
        return K #(B,C,C,k_size, k_size) # C*C kernels of shape (k_size,k_size)


    def compute_kernel(self):
        """
            Computes the kernel given the current parameters.
        """
        xyrange = torch.linspace(-1, 1, self.params.k_size).to(self.device)

        X,Y = torch.meshgrid(xyrange, xyrange,indexing='xy') # (k_size,k_size),  axis directions is x increasing to the right, y increasing to the bottom
        r = torch.sqrt(X**2+Y**2) # Radius values for each pixel of the kernel

        K = self.kernel_slice(r) #(B,C,C,k_size,k_size) The actual kernel

        # Normalize the kernel :
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(B,C,C,k_size,k_size), normalized kernels
    
    def kernel_to_fft(self, K):
        """
            Computes the fourier transform of the kernel.
        """
        # Pad kernel to match image size
        # For some reason, pad is left;right, top;bottom, (so W,H)
        K = F.pad(K, [0,(self.w-self.params.k_size)] + [0,(self.h-self.params.k_size)]) # (B,C,C,h,w)

        # Center the kernel on the top left corner for fft
        K = K.roll((-(self.params.k_size//2),-(self.params.k_size//2)),dims=(-1,-2)) # (B,C,C,h,w)

        K = torch.fft.fft2(K) # (B,C,C,h,w)

        return K #(B,C,C,h,w)

    def growth(self, u): # u:(B,C,C,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (B,C,C,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using other types of bumps
        mu = self.params.mu[..., None, None] # (B,C,C,1,1)
        sigma = self.params.sigma[...,None,None] # (B,C,C,1,1)
        mu = mu.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)
        sigma = sigma.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(B,C,C,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        ## TODO : this is where you will do the mixing of the batches
        ## convs[i] contains the convolution result of the i-th batch
        convs = self.get_fftconv(self.state) # (B,C,C,H,W) Convolutions for each channel interaction

        assert (self.h,self.w) == (convs.shape[-2], convs.shape[-1])

        weights = self.normal_weights[...,None, None] # (B,C,C,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

        ## TODO : growths[i] contains the growths of the i-th batch.
        ## If you want to mix and match growths and convs, you can mix
        ## the batch dimension of convs
        growths = self.growth(convs) # (B,C,C,H,W) growths for each channel interaction
        # Weight normalized growth :
        ##TODO : YOU WILL PROBABLY NEED TO COMPUTE dx DIFFERENTLY WHEN YOU DO THE MIXING
        dx = (growths*weights).sum(dim=1) #(B,C,H,W) # Use the weights to sum the growths of each channel

        # Apply growth and clamp
        ## TODO : YOU WILL HAVE TO REWRITE THE STATE UPDATE WHEN YOU DO THE MIXING
        self.state = torch.clamp(self.state + self.dt*dx, 0, 1) # (B,C,H,W)
        
        ## TODO : THIS CAN BE DELETED WHEN YOU DO THE MIXING. YOU SHOULD MAKE SURE
        ## YOU NATURALLY END UP WITH A (1,C,H,W) STATE.
        self.state = self.state[:1] # (1,C,H,W), keep only the first batch, in case of batched parameters
    
    def get_fftconv(self, state):
        """
            Compute convolution of state with kernel using fft
        """
        state = torch.fft.fft2(state) # (1,C,H,W) fourier transform
        state = state.expand(self.batch,-1,-1,-1) # (B,C,H,W) expanded for batched parameters
        state = state[:,:,None] # (B,C,1,H,W)
        state = state*self.fft_kernel # (B,C,C,H,W), convoluted
        state = torch.fft.ifft2(state) # (B,C,C,H,W), back to spatial domain

        return torch.real(state) # (B,C,C,H,W), convolved with the batch of kernels


    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,C) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1,-2)) # (1,C) mean mass for each color

    def draw(self):
        """
            Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0]

        if(self.C==1):
            toshow = toshow.expand(3,-1,-1)
        elif(self.C==2):
            toshow = torch.cat([toshow,torch.zeros_like(toshow)],dim=0)
        else :
            toshow = toshow[:3,:,:]
    
        self._worldmap = toshow


params =   {'k_size': 27, 
            'mu': torch.tensor([[[0.15]]]), 
            'sigma': torch.tensor([[[0.015]]]), 
            'beta': torch.tensor([[[[1.0]]]]), 
            'mu_k': torch.tensor([[[[0.5]]]]), 
            'sigma_k': torch.tensor([[[[0.15]]]]), 
            'weights': torch.tensor([[[1.0]]])}



# Create Lenia instance
polygon_size = 48
lenia = MultiLeniaTorch((500,500), batch=1, num_channels=1, dt=0.1, params=params, device='cpu')
seeds = [42]

# Initialize and save initial state
lenia.set_init_voronoi_batch(polygon_size, seeds=seeds, init_polygon_index=0)
lenia.draw()

# Set up display
pygame.display.set_mode((500, 500))

# Save initial frame
pygame.image.save(lenia.worldsurface, 'lenia0.png')

# Create directory for frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

# Run simulation and save frames
frame_count = 0
for t in range(200):
    lenia.step()
    
    if t % 4 == 0:  # Save every 4th frame to reduce file count
        lenia.draw()
        pygame.image.save(lenia.worldsurface, f"{frames_dir}/frame_{frame_count:04d}.png")
        frame_count += 1
        
        # Update display
        pygame.display.get_surface().blit(lenia.worldsurface, (0, 0))
        pygame.display.flip()

# Save final state
lenia.draw()
pygame.image.save(lenia.worldsurface, f'lenia_polygonsize{polygon_size}_seed{seeds[0]}_time{t}.png')

# Create video from frames
frames = [imageio.imread(f"{frames_dir}/frame_{i:04d}.png") for i in range(frame_count)]
imageio.mimsave(f"lenia_torch_{polygon_size}_{seeds[0]}.mp4", frames, fps=10)

# Clean up
pygame.quit()




""" for seed in np.random.randint(0, 2**32, size=1):
    lenia.set_init_voronoi_batch(polygon_size=48, seeds=[int(seed)], init_polygon_index=0)
    for _ in range(20):
        lenia.step()
    lenia.draw()
    pygame.image.save(lenia.worldsurface, f'lenia1_{seed}.png')
    for _ in range(20):
        lenia.step()
    lenia.draw()
    pygame.image.save(lenia.worldsurface, f'lenia2_{seed}.png')


print(lenia.params.param_dict)

plt.matshow(lenia.kernel[0,0,0], cmap='gray')
plt.savefig('kernel.png') """