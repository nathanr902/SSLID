from ScintCityPython.TPMRS_single_emitter import calculating_emmision
import torch.optim as optim
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import median_filter
import os
from skimage.metrics import mean_squared_error
from ScintCityPython.NN_models import Model1, Model2,ScintProcessStochasticModel,ScintProcessStochasticModel_no_CNN
class PositiveOptimizer(Optimizer):
    def __init__(self, params, lr=1):
        defaults = dict(lr=0.001)
        super(PositiveOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        # Perform a single optimization step
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                
                # Clamp the parameters to be positive
                p.data.clamp_(min=1.0)
        
        return loss
class emission_profile(nn.Module):
    def __init__(self,mod, sim_model, layer_vec, starts_with_scintilator,num_of_layers):
        super(emission_profile, self).__init__()  # Initialize the parent class

        # Convert layer_vec to a torch.nn.Parameter
        self.layer_vec = nn.Parameter(torch.tensor(layer_vec, dtype=torch.float64))
        self.mod=mod
        self.starts_with_dielectric = 1 - starts_with_scintilator
        self.z_bins = np.linspace(0, np.sum(layer_vec), 100)
        self.starts_with_scintilator = starts_with_scintilator
        self.bin_layer_vec = create_boolean_width(
            self.z_bins,
            self.layer_vec[self.starts_with_dielectric::2].detach().numpy(),
            self.layer_vec[self.starts_with_scintilator::2].detach().numpy(),
            self.starts_with_scintilator
        )
        self.model = sim_model
        self.num_of_layers=num_of_layers
    def set_binary_vec(self):
        self.bin_layer_vec = create_boolean_width(
            self.z_bins,
            self.layer_vec[self.starts_with_dielectric::2].detach().numpy(),
            self.layer_vec[self.starts_with_scintilator::2].detach().numpy(),
            self.starts_with_scintilator
        )

    def forward(self):
        bin_layer_vec,binned_layers_index=self.bin_layer_vec
        if self.mod =='2':
            emmsion_density_profile=emmsion_density_profile[0,0,:,:]
        elif self.mod=='4':
            emmsion_density_profile = self.model(torch.tensor(bin_layer_vec).double(),cell_size=torch.tensor(torch.sum(self.layer_vec)/100).double())
        emmsion_density_profile = torch.abs(emmsion_density_profile)
        return emmsion_density_profile
def generate_rand_length_vec(nLayersNS,sample_size,low=100,high=300):
    scintillator_list=[]
    dialectric_list=[]
    starts_with_scintilator=np.random.choice([0, 1], size=1, p=[0.5, 0.5])[0]
    #starts_with_scintilator=1
    for layer in range (nLayersNS):
        layer_width=np.random.uniform(low, high)
        if layer%2==starts_with_scintilator:
            dialectric_list.append(layer_width)
        else:
            scintillator_list.append(layer_width)
    dialectric_list=np.array(dialectric_list)
    scintillator_list=np.array(scintillator_list)
    thickness=np.zeros(nLayersNS)
    thickness[starts_with_scintilator::2]=dialectric_list
    thickness[1-starts_with_scintilator::2]=scintillator_list
    return thickness,starts_with_scintilator
def create_boolean_width(z_bins, scint, dialect, starts_with_scintilator):
    total_list = np.empty(scint.size + dialect.size, dtype=scint.dtype)
    total_list[(1 - starts_with_scintilator)::2] = scint
    total_list[starts_with_scintilator::2] = dialect
    boolean_list = np.zeros(z_bins.size - 1)
    acumulated_bins_depth = 0
    layer_counter = 0
    partial_sum_of_layers = total_list[0]
    z_diff = z_bins[1] - z_bins[0]
    binned_layers_index = []  # indexing where layers start and end
    for i in range(boolean_list.size):
        if (z_diff + acumulated_bins_depth) < partial_sum_of_layers:  # next bin is within the layer
            boolean_list[i] = (layer_counter + starts_with_scintilator) % 2
        else:
            if layer_counter < (total_list.size - 1):
                layer_counter += 1
                partial_sum_of_layers += total_list[layer_counter]
           
            boolean_list[i] = -1  # interface between layers
        if np.abs(boolean_list[i]) - np.abs(boolean_list[i - 1]) != 0:  # if layers change then need to append the value
            binned_layers_index.append(i)
        acumulated_bins_depth += z_diff
    if binned_layers_index[-1] < i and np.array(binned_layers_index).size<8:
        binned_layers_index.append(i)
    binned_layers_index=binned_layers_index[1:]
    return boolean_list.astype(int), np.array(binned_layers_index)
def optimise(mod,model,num_epochs,num_of_ini_points=1,optimisation_flag='constrained',plot_loss=False,num_of_layers=10,lr=1e-3,relative_wiehgt=1e-3):
    if plot_loss:
        plt.figure()

    for ini_point in range(num_of_ini_points):
        min_err=10
        width=400*3
        
        ini_layer_vec,starts_with_scintilator=generate_rand_length_vec(num_of_layers,2000)
        ini_layer_vec=width*ini_layer_vec/np.sum(ini_layer_vec)
        #dist='exponential_optim'
        #dist='optimisation'
        OnionOptim = emission_profile(mod,model, ini_layer_vec, starts_with_scintilator,num_of_layers)
        
        if optimisation_flag=='regular':
            optimizer = optim.Adam([OnionOptim.layer_vec], lr=lr)
        elif optimisation_flag=='constrained':
            optimizer =PositiveOptimizer([OnionOptim.layer_vec], lr=lr)
        loss_arr = []
        try:
            for epoch in range(num_epochs):
                OnionOptim.train()
                optimizer.zero_grad()
                emission_distribution = OnionOptim()  # assuming no inputs are needed
                sample_size=torch.sum(OnionOptim.layer_vec) 
                yield_ph=torch.sum(emission_distribution,dim=1)
                #print(f'photon yield is {yield_ph}')
                #print(f'size error is {(width-sample_size)**2}')
                half_max_indices_arr=half_max_indices(emission_distribution)
                #loss = torch.dot(yield_ph.double(), half_max_indices_arr.double())
                loss=torch.mean(half_max_indices_arr)
                #print(torch.mean(half_max_indices_arr))
                loss.backward()
                optimizer.step()
                loss_arr.append(loss.item())
                OnionOptim.set_binary_vec()
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_arr[-1]:.4f}')
                print(f'layers: {OnionOptim.layer_vec}')
            if plot_loss:
                plt.plot(loss_arr,label=f'interation number {ini_point}')
            if loss_arr[-1]<min_err:
                min_err=loss_arr[-1]
                min_vec=OnionOptim.layer_vec.detach().numpy()
                opt_stat_scint=starts_with_scintilator
        except:
            print ('final epoch is '+str(epoch))
    if plot_loss:
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()
    print(f'most optimal vec is {min_vec}')
    print(f'sws bit is {opt_stat_scint}')    
    print(f'min error is {min_err}')           
    print(f'most optimal vector is {min_vec} \n stat_with_scintilator {opt_stat_scint}')
    return opt_stat_scint,min_vec,min_err
    # Model loading and training setup
def half_max_indices(tensor_2d):
    """
    Given an NxM 2D torch tensor where each column decays exponentially,
    this function returns a 1D tensor of size M where each element represents
    the approximate index at which each column reaches half of its maximum value.

    Args:
        tensor_2d (torch.Tensor): A 2D tensor of shape (N, M).

    Returns:
        torch.Tensor: A 1D tensor of size M containing the half-max indices for each column.
    """

    max_values, _ = torch.max(tensor_2d, dim=0)  # Get max value for each column
    thershold = max_values / 5  # Compute half max values
    
    indices = torch.arange(tensor_2d.shape[0]).unsqueeze(1).expand_as(tensor_2d)  # Row indices
    mask = tensor_2d <= thershold  # Mask for values below half max
    
    half_max_indices = torch.where(mask, indices, torch.tensor(float('inf'))).min(dim=0)[0]
    
    return half_max_indices
loss_arr = []
#optimisation_flag='regular'
optimisation_flag='constrained'
#weights_path = r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth"
#weights_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2results/240417105148/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth'
#weights_path=r"/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2results/250105130503/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2_parameters.pth"
#weights_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/MSE_3/results_wieghts.pth'
weights_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/chi_3/results_wieghts.pth'
state_dict = torch.load(weights_path,map_location=torch.device('cpu'))
mod='4'
input_size=99
output_size=input_size*input_size
if mod=='1':
    model = Model1( input_size, input_size, output_size)
elif mod=='2':
    model = Model2( input_size, input_size, output_size)
elif mod=='3':
    model = ScintProcessStochasticModel( input_size, input_size, output_size)
elif mod=='4':
    model = ScintProcessStochasticModel_no_CNN( input_size, input_size, output_size)
                
model.load_state_dict(state_dict)

d_scint = 200
d_dielectric = 200
num_epochs = int(200)
optimise(mod,model,num_epochs,num_of_ini_points=1,optimisation_flag='constrained',plot_loss=False)




