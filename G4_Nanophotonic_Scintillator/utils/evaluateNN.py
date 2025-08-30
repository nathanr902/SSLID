
import sys
sys.path.append('/home/nathan.regev/software/git_repo/')
import G4_Nanophotonic_Scintillator.utils.generate_database as geant4
import numpy as np
import torch
import matplotlib.pyplot as plt
from ScintCityPython.NN_models import Model1, Model2,ScintProcessStochasticModel,ScintProcessStochasticModel_no_CNN
import pickle
import os
import pandas as pd
from scipy.stats import skew
import G4_Nanophotonic_Scintillator.utils.generate_database as geant4
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

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
def simulate_layers_distribution(scintillator_list,dialectric_list,start_with_scint,nLayersNS=10):
    sim_photons=100000
    bins=100
    max_rad=230e-5
    print(f'scint list is{scintillator_list}')
    print(f'dialectric list is{dialectric_list}')
    sample_size=np.sum(1e-6*scintillator_list)+np.sum(1e-6*dialectric_list)
    filename='simulation_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers'
    scenario=geant4.generate_scenario(1e-6*scintillator_list,1e-6*dialectric_list,sim_photons,startsWithScint=start_with_scint,nLayersNS=nLayersNS)
    ph_x,ph_y,ph_z=geant4.get_denisty_of_scenario(scenario,filename,database=True,path_pre='G4_Nanophotonic_Scintillator')#,path_pre='G4_Nanophotonic_Scintillator')
    ph_r=np.sqrt(np.power(ph_x,2)+np.power(ph_y,2))
        #ph_r=ph_r[ph_r<=max_rad]
        #ph_z=ph_z[ph_r<=max_rad]
    z_bins=np.linspace(0,sample_size,bins)
    r_bins=np.linspace(0,max_rad,bins)
    binned_layesr=geant4.create_boolean_width(z_bins,scintillator_list,dialectric_list,0)
    grid,edges_1,edes_2=np.histogram2d(ph_r,ph_z,bins=(r_bins,z_bins))
    return grid
def process_iteration(i):
    """Function that performs a single iteration of the loop."""
    global model, simulated_vec, state_dict, mod
    global ini_layer_vec, starts_with_dielectric, starts_with_scintilator

    """

    """
    restored_image=0
    grid = simulate_layers_distribution(
        ini_layer_vec[starts_with_dielectric::2],
        ini_layer_vec[starts_with_scintilator::2],
        starts_with_scintilator,
        nLayersNS=10
    )

    return grid  # Return both results
def iterate_network(i):
    res = model(simulated_vec)
    
    if mod == '2' or mod == '3':
        res = res[0, 0, :, :]

    restored_image = np.abs(res.detach().cpu().numpy())
    return restored_image
#generate random vector
#itrate simulation to get distribution
#iterate forward model to get probability desnity function

mod='4'
input_size=99
output_size=input_size*input_size
weights_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/MSE_3/results_wieghts.pth'
#weights_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/chi_3/results_wieghts.pth'
state_dict = torch.load(weights_path,map_location=torch.device('cpu'))
if mod=='1':
    model = Model1( input_size, input_size, output_size)
elif mod=='2':
    model = Model2( input_size, input_size, output_size)
elif mod=='3':
    model = ScintProcessStochasticModel( input_size, input_size, output_size)
elif mod=='4':
    #model = ScintProcessStochasticModel_no_CNN( input_size, input_size, output_size)
    #model = ScintProcessStochasticModel_no_CNN(state_dict['fc1.weight_mu'].shape[1], state_dict['fc2.weight_mu'].shape[1], state_dict['fc2.weight_mu'].shape[0])
    model = ScintProcessStochasticModel_no_CNN( input_size, input_size, output_size)
width=1000
cell_size = torch.tensor(width/100).double()
#ini_layer_vec,starts_with_scintilator=generate_rand_length_vec(10,2000)
ini_layer_vec=200*np.ones(10)
starts_with_scintilator=1
starts_with_dielectric=1-starts_with_scintilator
ini_layer_vec=width*ini_layer_vec/np.sum(ini_layer_vec)
z_bins = np.linspace(0, np.sum(ini_layer_vec), 100)

bin_layer_vec,_ = create_boolean_width(
            z_bins,
            ini_layer_vec[starts_with_dielectric::2],
            ini_layer_vec[starts_with_scintilator::2],
            starts_with_scintilator
        )
print(bin_layer_vec)
#scint_tensor=torch.ones(99).double()
#dialectric_tensor=torch.zeros(99).double()
simulated_vec=torch.tensor(bin_layer_vec).double()
#simulated_vec=torch.tensor(ini_layer_vec)
repetition_num=500
restored_images=[]
simulated_images=[]
"""
for i in range(repetition_num):
    res = model(simulated_vec)
    model.load_state_dict(state_dict)
    if mod =='2'or mod =='3':
        res=res[0,0,:,:]
    restored_images.append(np.abs(res.detach().cpu().numpy()))
    grid=simulate_layers_distribution(ini_layer_vec[starts_with_dielectric::2],ini_layer_vec[starts_with_scintilator::2],starts_with_scintilator,nLayersNS=10)
    simulated_images.append(grid)
"""
"""
with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(process_iteration, i, model, simulated_vec, state_dict, mod,
                        ini_layer_vec, starts_with_dielectric, starts_with_scintilator)
        for i in range(repetition_num)
    ]

    for future in futures:
        restored_image, grid = future.result()
        restored_images.append(restored_image)
        simulated_images.append(grid)
"""
#multithreaded simulations
with Pool(os.cpu_count()) as p:  # Use 10 processes (adjust based on your CPU)
        results = p.map(process_iteration, range(repetition_num))
#simulated_images = zip(*results)
simulated_images = np.array(results)
for i in range(repetition_num):#singlethreaded prediction
    res = model(simulated_vec,cell_size=cell_size)
    model.load_state_dict(state_dict)
    if mod =='2'or mod =='3':
        res=res[0,0,:,:]
    restored_images.append(np.abs(res.detach().cpu().numpy()))
restored_images=np.array(restored_images)
print(f'restored image shape :{restored_images.shape} \n simulated image shape: {simulated_images.shape}')

plt.figure(figsize=(10, 15))
plt.subplot(3,2,1)
plt.imshow(np.mean(restored_images,axis=0), cmap='gray')
plt.colorbar()
plt.title('epxected value  Result')
plt.xlabel('z-axis bins')
plt.ylabel('radial bins')
plt.subplot(3,2,3)
plt.imshow(np.std(restored_images,axis=0), cmap='gray')
plt.colorbar()
plt.title('std')
plt.xlabel('z-axis bins')
plt.ylabel('radial bins')
plt.subplot(3,2,5)
plt.plot(np.mean(restored_images,axis=(0,1)))


plt.subplot(3,2,2)
plt.imshow(np.mean(simulated_images,axis=0), cmap='gray')
plt.colorbar()
plt.title('epxected value  simulation')
plt.xlabel('z-axis bins')
plt.ylabel('radial bins')
plt.subplot(3,2,4)
plt.imshow(np.std(simulated_images,axis=0), cmap='gray')
plt.colorbar()
plt.title('std simulation')
plt.xlabel('z-axis bins')
plt.ylabel('radial bins')
plt.subplot(3,2,6)
plt.plot(np.mean(simulated_images,axis=(0,1)))
# Save the result

plt.savefig('PDF moments comparison _MMSE_fixed_layers.png')
plt.savefig('PDF moments comparison_MMSE_fixed_layers.svg')
data = {
    "restored images": restored_images,
    "simulated_images": simulated_images
}

# Save the dictionary as a pickle file
with open("statistical_analysis_data_MMSE_fixed.pkl", "wb") as file:
    pickle.dump(data, file)

print("Dictionary saved as pickle file successfully.")
#compare PDF