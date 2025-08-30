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
from ScintCityPython.NN_models import Model1, Model2
import G4_Nanophotonic_Scintillator.utils.generate_database as geant4
import datetime

class PositiveOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
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
                p.data.clamp_(min=0.0)
        
        return loss
class PH_yield(nn.Module):
    def __init__(self,dist_model,mod, sim_model, layer_vec, starts_with_scintilator,num_of_layers):
        super(PH_yield, self).__init__()  # Initialize the parent class

        # Convert layer_vec to a torch.nn.Parameter
        self.layer_vec = nn.Parameter(torch.tensor(layer_vec, dtype=torch.float64))
        self.mod=mod
        self.dist=dist_model
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
        bin_layer_vec, _ = self.bin_layer_vec
        emmsion_density_profile = self.model(torch.tensor(bin_layer_vec).double())
        if self.mod =='2':
            emmsion_density_profile=emmsion_density_profile[0,0,:,:]
        #emmsion_density_profile = emmsion_density_profile.detach().numpy()
        reg_layer_vec = self.layer_vec
        bin_layer_vec, binned_layers_index = self.bin_layer_vec
        return torch.abs(emmsion_density_profile),bin_layer_vec
def generate_rand_length_vec(nLayersNS,sample_size,low=100,high=300):
    scintillator_list=[]
    dialectric_list=[]
    starts_with_scintilator=np.random.choice([0, 1], size=1, p=[0.5, 0.5])[0]
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
    binned_layers_index = [0]  # indexing where layers start and end
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
    return boolean_list.astype(int), np.array(binned_layers_index)
def optimise(dist,mod,model,num_epochs,num_of_ini_points=1,optimisation_flag='constrained',plot_loss=False,num_of_layers=10,total_length=600*10,scint_length=1):
    photon_yield_array=[]
    lr=10
    yield_max=0
    sim_vec=np.nan
    start_with_scint=0
    for ini_point in range(num_of_ini_points):
        
    
        width=600*10
        ini_layer_vec,starts_with_scintilator=generate_rand_length_vec(num_of_layers,2000)
        ini_layer_vec=total_length*ini_layer_vec/np.sum(ini_layer_vec)
        #dist='exponential_optim'
        #dist='optimisation'
        PH_optim = PH_yield(dist,mod,model, ini_layer_vec, starts_with_scintilator,num_of_layers)

        if optimisation_flag=='regular':
            optimizer = optim.Adam([PH_optim.layer_vec], lr=lr)
        elif optimisation_flag=='constrained':
            optimizer =PositiveOptimizer([PH_optim.layer_vec], lr=lr)
        loss_arr = []
        accuracy_arr = []
        opt_stat_scint=0
        
        epoch_yield_array=[]
        for epoch in range(num_epochs):
            PH_optim.train()
            optimizer.zero_grad()
            emmsion_density_profile,bin_layer_vec = PH_optim()  # assuming no inputs are needed
            yield_ph=torch.sum(emmsion_density_profile)
            
            #scintilating_mat=torch.sum(bin_layer_vec)
            layer_loc=PH_optim.layer_vec.detach().numpy()
            scintilating_mat=torch.sum(PH_optim.layer_vec[1-PH_optim.starts_with_scintilator::2])
            total_mat=torch.sum(PH_optim.layer_vec)
            loss = -torch.abs(yield_ph)+0.001*(scintilating_mat-scint_length)**2+0.001*(total_mat-total_length)**2
            epoch_yield_array.append(yield_ph.detach().numpy())
            #loss = -torch.sum(emission_theta[1])
            #loss = torch.tensor(loss, requires_grad=True)  # ensure the loss is a torch tensor
            loss.backward()
            
            optimizer.step()
            
            loss_arr.append(loss.item())
            PH_optim.set_binary_vec()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_arr[-1]:.4f}')
            print(f'layers: {PH_optim.layer_vec}')
            print(f'yield is: {yield_ph}')
            print(f'scint layers vec is {PH_optim.layer_vec[1-PH_optim.starts_with_scintilator::2]}')
            print(f'scinitliating layers are: {scintilating_mat}')
        photon_yield_array.append(np.array(epoch_yield_array))
        
        if(epoch_yield_array[-1]>yield_max):
            yield_max=epoch_yield_array[-1]
            sim_vec=PH_optim.layer_vec
            start_with_scint=PH_optim.starts_with_scintilator
    if plot_loss:
        plt.figure()
        for ep_yield in photon_yield_array:
            plt.plot(ep_yield,'.')
        plt.xlabel('Epoch')
        plt.ylabel('yield')
        plt.title('Training yield over Epochs')
        plt.show()
        plt.savefig('yield runs')
    else:
        return sim_vec.detach().numpy(),start_with_scint
    # Model loading and training setup
def generate_scenraio(nLayersNS,meas_flag,scintillatorThickness,dielectricThickness,startsWithScint,sim_photons):
    tmp=scintillatorThickness
    scintillatorThicknessList=','.join(map(str, tmp))
    #print(scintillatorThicknessList)
    scintillatorThickness=scintillatorThickness[0]
    dialectricThicknessList=','.join(map(str, dielectricThickness))
    dielectricThickness=dielectricThickness[0]
    scenario="""
        /structure/xWorld 0.1
        /structure/yWorld 0.1
        /structure/zWorld 0.1

        /structure/isMultilayerNS 1
        /structure/nLayersNS {}
        /structure/scintillatorType {}
        /structure/substrateThickness 0.1
        /structure/scintillatorThickness {} 
        /structure/dielectricThickness {}
        /structure/scintillatorThicknessList {} 
        /structure/dielectricThicknessList {} 
        /structure/startWithScintillator {} 

        /structure/isAuLayer 0

        /structure/constructDetectors 0
        /structure/nGridX 1
        /structure/nGridY 1
        /structure/detectorDepth 0.0001

        /structure/checkDetectorsOverlaps 1

        /structure/scintillatorLifeTime 2.5
        /structure/scintillationYield 9000.0
        /structure/isPbStoppinglayer 0
        /run/initialize


        /run/beamOn {}
        """.format(nLayersNS,meas_flag,scintillatorThickness,dielectricThickness,scintillatorThicknessList,dialectricThicknessList,startsWithScint,sim_photons)
    return scenario
def replace_and_sum_neighbors(arr):
    i = 0
    new_arr=[]
    while i < len(arr)-1:
        # If the current element is less than 1
        if arr[i] < 1:
            start = i
            while i < len(arr) and arr[i] < 1:
                i += 1
            end = i
            # Sum neighbors around the range
            sum_neighbors = 0
            if start > 0:
                sum_neighbors += arr[start - 1]
            if end < len(arr):
                sum_neighbors += arr[end]
            if (end-start)%2==0:
               new_arr.append(arr[end]) 
            else:
                new_arr.append(sum_neighbors) 
        else:
            new_arr.append(arr[i])
            i += 1
    return new_arr
def get_denisty_of_scenario(scenario,file_name='evaluate',database=False,path_pre='G4_Nanophotonic_Scintillator'):
    run_mac_name = 'run.mac'
    root_file_name = file_name + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+'.root'
    scenario = '/system/root_file_name ' + root_file_name + '\n' + scenario
    
    with open(run_mac_name, 'w') as f:
        f.writelines(scenario)
    
    #run_mac_name='run.mac'
    command='./'+path_pre+'/build/NS ' + run_mac_name
    print (command)
    os.system(command)
    #root_file_name='output0.root'
    #photonsX, photonsY, photonsZ = rs.read_simulation(root_file_name, property='fType', key='opticalphoton')
    #eX, eY, eZ = rs.read_simulation(root_file_name, property='fProcess', key='phot')
    #print (photonsZ)
    #os.remove(root_file_name)
    os.remove(run_mac_name) 
    photonsX, photonsY, photonsZ = rs.read_simulation(root_file_name, property='fType', key='opticalphoton')
    os.remove(root_file_name) 
    return photonsX, photonsY, photonsZ

mod = '2'
loss_arr = []
#optimisation_flag='regular'
optimisation_flag='constrained'
#weights_path = r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth"
#weights_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2results/240417105148/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth'
weights_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2results/learning_rate_0.01/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2_parameters.pth'
#weights_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2results/250105130503/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2_parameters.pth'
#weights_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2results/250105114443/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2_parameters.pth'
state_dict = torch.load(weights_path,  map_location=torch.device('cpu'))
if mod == '1':
    model = Model1(state_dict['fc1.weight'].shape[1], state_dict['fc2.weight'].shape[1], state_dict['fc2.weight'].shape[0])
elif mod == '2':
    model = Model2(99, 99, 99 * 99)
model.load_state_dict(state_dict)

d_scint = 200
d_dielectric = 200

theta_cutoff = 0.1

ini_layer_vec = np.array([d_scint, d_dielectric, d_scint, d_dielectric, d_scint, d_dielectric])
starts_with_scintilator = 0
lr_list=[1e3]
lr=1
num_epochs = int(3000)
min_vec=np.zeros(6)
total_depth=4000
scint_depth_Value=np.linspace(100,total_depth-200,40)
#scint_depth_Value=[100,500,800,1000]
yield_hybrid=[]
yield_bulk=[]
part_num=10000000
energy=0.01 #MeV
for depth in scint_depth_Value:
    # Use torch optimizer to optimize layer_vec
    #dist='exponential_optim'
    dist='optimisation'
    #optimise(dist,mod,model,num_epochs,num_of_ini_points=10,optimisation_flag='constrained',plot_loss=False,total_length=1200,scint_length=100)
    layers_stuct,start_with_scintilator=optimise(dist,mod,model,num_epochs,num_of_ini_points=20,optimisation_flag='constrained',plot_loss=False,total_length=total_depth,scint_length=depth)
    
    new_stuct=np.array(replace_and_sum_neighbors(layers_stuct))
    print(f'optimised structure {layers_stuct}')
    print(f'concatenated sturcture {new_stuct}')
    scint_layer=new_stuct[1-start_with_scintilator::2]
    dialecric_layer=new_stuct[start_with_scintilator::2]
    if scint_layer.size>dialecric_layer.size:
        scint_layer=scint_layer[:-1]
    elif scint_layer.size<dialecric_layer.size:
        dialecric_layer=dialecric_layer[:-1]
    print(f'layers devision {scint_layer} \n {dialecric_layer}') 
    scenario=geant4.generate_scenario(1e-6*scint_layer,1e-6*dialecric_layer,part_num,startsWithScint=start_with_scintilator,nLayersNS=int(2*scint_layer.size))
    ph_x,ph_y,ph_z=geant4.get_denisty_of_scenario(scenario,'filename',database=True,path_pre='G4_Nanophotonic_Scintillator')
    yield_hybrid.append(len(ph_x))
    scenario=geant4.generate_scenario(1e-6*depth,0,part_num,startsWithScint=1,nLayersNS=1,scintillator_type= 'bulk')
    ph_x,ph_y,ph_z=geant4.get_denisty_of_scenario(scenario,'filename',database=True,path_pre='G4_Nanophotonic_Scintillator')
    yield_bulk.append(len(ph_x))
print('finished optimising')
plt.figure()
plt.plot(scint_depth_Value,np.array(yield_hybrid)/(energy*part_num),'.-',label='hybrid yield')
plt.plot(scint_depth_Value,np.array(yield_bulk)/(energy*part_num),'.-',label='bulk yield')
plt.legend()
plt.xlabel('scintilator depth [nm]')
plt.ylabel('yield [MeV ^-1]')
plt.show()
plt.savefig('yield optimization.png')
plt.savefig('yield optimization.svg')
# Example plotting data
plot_data = {
    "scint_depth_Value": scint_depth_Value,
    "yield_hybrid": np.array(yield_hybrid)/(energy*part_num),
    "yield_bulk": np.array(yield_bulk)/(energy*part_num)
}

# Save the data as a pickle file
with open("simulation_raw_data.pkl", "wb") as file:
    pickle.dump(plot_data, file)
