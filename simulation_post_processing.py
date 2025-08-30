import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import G4_Nanophotonic_Scintillator.utils.generate_database as geant4
from ScintCityPython.TPMRS_single_emitter import calculating_emmision
import pandas as pd
def smooth_yield_optim():
    file_path = '/home/nathan.regev/software/git_repo/simulation_raw_data.pkl'

    # Open the file in read-binary mode
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    #print(data)
    smoothed_y = savgol_filter(data['yield_hybrid'], window_length=5, polyorder=1)
    plt.figure()
    plt.plot(smoothed_y[10:])
    plt.plot(data['scint_depth_Value'][10:])
    plt.savefig('processesed simulation data .png')
    plt.savefig('processesed simulation data .svg')



    file_path = '/home/nathan.regev/software/git_repo/simulation_purcell_raw_data.pkl'

    # Open the file in read-binary mode
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    theta=data['theta']
    exp_data=data['emission_exp']
    PEP_data=data['emission_optim']
    print(np.sum(PEP_data[(np.abs(theta)<0.3)]))
    print(np.sum(exp_data[(np.abs(theta)<0.3)]))
def plot_alternating_rectangles(thicknesses,title, colors=['blue', 'green']):
    """
    Generates an image of rectangles with alternating colors.

    Parameters:
    - thicknesses: A list of thicknesses for each rectangle.
    - colors: A list of two colors to alternate between (default is ['blue', 'green']).
    """
    # Set up the figure and axis
    thicknesses=10*thicknesses/np.sum(thicknesses)
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 1)  # x-axis goes from 0 to 1
    ax.set_ylim(0, sum(thicknesses))  # y-axis will span the total thickness
    
    current_position = 0  # Start at the bottom of the plot
    
    # Loop through the thicknesses and draw rectangles
    for i, thickness in enumerate(thicknesses):
        color = colors[i % 2]  # Alternate between the two colors
        rect = plt.Rectangle((0, current_position), 1, thickness, color=color)
        ax.add_patch(rect)
        current_position += thickness  # Move the starting position for the next rectangle
    
    # Remove the axis for a cleaner look
    ax.axis('off')
    
    # Display the image
    plt.show()
    plt.savefig('layer_structure '+title+'.svg')
def simulate_total_yield(raw_data_list,start_with_scint=1,nLayersNS=10):
    dialectric_list=raw_data_list[4]#layer_structure[1-start_with_scint::2]
    scintillator_list=raw_data_list[3]
    sim_photons=10000000
    bins=100
    max_rad=230e-5
    sample_size=np.sum(1e-6*scintillator_list)+np.sum(1e-6*dialectric_list)
    filename='simulation_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers'
    scenario=geant4.generate_scenario(1e-6*scintillator_list,1e-6*dialectric_list,sim_photons,startsWithScint=start_with_scint,nLayersNS=nLayersNS)
    ph_x,ph_y,ph_z=geant4.get_denisty_of_scenario(scenario,filename,database=True,path_pre='G4_Nanophotonic_Scintillator')
    ph_r=np.sqrt(np.power(ph_x,2)+np.power(ph_y,2))
        #ph_r=ph_r[ph_r<=max_rad]
        #ph_z=ph_z[ph_r<=max_rad]
    z_bins=np.linspace(0,sample_size,bins)
    r_bins=np.linspace(0,max_rad,bins)
    binned_layesr=geant4.create_boolean_width(z_bins,scintillator_list,dialectric_list,0)
    grid,edges_1,edes_2=np.histogram2d(ph_r,ph_z,bins=(r_bins,z_bins))
    saving_dict={'scintlator':[],'dialectric':[],'binned':[],'grid':[]}
    saving_dict['scintlator'].append(scintillator_list)
    saving_dict['dialectric'].append(dialectric_list)
    saving_dict['binned'].append(binned_layesr)
    saving_dict['grid'].append(grid)
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(saving_dict, f)

    #simulate actual photon propegation 
    absorption_scint=3.95e4 # [nm]]
    absorption_other=4.22e6#[nm]
    control_simulation = {
        'is_Gz': 1,
        'dz_Nz': 10,
        'distribution': 'simulation',
        'simulation_path':filename+'.pkl',
        'absorption_scint': absorption_scint,
        'absorption_other': absorption_other,
        'plot_profile': 0,
        'load_data_from_dir': 1,
        'num_of_layers':nLayersNS
    }
    sim_emmsioion_ditribution,sim_emission_theta,sim_emission_phi=calculating_emmision(control_simulation)
    sim_emission_theta[1]=sim_emission_theta[1].detach().numpy()
    sim_emission_phi[1]=sim_emission_phi[1].detach().numpy()
    os.remove(filename+'.pkl')
    return len(ph_x), sim_emission_theta[1]
def print_logscale():
    #file_path = '/home/nathan.regev/software/git_repo/simulation_purcell_raw_data_stochastical_30_01_25.pkl'
    file_path = '/home/nathan.regev/software/git_repo/simulation_purcell_raw_data_stochastical.pkl'
    # Open the file in read-binary mode
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    """
    plot_data = {
    "theta": sim_emission_theta_exp[0],
    "emission_exp": np.abs(sim_emission_theta_exp[1]),
    "emission_optim": np.abs(sim_emission_theta_optim[1]),
    "emission_optim_chi": np.abs(sim_emission_theta_optim_chi[1]),
    'raw_exp':exp_data,
    'raw_PEP':PEP_data,
    'raw_chi':chi_data

    }
    """
    theta=data['theta']
    exp_data=data['emission_exp']
    
    PEP_data=data['emission_optim']
    
    chi_data=data['emission_optim_chi']

    layer_struct_exp=np.array(data['raw_exp'][0])
    yield_exp,emission_exp=simulate_total_yield(data['raw_exp'],start_with_scint=1)
    layer_struct_optim=np.array(data['raw_PEP'][0])
    yield_PEP,emission_PEP=simulate_total_yield(data['raw_PEP'],start_with_scint=1)
    layer_struct_optim_chi=np.array(data['raw_chi'][0])
    yield_chi,emission_chi=simulate_total_yield(data['raw_chi'],start_with_scint=1)
    print(np.sum(data['raw_chi'][3]+data['raw_chi'][4]))
    print(np.sum(data['raw_exp'][3]+data['raw_exp'][4]))
    print(np.sum(data['raw_PEP'][3]+data['raw_PEP'][4]))
    #print(f'sim_yield PEP is {np.sum(data['raw_PEP'][1])}')
    #print(f'sim_yield exp is {np.sum(data['raw_exp'][1])}')
    window_size=20
    theta_optim=0.5
    sim_emission_theta_optim_chi = pd.Series(np.abs(emission_chi)).rolling(window=window_size).mean()
    sim_emission_theta_optim = pd.Series(np.abs(emission_PEP)).rolling(window=window_size).mean()
    sim_emission_theta_exp= pd.Series(np.abs(emission_exp)).rolling(window=window_size).mean()
    plt.figure()
    plt.plot(theta,np.abs(sim_emission_theta_optim_chi),label='chi distribution optimisation')
    plt.plot(theta,np.abs(sim_emission_theta_exp),label='exponential distribution optimisation')
    plt.plot(theta,np.abs(sim_emission_theta_optim),label='NN distribution optimisation')
    plt.legend()
    plt.show()
    plt.savefig('post_sim_eval_logscale_norm_baseline.png')
    plt.figure()
    plt.semilogy(theta,np.abs(sim_emission_theta_optim_chi),label='chi distribution optimisation')
    plt.semilogy(theta,np.abs(sim_emission_theta_exp),label='exponential distribution optimisation')
    plt.semilogy(theta,np.abs(sim_emission_theta_optim),label='NN distribution optimisation')
    plt.legend()
    plt.show()
    plt.savefig('post_sim_eval_logscale.png')
    plt.figure()
    plt.semilogy(theta,np.abs(chi_data),label='chi distribution optimisation')
    plt.semilogy(theta,np.abs(exp_data),label='exponential distribution optimisation')
    plt.semilogy(theta,np.abs(PEP_data),label='NN distribution optimisation')
    #plt.xlabel(r'$\theta$ - far-field angle [rad]')
    #plt.ylabel('outcoupled emission rate enhancement [a.u.]')
    plt.legend()
    plt.show()
    plt.savefig('long_eval_logscale.png')
    plot_alternating_rectangles(layer_struct_optim,'optim')
    plot_alternating_rectangles(layer_struct_exp,'exp')
    plot_alternating_rectangles(layer_struct_optim_chi,'chi')
    print(f'exp size {np.sum(layer_struct_exp)} \n pep size {np.sum(layer_struct_optim)} \n chi size {np.sum(layer_struct_optim_chi)}  ')
    print(f'exp structure: {layer_struct_exp} , yield is {yield_exp}')
    print(f'PEP structure: {layer_struct_optim}, yield is{yield_PEP} ')
    print(f'chi structure: {layer_struct_optim_chi}, yield is{yield_chi} ')
    print(f'chi purcell gain {np.sum(sim_emission_theta_optim_chi)/yield_chi}')
    print(f'exp purcell gain {np.sum(sim_emission_theta_exp)/yield_exp}')
    print(f'PEP purcell gain {np.sum(sim_emission_theta_optim)/yield_exp}')
    print(f'chi optimisation gain {np.sum(sim_emission_theta_optim_chi[np.abs(theta)<theta_optim])}')
    print(f'exp optimisation gain {np.sum(sim_emission_theta_exp[np.abs(theta)<theta_optim])}')
    print(f'PEP optimisation gain {np.sum(sim_emission_theta_optim[np.abs(theta)<theta_optim])}')
    #plt.savefig('long_eval_stochastical.svg')
def calculate_statistics():
    with open("statistical_analysis_data_chi_fixed.pkl", "rb") as file:
        data = pickle.load(file)

    # Extract the contents
    print(data.keys())
    #restored_images = data.get("restored images", None)
    simulated_images = data['simulated_images']
    mean_image_simulated=np.mean(simulated_images,axis=0)

    std_image_simulated=np.std(simulated_images,axis=0)
    snr_image_simulated=mean_image_simulated/(std_image_simulated+0.01)
    # Print to verify the content
    print('finished_function')
    plt.figure()
    plt.imshow(snr_image_simulated,cmap='gray')
    plt.savefig('SNR_chi.png')

    restored_images = data['restored images']
    mean_image_restored=np.mean(restored_images,axis=0)
    
    std_image_restored=np.std(restored_images,axis=0)
    snr_image_restored=mean_image_restored/(std_image_restored+0.01)
    # Print to verify the content
    print('finished_function')
    plt.figure()
    plt.imshow(snr_image_restored,cmap='gray')
    
    plt.savefig('SNR_chi_restored.png')
def loss_graph(): 
        with open("/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/250302121319/trainig loss arrays.pkl", "rb") as file:
            data = pickle.load(file)
        print(data.keys())
        plt.figure()
        plt.plot(data['model error'],label='MSE error')
        #plt.plot(data['dialectric_loss_arr'],label='dielectric material error')
        #plt.plot(data['cross_cycle_loss_arr'],label='noise robustness error')
        plt.legend()
        plt.savefig('loss_graphs_post.png')
def alternatig_rectangle_wrapper(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    theta=data['theta']
    exp_data=data['emission_exp']
    
    PEP_data=data['emission_optim']
    
    chi_data=data['emission_optim_chi']
    layer_struct_exp=np.array(data['raw_exp'][0])
    layer_struct_optim=np.array(data['raw_PEP'][0])
    layer_struct_optim_chi=np.array(data['raw_chi'][0])
    layer_struct_optim_chi=np.array(data['raw_chi'][0])
    layer_struct_optim_aneal=np.array(data['raw_aneal'][0])
    plot_alternating_rectangles(layer_struct_optim,'optim')
    plot_alternating_rectangles(layer_struct_exp,'exp')
    plot_alternating_rectangles(layer_struct_optim_chi,'chi')
    plot_alternating_rectangles(layer_struct_optim_aneal,'anneal')
#calculate_statistics()
#print_logscale()
#loss_graph()
alternatig_rectangle_wrapper(r'/home/nathan.regev/software/git_repo/optimisation_results/3000_250528152400/simulation_purcell_raw_data_stochastical.pkl')
