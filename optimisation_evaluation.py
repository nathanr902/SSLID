import sys
sys.path.append('/home/nathan.regev/software/git_repo/')
import ScintCityPython.END2END_optim as opt_block
from ScintCityPython.TPMRS_single_emitter import calculating_emmision
import G4_Nanophotonic_Scintillator.utils.generate_database as geant4
import numpy as np
import torch
import matplotlib.pyplot as plt
from ScintCityPython.NN_models import Model1,Model2,ScintProcessStochasticModel,ScintProcessStochasticModel_no_CNN
import pickle
import os
import pandas as pd
from datetime import datetime
def get_opt_struct_and_dist(opt_type,
                        weights_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2results/learning_rate_0.01/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2_parameters.pth',
                        mod='2',
                        num_of_layers=10,
                        width=2000,
                        lr=1e-3,
                        relative_wiehgt=1e-3,
                        num_epochs = 2000,
                        num_of_ini_points=20,
                        optimsiation_flag='gradient'
                        ):
    input_size=99
    output_size=input_size*input_size
    loss_arr = []
    #optimisation_flag='regular'
    optimisation_flag='constrained'
    #weights_path = r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth"
    #weights_path
    #state_dict = torch.load(weights_path)
    state_dict = torch.load(weights_path,map_location=torch.device('cpu'))
    """
    if mod == '1':
        model = Model1(state_dict['fc1.weight'].shape[1], state_dict['fc2.weight'].shape[1], state_dict['fc2.weight'].shape[0])
    elif mod == '2':
        model = Model2(99, 99, 99 * 99)
    """
    if opt_type=='optimisation':
        if mod=='1':
            model = Model1( input_size, input_size, output_size)
        elif mod=='2':
            model = Model2( input_size, input_size, output_size)
        elif mod=='3':
            model = ScintProcessStochasticModel( input_size, input_size, output_size)
        elif mod=='4':
            model = ScintProcessStochasticModel_no_CNN( input_size, input_size, output_size)
                
        model.load_state_dict(state_dict)
    else:
        model=None

    d_scint = 200
    d_dielectric = 200
    starts_with_scintilator = 0

    min_vec=np.zeros(num_of_layers)
    # Use torch optimizer to optimize layer_vec
    if optimsiation_flag=='gradient':
        opt_stat_scint,min_vec,min_err=opt_block.optimise(opt_type,mod,model,num_epochs,num_of_ini_points=num_of_ini_points,optimisation_flag='constrained',plot_loss=False,num_of_layers=num_of_layers,lr=lr,relative_wiehgt=relative_wiehgt,width=width)
    else:
        starts_with_scintilator=1
        opt_stat_scint_1,min_vec_1,min_err_1=opt_block.simulated_annealing(opt_type,mod,model, starts_with_scintilator,num_of_layers, width, relative_wiehgt, num_epochs, theta_cutoff=0.5, plot_loss=False)
        starts_with_scintilator=0
        opt_stat_scint_0,min_vec_0,min_err_0=opt_block.simulated_annealing(opt_type,mod,model, starts_with_scintilator,num_of_layers, width, relative_wiehgt, num_epochs, theta_cutoff=0.5, plot_loss=False)
        if min_err_0<min_err_1:
            opt_stat_scint=opt_stat_scint_0
            min_vec=min_vec_0
            min_err=min_err_0
        else:
            opt_stat_scint=opt_stat_scint_1
            min_vec=min_vec_1
            min_err=min_err_1
    opt_stat_scint=1-opt_stat_scint
    ## get actual photon distirbution
    nLayersNS=min_vec.size
    dialectric_list=min_vec[1-opt_stat_scint::2]
    scintillator_list=min_vec[opt_stat_scint::2]
    wavelength=430e-6
    N=1
    sim_photons=10000
    bins=100
    max_rad=230e-5
    sample_size=np.sum(1e-6*scintillator_list)+np.sum(1e-6*dialectric_list)
    filename='simulation_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers'
    scenario=geant4.generate_scenario(1e-6*scintillator_list,1e-6*dialectric_list,sim_photons,startsWithScint=opt_stat_scint,nLayersNS=nLayersNS)
    ph_x,ph_y,ph_z=geant4.get_denisty_of_scenario(scenario,filename,database=True,path_pre='G4_Nanophotonic_Scintillator')
    ph_r=np.sqrt(np.power(ph_x,2)+np.power(ph_y,2))
        #ph_r=ph_r[ph_r<=max_rad]
        #ph_z=ph_z[ph_r<=max_rad]
    z_bins=np.linspace(0,sample_size,bins)
    r_bins=np.linspace(0,max_rad,bins)
    binned_layesr=geant4.create_boolean_width(z_bins,scintillator_list,dialectric_list,starts_with_scintilator)
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
    sim_emission_theta[1]=len(ph_x)*sim_emission_theta[1].detach().numpy()
    sim_emission_phi[1]=len(ph_x)*sim_emission_phi[1].detach().numpy()
    os.remove(filename+'.pkl')
    return min_vec,grid,sim_emission_theta,scintillator_list,dialectric_list
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
    plt.savefig('layer_structure '+title)
def sweep_sample_thickness():
    current_date = datetime.now()
    directory_reults=r'/home/nathan.regev/software/git_repo/optimisation_results/sweep_widths'
    if not os.path.exists(directory_reults):
        os.makedirs(directory_reults)
    output_path=os.path.join(directory_reults,current_date.strftime('%y%m%d%H%M%S'))
    os.mkdir(output_path)
    chi_square_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/chi_3/results_wieghts.pth'
    MSE_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/MSE_3/results_wieghts.pth'
    stochastical_MSE_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4_parameters.pth'
    width_list=np.linspace(2000,7000,5)
    loss_exp=[]
    loss_chi=[]
    loss_PEP=[]
    emission_exp=[]
    emission_chi=[]
    emission_PEP=[]
    num_epochs = 5000
    num_of_ini_points=5
    theta_thershold=0.5
    window_size=20
    for width in width_list:
        print(f'begin {width} optimisation')
        layer_struct_exp,grid_exp,sim_emission_theta_exp,scintillator_list_exp,dialectric_list_exp=get_opt_struct_and_dist('exponential_optim',num_of_layers=10,width=width,num_epochs = num_epochs,num_of_ini_points=num_of_ini_points)
        exp_data=[layer_struct_exp,grid_exp,sim_emission_theta_exp,scintillator_list_exp,dialectric_list_exp]
        
        
        print('finished exponential')

        layer_struct_optim_chi,grid_optim_chi,sim_emission_theta_optim_chi,scintillator_list_chi,dialectric_list_chi=get_opt_struct_and_dist('optimisation',weights_path=chi_square_path,mod='4',num_of_layers=10,width=width,num_epochs = num_epochs,num_of_ini_points=num_of_ini_points)
        
        chi_data=[layer_struct_optim_chi,grid_optim_chi,sim_emission_theta_optim_chi,scintillator_list_chi,dialectric_list_chi]
        print('finished chi_optimiseation')
        layer_struct_optim,grid_optim,sim_emission_theta_optim,scintillator_list_optim,dialectric_list_optim=get_opt_struct_and_dist('optimisation',weights_path=MSE_path,mod='4',num_of_layers=10,width=width,num_epochs = num_epochs,num_of_ini_points=num_of_ini_points)
        
        PEP_data=[layer_struct_optim,grid_optim,sim_emission_theta_optim,scintillator_list_optim,dialectric_list_optim]
        print('finished MSE_optimiseation')
        
        sim_emission_theta_optim_chi[1] = pd.Series(np.abs(sim_emission_theta_optim_chi[1])).rolling(window=window_size).mean()
        sim_emission_theta_optim[1] = pd.Series(np.abs(sim_emission_theta_optim[1])).rolling(window=window_size).mean()
        sim_emission_theta_exp[1] = pd.Series(np.abs(sim_emission_theta_exp[1])).rolling(window=window_size).mean()
        loss_PEP.append(np.sum(np.abs(sim_emission_theta_optim[1])))
        loss_chi.append(np.sum(np.abs(sim_emission_theta_optim_chi[1])))
        loss_exp.append(np.sum(np.abs(sim_emission_theta_exp[1])))
        
        emission_exp.append(np.abs(sim_emission_theta_exp[1]))
        emission_chi.append(np.abs(sim_emission_theta_optim_chi[1]))
        emission_PEP.append(np.abs(sim_emission_theta_optim[1]))
    plot_data = {
    "theta": sim_emission_theta_optim[0],
    'loss_PEP':loss_PEP,
    'loss_chi':loss_chi,
    'loss_exp ':loss_exp,
    'emission_exp':emission_exp,
    'emission_chi':emission_chi,
    'emission_PEP':emission_PEP
    }
    """
    plot_data = {
        "theta": sim_emission_theta_exp[0],
        "emission_exp": np.abs(sim_emission_theta_exp[1]),
        "emission_optim": np.abs(sim_emission_theta_optim[1])
    }
    """
    # Save the data as a pickle file
    with open(os.path.join(output_path,"optimisation_sweep.pkl"), "wb") as file:
        pickle.dump(plot_data, file)
    plt.figure()
    plt.plot(width_list,loss_PEP,'.',label='MSE optimisation')
    plt.plot(width_list,loss_chi,'.',label='chi optimisation')
    plt.plot(width_list,loss_exp,'.',label='exp optimisation')
    plt.legend()
    plt.savefig(os.path.join(output_path,'optimisation_sweep.png'))
    plt.savefig(os.path.join(output_path,'optimisation_sweep.svg'))
    
#dist='exponential_optim'
#dist='optimisation'
def optimise_single_width():
    current_date = datetime.now()
    directory_reults=r'/home/nathan.regev/software/git_repo/optimisation_results'
    if not os.path.exists(directory_reults):
        os.makedirs(directory_reults)
    width=2000
    output_path=os.path.join(directory_reults,f"{width}_"+current_date.strftime('%y%m%d%H%M%S'))
    os.mkdir(output_path)
    
    chi_square_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/chi_3/results_wieghts.pth'
    MSE_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/MSE_3/results_wieghts.pth'
    stochastical_MSE_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4_parameters.pth'
    layer_struct_aneal,grid_aneal,sim_emission_theta_aneal,scintillator_list_aneal,dialectric_list_aneal=get_opt_struct_and_dist('optimisation',weights_path=chi_square_path,num_of_ini_points=1,num_epochs=10000,mod='4',num_of_layers=10,width=width,optimsiation_flag='anealing')
    annelaing_data=[layer_struct_aneal,grid_aneal,sim_emission_theta_aneal,scintillator_list_aneal,dialectric_list_aneal]
    print('finished anealing_optimiseation')
    layer_struct_exp,grid_exp,sim_emission_theta_exp,scintillator_list_exp,dialectric_list_exp=get_opt_struct_and_dist('exponential_optim',num_of_layers=10,width=width)
    exp_data=[layer_struct_exp,grid_exp,sim_emission_theta_exp,scintillator_list_exp,dialectric_list_exp]
    print('finished exponential')
    layer_struct_optim_chi,grid_optim_chi,sim_emission_theta_optim_chi,scintillator_list_chi,dialectric_list_chi=get_opt_struct_and_dist('optimisation',weights_path=chi_square_path,mod='4',num_of_layers=10,width=width)
    print('finished chi_optimiseation')
    chi_data=[layer_struct_optim_chi,grid_optim_chi,sim_emission_theta_optim_chi,scintillator_list_chi,dialectric_list_chi]
    layer_struct_optim,grid_optim,sim_emission_theta_optim,scintillator_list_optim,dialectric_list_optim=get_opt_struct_and_dist('optimisation',weights_path=MSE_path,mod='4',num_of_layers=10,width=width)
    PEP_data=[layer_struct_optim,grid_optim,sim_emission_theta_optim,scintillator_list_optim,dialectric_list_optim]
    print('finished MSE_optimiseation')



    #plot_alternating_rectangles(layer_struct_optim,'optim')
    #plot_alternating_rectangles(layer_struct_exp,'exp')
    avg_optim=[]
    avg_exp=[]
    window_size=20
    sim_emission_theta_optim_chi[1] = pd.Series(np.abs(sim_emission_theta_optim_chi[1])).rolling(window=window_size).mean()
    sim_emission_theta_optim[1] = pd.Series(np.abs(sim_emission_theta_optim[1])).rolling(window=window_size).mean()
    sim_emission_theta_exp[1] = pd.Series(np.abs(sim_emission_theta_exp[1])).rolling(window=window_size).mean()
    sim_emission_theta_aneal[1] = pd.Series(np.abs(sim_emission_theta_aneal[1])).rolling(window=window_size).mean()
    plt.figure()
    plt.plot(sim_emission_theta_optim[0],np.abs(sim_emission_theta_optim_chi[1]),label='chi distribution optimisation')
    plt.plot(sim_emission_theta_optim[0],np.abs(sim_emission_theta_exp[1]),label='exponential distribution optimisation')
    plt.plot(sim_emission_theta_optim[0],np.abs(sim_emission_theta_aneal[1]),label='anneal distribution optimisation')
    plt.plot(sim_emission_theta_optim[0],np.abs(sim_emission_theta_optim[1]),label='NN distribution optimisation')
    #plt.xlabel(r'$\theta$ - far-field angle [rad]')
    #plt.ylabel('outcoupled emission rate enhancement [a.u.]')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_path,'long_eval_stochastical.png'))
    plt.savefig(os.path.join(output_path,'long_eval_stochastical.svg'))
    plot_data = {
        "theta": sim_emission_theta_optim[0],
        "emission_exp": np.abs(sim_emission_theta_exp[1]),
        "emission_optim": np.abs(sim_emission_theta_optim[1]),
        "emission_optim_chi": np.abs(sim_emission_theta_optim_chi[1]),
        "emission_optim_aneal": np.abs(sim_emission_theta_aneal[1]),
        'raw_exp':exp_data,
        'raw_PEP':PEP_data,
        'raw_chi':chi_data,
        'raw_aneal':annelaing_data
    }
    """
    plot_data = {
        "theta": sim_emission_theta_exp[0],
        "emission_exp": np.abs(sim_emission_theta_exp[1]),
        "emission_optim": np.abs(sim_emission_theta_optim[1])
    }
    """
    # Save the data as a pickle file
    with open(os.path.join(output_path,"simulation_purcell_raw_data_stochastical.pkl"), "wb") as file:
        pickle.dump(plot_data, file)
        
#sweep_sample_thickness()
optimise_single_width()