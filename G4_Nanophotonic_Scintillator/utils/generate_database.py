import os, sys
import pickle
import random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, './analysis')
sys.path.insert(0, '/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/analysis/')

try:
    import G4_Nanophotonic_Scintillator.analysis.read_simulation as rs
except:
    import read_simulation as rs
import datetime
#os.l['DISPLAY'] = 'localhost:10.0'
# Run the script from the build directory
import concurrent.futures
import multiprocessing
import uuid
import pandas as pd
def generate_unique_string():
    # Generate a UUID4, which is a randomly generated unique identifier
    unique_id = uuid.uuid4()
    # Convert the UUID to a string
    unique_string = str(unique_id)
    return unique_string
def generate_scenario(scintillatorThickness,dielectricThickness,sim_photons,startsWithScint=0,nLayersNS=1,scintillator_type= 'aperiodic'):
    if scintillator_type== 'bulk':
        meas_flag=0
        scintillatorThicknessList=','.join(map(str, [scintillatorThickness]))
        #print(scintillatorThicknessList)
        dialectricThicknessList=','.join(map(str, [dielectricThickness]))
        startsWithScint=0
    elif scintillator_type== 'periodic':
        meas_flag=1
        scintillatorThicknessList=''
        dialectricThicknessList=''
    elif scintillator_type== 'aperiodic':
        meas_flag=2
        tmp=scintillatorThickness
        scintillatorThicknessList=','.join(map(str, tmp))
        #print(scintillatorThicknessList)
        scintillatorThickness=scintillatorThickness[0]
        dialectricThicknessList=','.join(map(str, dielectricThickness))
        dielectricThickness=dielectricThickness[0]
    #scintillator_type= 'periodic'
    #scintillator_type= 'aperiodic'


    # Generate two random seed values
    seed1 = random.randint(10000, 99999)
    seed2 = random.randint(10000, 99999)

    scenario = """
        /random/setSeeds {} {}

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
    """.format(seed1, seed2, nLayersNS, meas_flag, scintillatorThickness, dielectricThickness, scintillatorThicknessList, dialectricThicknessList, startsWithScint, sim_photons)


    return scenario
def get_denisty_of_scenario(scenario,file_name,database=False,path_pre=''):
    os.system(". /home/nathan.regev/software/geant4/geant4-v11.2.1-install/share/Geant4/geant4make/geant4make.sh")
    unique_name = generate_unique_string()
    run_mac_name = 'run_' + unique_name + '.mac'
    root_file_name = file_name + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+unique_name+'.root'
    scenario = '/system/root_file_name ' + root_file_name + '\n' + scenario
    
    with open(run_mac_name, 'w') as f:
        f.writelines(scenario)
    
    #run_mac_name='run.mac'
    if(path_pre==''):
        command='./build/NS ' + run_mac_name
    else:
        command='./'+path_pre+'/build/NS ' + run_mac_name
    print (command)
    os.system(command)
    #root_file_name='output0.root'
    #photonsX, photonsY, photonsZ = rs.read_simulation(root_file_name, property='fType', key='opticalphoton')
    #eX, eY, eZ = rs.read_simulation(root_file_name, property='fProcess', key='phot')
    #print (photonsZ)
    #os.remove(root_file_name)
    os.remove(run_mac_name)
    if(database):
       
       photonsX, photonsY, photonsZ = rs.read_simulation(root_file_name, property='fType', key='opticalphoton')
       os.remove(root_file_name) 
       return photonsX, photonsY, photonsZ

def shmoo_width(scintillator_list,dialectric_list,sim_photons=10000):
    results=pd.DataFrame(columns=['scintillatorThickness','dielectricThickness','X_coords','Y_coords','Z_coords'])
    for scintillatorThickness in scintillator_list:
        for dielectricThickness in dialectric_list:
            scenario=generate_scenario(scintillatorThickness,dielectricThickness,sim_photons)
            photonsX,photonsY,photonsZ=get_denisty_of_scenario(scenario)
            zthickness=(scintillatorThickness+dielectricThickness)*15+0.0001
            photonsX_sample=photonsX[photonsZ<=zthickness]
            photonsY_sample=photonsY[photonsZ<=zthickness]
            photonsZ_sample=photonsZ[photonsZ<=zthickness]
            iteration = {"scintillatorThickness": scintillatorThickness, "dielectricThickness": dielectricThickness, "X_coords": photonsX_sample,"Y_coords": photonsY_sample,"Z_coords": photonsZ_sample}
            results= results._append(iteration, ignore_index=True)
    return results
def create_boolean_width(z_bins,scint,dialect,starts_with_scintilator):
    total_list=  np.empty(scint.size + dialect.size, dtype=scint.dtype)
    total_list[(1-starts_with_scintilator)::2]=scint
    total_list[starts_with_scintilator::2]=dialect
    boolean_list=np.zeros(z_bins.size-1)
    acumulated_bins_depth=0
    layer_counter=0
    partial_sum_of_layers=total_list[0]
    z_diff=z_bins[1]-z_bins[0]
    for i in range(boolean_list.size):
        if (z_diff+acumulated_bins_depth)<partial_sum_of_layers: #next bin is within the layer
            boolean_list[i]=(layer_counter+starts_with_scintilator)%2
        else:
            #print(i)
            if layer_counter<(total_list.size-1):
                layer_counter+=1
                partial_sum_of_layers+=total_list[layer_counter]
            boolean_list[i]=-1 #interface between layers
        acumulated_bins_depth+=z_diff
    return boolean_list

def run_database_simulation(nLayersNS=6,data_base_size=5,sim_photons=1000000,wavelength=430e-6,database_type='gaussian',bins=100,max_rad=230e-5,threshold_low=200e-6,sample_size=430e-5):
    filename=str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+'Database_'+str(data_base_size)+'samples_nonpriodic_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers_PbOSiO2'
    print('filename is '+filename)
    saving_dict={}
    keys=['scintlator','dialectric','binned','grid']
    
    
    max_threshold=sample_size#2*sample_size/(nLayersNS)
    for key in keys:
        saving_dict[key]=[]

    #df = pd.DataFrame(columns=['scintlator','dialectric','grid'])
    for i in range(data_base_size):
        starts_with_scintilator=np.random.choice([0, 1], size=1, p=[0.5, 0.5])[0]
        sample_size=np.random.uniform(wavelength*(nLayersNS-2), wavelength*(nLayersNS+2), size=1)
        sample_size=sample_size[0]
        print(starts_with_scintilator)
        if database_type=='uniform':
            low=wavelength/3
            high=2*wavelength/3
            scintillator_list=[]
            dialectric_list=[]
            layer_width = np.random.uniform(low, high, size=nLayersNS)
            layer_width=sample_size*layer_width/(np.sum(layer_width))
            for layer,ind_layer in zip(layer_width,range (nLayersNS)):
                if ind_layer%2==starts_with_scintilator:
                    dialectric_list.append(layer)
                else:
                    scintillator_list.append(layer)
            dialectric_list=np.array(dialectric_list)
            scintillator_list=np.array(scintillator_list)
        elif database_type=='gaussian':
            sigma=wavelength/2
            mu=wavelength/2
            scintillator_list=[]
            dialectric_list=[]
            for layer in range (nLayersNS):
                if layer<nLayersNS-1:
                    layer_width=np.random.normal (mu,sigma,1)[0]
                    while layer_width<threshold_low or layer_width>max_threshold:
                        layer_width=np.random.normal (mu,sigma,1)[0]
                else:
                    #print ('final condition {}'.format(layer))
                    
                    layer_width=sample_size-np.sum(np.array(dialectric_list))-np.sum(scintillator_list)
                if layer%2==starts_with_scintilator:
                    dialectric_list.append(layer_width)
                else:
                    scintillator_list.append(layer_width)
            dialectric_list=np.array(dialectric_list)
            scintillator_list=np.array(scintillator_list)
        elif database_type=='constant':
            scintillator_list=np.ones(nLayersNS//2)*0.0005
            dialectric_list=np.ones(nLayersNS//2)*0.0005
        try:
            scenario=generate_scenario(scintillator_list,dialectric_list,sim_photons,starts_with_scintilator,nLayersNS=nLayersNS)
            
            print(f'run scenario - {i} iteration')
            ph_x,ph_y,ph_z=get_denisty_of_scenario(scenario,filename,database=True,path_pre='G4_Nanophotonic_Scintillator')
            print(ph_x)
            ph_r=np.sqrt(np.power(ph_x,2)+np.power(ph_y,2))
                #ph_r=ph_r[ph_r<=max_rad]
                #ph_z=ph_z[ph_r<=max_rad]
            z_bins=np.linspace(0,sample_size,bins)
            r_bins=np.linspace(0,max_rad,bins)
            binned_layesr=create_boolean_width(z_bins,scintillator_list,dialectric_list,starts_with_scintilator)
            grid,edges_1,edes_2=np.histogram2d(ph_r,ph_z,bins=(r_bins,z_bins))
            #df = df.append({'scintlator':scint,'dialectric':dialectic,'grid':grid}, ignore_index=True)
            saving_dict['scintlator'].append(scintillator_list)
            saving_dict['dialectric'].append(dialectric_list)
            saving_dict['binned'].append(binned_layesr)
            saving_dict['grid'].append(grid)
            print(len(ph_x))
        except:
            print('bad paramters - skippting measurment')
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(saving_dict, f)


    #df=shmoo_width(scintillator_list,dialectric_list,sim_photons)
    #df.to_pickle('periodic_shmoo_width_results.pkl')
import numpy as np
import datetime
import pickle
from multiprocessing import Pool

def process_database_iteration(args):
    """Function to run a single database simulation iteration."""
    i, nLayersNS, sim_photons, wavelength, database_type, bins, max_rad, threshold_low, sample_size = args
    
    starts_with_scintilator = np.random.choice([0, 1], size=1, p=[0.5, 0.5])[0]
    sample_size = np.random.uniform(wavelength * (nLayersNS - 2), wavelength * (nLayersNS + 2), size=1)[0]
    
    scintillator_list, dialectric_list = [], []
    max_threshold = sample_size  # Maximum layer thickness
    
    # Generate layers based on `database_type`
    if database_type == 'uniform':
        low, high = wavelength / 3, 2 * wavelength / 3
        layer_width = np.random.uniform(low, high, size=nLayersNS)
        layer_width = sample_size * layer_width / np.sum(layer_width)
        for layer, ind_layer in zip(layer_width, range(nLayersNS)):
            if ind_layer % 2 == starts_with_scintilator:
                dialectric_list.append(layer)
            else:
                scintillator_list.append(layer)
    elif database_type == 'gaussian':
        sigma, mu = wavelength / 2, wavelength / 2
        for layer in range(nLayersNS):
            if layer < nLayersNS - 1:
                layer_width = np.random.normal(mu, sigma, 1)[0]
                while layer_width < threshold_low or layer_width > max_threshold:
                    layer_width = np.random.normal(mu, sigma, 1)[0]
            else:
                layer_width = sample_size - np.sum(dialectric_list) - np.sum(scintillator_list)
            if layer % 2 == starts_with_scintilator:
                dialectric_list.append(layer_width)
            else:
                scintillator_list.append(layer_width)
    elif database_type == 'constant':
        scintillator_list = np.ones(nLayersNS // 2) * 0.0005
        dialectric_list = np.ones(nLayersNS // 2) * 0.0005

    # Convert lists to NumPy arrays
    dialectric_list = np.array(dialectric_list)
    scintillator_list = np.array(scintillator_list)

    try:
        scenario = generate_scenario(scintillator_list, dialectric_list, sim_photons, starts_with_scintilator, nLayersNS=nLayersNS)
        ph_x, ph_y, ph_z = get_denisty_of_scenario(scenario, "temp", database=True, path_pre='G4_Nanophotonic_Scintillator')

        ph_r = np.sqrt(np.power(ph_x, 2) + np.power(ph_y, 2))
        z_bins = np.linspace(0, sample_size, bins)
        r_bins = np.linspace(0, max_rad, bins)
        
        binned_layers = create_boolean_width(z_bins, scintillator_list, dialectric_list, starts_with_scintilator)
        grid, _, _ = np.histogram2d(ph_r, ph_z, bins=(r_bins, z_bins))
        
        return scintillator_list, dialectric_list, binned_layers, grid

    except:
        print(f'Bad parameters in iteration {i} - Skipping')
        return None  # Return None for bad cases

def run_database_simulation_parallel(nLayersNS=6, data_base_size=5, sim_photons=1000000, wavelength=430e-6, 
                                     database_type='gaussian', bins=100, max_rad=230e-5, 
                                     threshold_low=200e-6, sample_size=430e-5, num_workers=4):
    """Runs database simulation in parallel using Pool"""
    
    # Generate filename
    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + f'_Database_{data_base_size}samples_nonperiodic_sim_{sim_photons}_photons_{nLayersNS}layers_PbOSiO2_varied_size'
    print('Filename:', filename)

    # Create argument list for parallel processing
    args_list = [(i, nLayersNS, sim_photons, wavelength, database_type, bins, max_rad, threshold_low, sample_size) 
                 for i in range(data_base_size)]

    # Run multiprocessing
    with Pool(num_workers) as p:
        results = p.map(process_database_iteration, args_list)

    # Filter out failed cases (None values)
    results = [res for res in results if res is not None]

    # Organize results into saving dictionary
    saving_dict = {'scintlator': [], 'dialectric': [], 'binned': [], 'grid': []}
    for scintillator_list, dialectric_list, binned_layers, grid in results:
        saving_dict['scintlator'].append(scintillator_list)
        saving_dict['dialectric'].append(dialectric_list)
        saving_dict['binned'].append(binned_layers)
        saving_dict['grid'].append(grid)

    # Save to file
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(saving_dict, f)

    print(f"Database saved as {filename}.pkl")


def run_periodic_simulation():
    nLayersNS=2
    database_type='constant'#'gaussian'
    wavelength=430e-9
    N=1
    sim_photons=1000000
    filename='priodic_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers'
    if database_type=='uniform':
        low=wavelength/3
        high=2*wavelength/3
        scintillator_list = np.random.uniform(low, high, N)
        dialectric_list = np.random.uniform(low, high, N)
    elif database_type=='gaussian':
        sigma=wavelength/2
        mu=wavelength/2
        scintillator_list = np.random.random.normal(mu,sigma,N)
        dialectric_list =np.random.random.normal (mu,sigma,N)
    elif database_type=='constant':
        scintillator_list=np.ones(N)*0.0005
        dialectric_list=np.ones(N)*0.0005
    #scintillator_list=np.linspace(2e-4,5e-4,3)
    #dialectric_list=np.linspace(2e-4,5e-4,3)
    z_dim=scintillator_list
    scenario=generate_scenario(z_dim[0],z_dim[0],sim_photons,nLayersNS=nLayersNS)
    get_denisty_of_scenario(scenario,filename)
    #df=shmoo_width(scintillator_list,dialectric_list,sim_photons)
    #df.to_pickle('periodic_shmoo_width_results.pkl')
def run_nonperiodic_simulation():
    nLayersNS=6
    database_type='constant'#'gaussian'
    wavelength=430e-9
    N=1
    sim_photons=100000
    filename='nonpriodic_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers'
    if database_type=='uniform':
        low=wavelength/3
        high=2*wavelength/3
        scintillator_list = np.random.uniform(low, high, nLayersNS)
        dialectric_list = np.random.uniform(low, high, nLayersNS)
    elif database_type=='gaussian':
        sigma=wavelength/2
        mu=wavelength/2
        scintillator_list = np.random.random.normal(mu,sigma,nLayersNS)
        dialectric_list =np.random.random.normal (mu,sigma,nLayersNS)
    elif database_type=='constant':
        scintillator_list=0.005*np.ones(nLayersNS//2)
        dialectric_list=0.002*np.ones(nLayersNS//2)
    #scintillator_list=np.linspace(2e-4,5e-4,3)
    #dialectric_list=np.linspace(2e-4,5e-4,3)
    z_dim=scintillator_list
    scenario=generate_scenario(scintillator_list,dialectric_list,sim_photons,nLayersNS=nLayersNS)
    get_denisty_of_scenario(scenario,filename)
    #df=shmoo_width(scintillator_list,dialectric_list,sim_photons)
    #df.to_pickle('periodic_shmoo_width_results.pkl')
def run_bulk_simulation():
    
    wavelength=430e-9
    N=1
    sim_photons=10000
    filename='bulk_sim_'+str(sim_photons)+'_photons'
    database_type='constant'#'gaussian'
    wavelength=430e-9
    #sim_photons=100
    if database_type=='uniform':
        low=wavelength/3
        high=2*wavelength/3
        z_list = np.random.uniform(low, high, N)
    elif database_type=='gaussian': 
        sigma=wavelength/2
        mu=wavelength/2
        z_list = np.random.random.normal(mu,sigma,N)
    elif database_type=='constant':
        z_list=np.ones(N)*0.002
    z_dim=z_list[0]
    #z_dim=0.01
    scenario=generate_scenario(z_dim,z_dim,sim_photons,scintillator_type= 'bulk')
    #print (scenario)
    get_denisty_of_scenario(scenario,filename)

if __name__ == "__main__":
    global scintillator_type
    #scintillator_type= 'bulk'
    #scintillator_type= 'periodic'
    scintillator_type= 'aperiodic'
    
    data_base=True
    if not data_base:
        if scintillator_type=='periodic':
            run_periodic_simulation()
        elif scintillator_type=='bulk':
            run_bulk_simulation()
        elif scintillator_type=='aperiodic':
            run_nonperiodic_simulation()
    else:
        scintillator_type= 'aperiodic'
        print(os.cpu_count())
        #run_database_simulation(nLayersNS=10,data_base_size=1000,sim_photons=100000,wavelength=430e-6,database_type='uniform',bins=100,max_rad=230e-5,threshold_low=200e-6,sample_size=10*430e-6)
        run_database_simulation_parallel(num_workers=os.cpu_count(),nLayersNS=10,data_base_size=100000,sim_photons=100000,wavelength=430e-6,database_type='uniform',bins=100,max_rad=230e-5,threshold_low=200e-6,sample_size=10*430e-6)  # Adjust workers as needed

