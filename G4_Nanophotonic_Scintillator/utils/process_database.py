import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import uproot
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
def generte_grid_denisty(photons_position,axis_res,axis_dim,substrate=0,normalisation=1):
    #x_vec=np.arange(-axis_dim[0],axis_dim[0],axis_res[0])
    #y_vec=np.arange(-axis_dim[1],axis_dim[1],axis_res[1])
    #z_vec=np.arange(0,axis_dim[2],axis_res[2])
    x_vec=np.arange(-0.1,0.1,axis_res[0])
    y_vec=np.arange(-0.1,0.1,axis_res[1])
    z_vec=np.arange(substrate,axis_dim[2],axis_res[2])
    density_map=np.zeros((len(x_vec),len(y_vec),len(z_vec)))
    radius_vec = [[] for _ in range(len(z_vec))]
    valid_x=photons_position[0][photons_position[2]>substrate]
    valid_y=photons_position[1][photons_position[2]>substrate]
    valid_z=photons_position[2][photons_position[2]>substrate]
    for photonx,photony,photonz in zip(valid_x,valid_y,valid_z):
        #maybe need to add a verification that photon indexes are in correct location
        """
        x_index = np.abs(x_vec - photonx).argmin()
        y_index = np.abs(y_vec - photony).argmin()
        z_index = np.abs(z_vec - photonz).argmin()
        """
        x_index=int((photonx+axis_dim[0])/(axis_res[0]))# because x values are also negative, need to move list to positive part
        y_index=int((photony+axis_dim[1])/(axis_res[1]))
        z_index=int((photonz-substrate)/axis_res[2])-1
        #print('x: {}, y :{} , z :{}'.format(x_index,y_index,z_index))
        density_map[x_index,y_index,z_index]+=1
        radius_vec[z_index].append(np.sqrt(photonx**2+photony**2))
    return density_map/normalisation,radius_vec,z_vec
def generate3dmodel(X_coords,Y_coords,Z_coords,axis_dim,axis_res):
    photons_position=[X_coords,Y_coords,Z_coords]
    #print (photons_position)
    x_vec=np.arange(-axis_dim[0],axis_dim[0],axis_res[0])
    y_vec=np.arange(-axis_dim[1],axis_dim[1],axis_res[1])
    z_vec=np.arange(0,axis_dim[2],axis_res[2])
    map=generte_grid_denisty(photons_position,axis_res,axis_dim,normalisation=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each 2D slice along the third dimension
    #ax.imshow(map, extent=[0, axis_dim[0], 0,axis_dim[1],0,axis_dim[2]],origin='lower', alpha=0.5)
    """
    for i in range(map.shape[2]):
        x, y= np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]))
        ax.imshow(map[:, :, i], extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', alpha=0.5)
    """
    intensity=0.5*map/np.max(map)
    coord_X_coords=X_coords[(X_coords<axis_dim[0])*( Y_coords<axis_dim[1])]
    coord_Y_coords=Y_coords[(X_coords<axis_dim[0])*( Y_coords<axis_dim[1])]
    coord_Z_coords=Z_coords[(X_coords<axis_dim[0])*( Y_coords<axis_dim[1])]
    ax.scatter(coord_X_coords,coord_Y_coords,coord_Z_coords)
    for i,x in zip(range(len(x_vec)),x_vec):
        for j,y in zip(range(len(y_vec)),y_vec):
            for k,z in zip(range(len(z_vec)),z_vec):
                ax.bar3d(x, y, z,x_vec[1]-x_vec[0], y_vec[1]-y_vec[0],z_vec[1]-z_vec[0], shade=True, alpha=intensity[i,j,k], color='blue')
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('3D Imshow-style Plot')
    plt.show()
def proccess_database(raw_results):
    print (raw_results)
    xcoords=[]
    ycoords=[]
    zcoords=[]
    for i in range (N):
        #print (raw_results['X_coords'][i].shape)
        print(np.max(raw_results["Z_coords"][i]))
        xcoords=raw_results["X_coords"][i]
        ycoords=raw_results["Y_coords"][i]
        zcoords=raw_results["Z_coords"][i]
    photons_position=[xcoords,ycoords,zcoords]
    axis_dim=[0,0,10]
    axis_res=[1e-2,1e-2,1e-1]
    map,radius_vec,zvec=generte_grid_denisty(photons_position,axis_res,axis_dim)
    return map,radius_vec,zvec
def photon_model_function(z,z_0,rho_0):
    L_bulk=np.max(z)
    z=z-L_bulk/2
    z_0=np.abs(z_0)
    rho_0=np.abs(rho_0)
    return rho_0*z_0*(-2*np.exp(-L_bulk/(2*z_0))*np.cosh(z/z_0)+2)
def photon_model_nonsymetrical(z,z_0,z_1,rho_0):
    L_bulk=np.max(z)
    z=z-L_bulk/2
    z_0=np.abs(z_0)
    z_1=np.abs(z_1)
    rho_0=np.abs(rho_0)
    return rho_0*(z_0+z_1-(z_0*np.exp((-z-L_bulk/2)/z_0)+z_1*np.exp((z-L_bulk/2)/z_1)))
def electron_model_symetrical(z,z_0,rho_0,ref_coef):
    L_bulk=np.max(z)
    z=z-L_bulk/2
    z_0=np.abs(z_0)
    rho_0=np.abs(rho_0)
    return rho_0*z_0*((-2*np.exp(-L_bulk/(2*z_0))+4*ref_coef*np.exp(-L_bulk/(z_0))*np.cosh(L_bulk/(2*z_0)))*np.cosh(z/z_0)+2)
if __name__ == "__main__":
    N=1
    meas_type='bulk'
    path='/home/nathan_regev/software/git_repo/G4_Nanophotonic_Scintillator'
    #root_file = uproot.open('bulk_sim_10000000_photons20240302162231561864.root')
    root_file = uproot.open('nonpriodic_sim_1000000_photons_6layers20240307145343303255.root')
    
    #root_file = uproot.open('priodic_sim_1000000_photons_3layers.root')
    #root_file = uproot.open('short_root_file_1e5_photons.root')
    photons = root_file["Photons"]
    filter_field="fCreatorProcess"
    #filter_field="fSubProcessType"
    print(photons.keys())
    ##print(np.array(photons).shape)
    prop = [t for t in photons['fType'].array()]
    #print(prop)
    relevant_ind = [t == 'lepton' for t in prop]
    relevant_ind_gamma=[t == 'gamma' for t in prop]
    relevant_ind_photons = [t == 'opticalphoton' for t in prop]
    #print(np.sum(relevant_ind))
    #print(processes)
    dict_proc={}
    #for proc in processes:
    processType = [t for t in photons["fProcess"].array()]
    creatorProcess = [t for t in photons["fCreatorProcess"].array()]
    stepIndex=[t for t in photons["StepIndex"].array()]
    generated_electron_filter=[proces=='none' and creatorProc == 'phot' and ind==1 for ((proces,creatorProc),ind) in zip(zip(processType, creatorProcess),stepIndex)]
    #all_electron_filter=[proces=='none' and creatorProc == 'phot' for (proces,creatorProc) in zip(processType, creatorProcess)]
    all_electron_filter=[proces=='none' and creatorProc == 'phot' for (proces,creatorProc) in zip(processType, creatorProcess)]
    #generated_electron_filter=all_electron_filter
    print('first step electrons - {}'.format(np.sum(generated_electron_filter)))
    print('all steps electrons - {}'.format(np.sum(all_electron_filter)))
    lifetime= np.array(photons["StepIndex"].array())[all_electron_filter]
    generated_electronsZ= np.array(photons["fZ"].array())[generated_electron_filter]
    all_electronsZ= np.array(photons["fZ"].array())[all_electron_filter]
    opticalphotonsZ = np.array(photons["fZ"].array())[relevant_ind_photons]
    gamma_Z=np.array(photons["fZ"].array())[relevant_ind_gamma]
    photonsT=np.array(photons["fT"].array())[generated_electron_filter]
    photonsWLEN=np.array(photons["fWlen"].array())[generated_electron_filter]
    generated_electrons_energy=np.array(photons["Energy"].array())[generated_electron_filter]
    #z_bins=np.linspace(0,np.max(photonsZ),30)

    med=np.median(photonsT)
    std=np.std(photonsT)
    time=1e-4
    #bin_num=30
    bin_num=100
    plt.figure()
    data,z_e,patch=plt.hist(generated_electronsZ,bins=bin_num)
    z_e=z_e[:-1]
    plt.title('generated electron distribution')
    plt.xlabel('z coordinate [mm]')
    indeces_list_generated_electronsZ=np.digitize(generated_electronsZ,bins=z_e)#this array contain hold information over each electron's bin
    
    binned_energies=[[] for i in range(bin_num)]
    for E,ind in zip(generated_electrons_energy,indeces_list_generated_electronsZ):
        binned_energies[ind-1].append(E)
    avg_energy=np.zeros(bin_num)
    for i,bin_en in zip(range(len(binned_energies)),binned_energies):
        avg_energy[i]=np.mean(bin_en)
    plt.figure()
    plt.title('average energy of generated electrons in each bin ')
    plt.plot(z_e,avg_energy,'.')
    plt.xlabel('distance [mm]')
    plt.ylabel('energy [eV]')
    plt.figure()
    real_data_e,z_e,patch=plt.hist(all_electronsZ,bins=bin_num)
    z_e=z_e[:-1]
    try:
        popt_e_sym,pcov=curve_fit(electron_model_symetrical,z_e,real_data_e,p0=[0.0001,500,1])
        plt.plot(z_e,electron_model_symetrical(z_e,*popt_e_sym),label='symmetrical')
        print('symetrical electron coeficeints'+str(popt_e_sym))
    except:
        print('could not fit data')
    plt.title('all electron distribution')
    plt.xlabel('z coordinate [mm]')
    plt.legend()
   

    plt.figure()
    n,bins,patch=plt.hist(lifetime,bins=bin_num)
    plt.axhline(y=np.max(n)/np.e, color='red', linestyle='--', label='Horizontal Line at y=0.5')
    plt.title('electrons lifetime distribution')
    plt.xlabel('step')
    plt.figure()
    real_data,z_ph,patch=plt.hist(opticalphotonsZ,bins=bin_num)
    z_ph=z_ph[:-1]
    try:
        popt_sym,pcov=curve_fit(photon_model_function,z_ph,real_data,p0=[0.0001,500])
        popt_non_sym,pcov=curve_fit(photon_model_nonsymetrical,z_ph,real_data,p0=[0.0001,0.0001,500])
        plt.plot(z_ph,photon_model_function(z_ph,*popt_sym),label='symmetrical')
        plt.plot(z_ph,photon_model_nonsymetrical(z_ph,*popt_non_sym),label='non symmetrical')
        print('symetrical coeficeints'+str(popt_sym))
        print('non symetrical coeficeints'+str(popt_non_sym))
    except:
        print('could not fit data')
    plt.title('photons distribution')
    plt.xlabel('z coordinate [mm]')
    plt.legend()



    plt.figure()
    plt.hist(gamma_Z,bins=bin_num)
    plt.title('gamma distribution')
    plt.xlabel('z coordinate [mm]')
    
   
    plt.show()

    
