# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:16:03 2024

@author: nathan
"""
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
sys.path.append('/home/nathan.regev/software/git_repo/')
from ScintCityPython.NN_models import Model1, Model2,ScintProcessStochasticModel,ScintProcessStochasticModel_no_CNN

"""
class Model1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double() 
        self.fc2 = nn.Linear(hidden_size, output_size).double() 
        self.activation = nn.ReLU().double()
        self.grid_dim=input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(self.grid_dim, self.grid_dim)
        # conv layers....

        return x
class Model2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double() 
        self.fc2 = nn.Linear(hidden_size, output_size).double()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5,padding=2).double()
        #self.conv1.bias.data = self.conv1.bias.data.double()
        self.activation = nn.ReLU().double()
        self.grid_dim=input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(-1, 1, self.grid_dim, self.grid_dim)
        x = self.conv1(x)
        # conv layers....
        #x = self.activation(x)
        
        

        return x
    """
#define chi_square error:
class ChiSquareLoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super(ChiSquareLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        input = F.normalize(input, p=1, dim=1)  # Normalize to make it a probability distribution
        target = F.normalize(target, p=1, dim=1)
        chi_sq = torch.sum((input - target) ** 2 / (target + self.epsilon), dim=1)
        return torch.nanmean(chi_sq)
class JSDivergenceLoss(nn.Module):
    def __init__(self, epsilon=1e-10):
        super(JSDivergenceLoss, self).__init__()
        self.epsilon = epsilon  # Small constant to avoid log(0)

    def forward(self, p, q):
        # Ensure the inputs are valid probability distributions (normalized)
        p = F.normalize(p, p=1, dim=1)  # Normalize over features
        q = F.normalize(q, p=1, dim=1)

        # Compute the average distribution
        m = 0.5 * (p + q)

        # Compute KL divergence between p and m, q and m
        kl_p_m = torch.sum(p * torch.log(p / (m + self.epsilon) + self.epsilon), dim=1)
        kl_q_m = torch.sum(q * torch.log(q / (m + self.epsilon) + self.epsilon), dim=1)

        # Jensen-Shannon Divergence
        jsd = 0.5 * (kl_p_m + kl_q_m)
        return jsd.mean()  # Return the mean JS divergence across the batch
# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.double), torch.tensor(self.labels[idx], dtype=torch.double)

#params

database_cutoff=0.9 
num_epochs=30

#load dataset
folder_path=r"/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/"
#data_name="20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers.pkl"
#data_name="20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2.pkl"
#data_name='20250209125901955984Database_100000samples_nonpriodic_sim_10000_photons_10layers_PbOSiO2.pkl'
data_name="/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
mod='4'
current_date = datetime.now()
databse_folder=os.path.join(folder_path, data_name.strip('.pkl')+'_model'+str(mod)+'results')
if not os.path.exists(databse_folder):
    os.mkdir(databse_folder)
train_result_path=os.path.join(databse_folder,current_date.strftime('%y%m%d%H%M%S'))
os.mkdir(train_result_path)
with open(os.path.join(folder_path,data_name), 'rb') as f:
    datadict= pickle.load(f)
wieghts_path=os.path.join(train_result_path,'results_wieghts.pth')
# for i in range(5):
#     datadict['binned'][i]=datadict['binned'][i][:-1]    
batch_size=len(datadict['binned'])
X_train=datadict['binned'][:int(database_cutoff*batch_size)]
sample_size_train=[np.sum(np.array(scint))+np.sum(np.array(dial)) for scint,dial in zip(datadict['scintlator'][:int(database_cutoff*batch_size)],datadict['dialectric'][:int(database_cutoff*batch_size)])]
X_val=datadict['binned'][int(database_cutoff*batch_size):]
y_train=datadict['grid'][:int(database_cutoff*batch_size)]
y_val=datadict['grid'][int(database_cutoff*batch_size):]
sample_size_val=[np.sum(np.array(scint))+np.sum(np.array(dial)) for scint,dial in zip(datadict['scintlator'][int(database_cutoff*batch_size):],datadict['dialectric'][int(database_cutoff*batch_size):])]
"""
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)    
"""
# For trainning
if mod=='1':
    model = Model1( datadict['binned'][0].size, datadict['binned'][0].size, datadict['grid'][0].size)
elif mod=='2':
    model = Model2( datadict['binned'][0].size, datadict['binned'][0].size, datadict['grid'][0].size)
elif mod=='3':
    model = ScintProcessStochasticModel( datadict['binned'][0].size, datadict['binned'][0].size, datadict['grid'][0].size)
elif mod=='4':
    model = ScintProcessStochasticModel_no_CNN( datadict['binned'][0].size, datadict['binned'][0].size, datadict['grid'][0].size)


# Get a single prediction
thicknesses = torch.tensor(datadict['binned'][0]).double() # input thicknesses
#cloud_of_events = model(thicknesses)


# Create a dummy dataset
#inputs = torch.randn(200, 99)  
#targets = torch.randn(200, 99, 99)  

# Define a loss function and an optimizer
criterion = nn.MSELoss() # nn.CrossEntropyLoss
#criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#criterion = JSDivergenceLoss()
#criterion=ChiSquareLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
# Train the network

model_loss_arr=[]
no_material_loss_arr=[]
dialectric_loss_arr=[]
cross_cycle_loss_arr=[]
scintilator_loss_arr=[]
accuracy_arr=[]
print(wieghts_path)
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Move model to the device
model = model.to(device)
model = model.double()  # Ensure model uses double precision if needed

# Training loop
for epoch in range(num_epochs):
    model.train()
    #for inputs, labels in zip(X_train, y_train):
    for index in range (len(X_train)):
        inputs=X_train[index]
        labels=y_train[index]
        cell_size=sample_size_train[index]/100
        # Move inputs and labels to the devicev
        inputs = torch.tensor(inputs).double().to(device)
        labels = torch.tensor(labels).double().to(device)
        cell_size = torch.tensor(cell_size).double().to(device)
        # Create tensors on the same device
        zeros_tensor = -torch.ones_like(inputs, device=device)
        dialectric_only_tensor = torch.zeros_like(inputs, device=device)
        scint_only_tensor = torch.ones_like(inputs, device=device)
        indices = torch.randperm(inputs.size(0))[:10]
        corss_cycle_tensor=inputs.clone()
        corss_cycle_tensor[indices]=1-torch.abs(corss_cycle_tensor[indices]) #flips value of tensors
        # Zero the parameter gradients
        optimizer.zero_grad()
        #add dropout mechanism
        # Forward pass
        cloud_of_events = model(inputs,cell_size=cell_size)
        #zero_emission = model(zeros_tensor,cell_size=cell_size)
        dialectric_only_emission = model(dialectric_only_tensor,cell_size=cell_size)
        #scint_only_emission = model(scint_only_tensor,cell_size=cell_size)
        cross_cycle_emission=model(corss_cycle_tensor,cell_size=cell_size)
        # Compute errors
        #no_material_error = torch.sum(zero_emission**2)
        dialectric_error = torch.sum(dialectric_only_emission**2)
        cross_cycle_error=criterion(cloud_of_events, cross_cycle_emission)**2
        #scintilator_error = torch.abs(torch.sum(scint_only_emission**2) - 700**2)
        #std_dev = torch.mean(torch.std(scint_only_emission, dim=0))+torch.mean(torch.std(dialectric_only_emission, dim=0))
        if mod =='2' or mod =='3':
            model_error = criterion(cloud_of_events[0,0,:,:], labels)
        else:
            model_error = criterion(cloud_of_events, labels)
        
        # Total loss
        #loss = model_error + no_material_error + dialectric_error+cross_cycle_error+std_dev
        loss = model_error #+ dialectric_error + cross_cycle_error
        # Logging
        print(f'Loss function is {loss}')
        print(f'Yield for diaelectric is {dialectric_error}')
        """
        print(f'no material erro is {no_material_error}')
        #print(f'nulk scint error is {scintilator_error}')
        print(f'cross cycle error is {cross_cycle_error}')
        """
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    # Validation loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        #loss_arr_it = []
        #accuracy_arr_it = []
        model_loss_arr_it=[]
        no_material_loss_arr_it=[]
        dialectric_loss_arr_it=[]
        cross_cycle_loss_arr_it=[]
        scintilator_loss_arr_it=[]
        accuracy_arr=[]
        for index in range (len(X_val)):
            inputs=X_train[index]
            labels=y_train[index]
            cell_size=sample_size_train[index]/100
            # Move inputs and labels to the devicev
            inputs = torch.tensor(inputs).double().to(device)
            labels = torch.tensor(labels).double().to(device)
            cell_size = torch.tensor(cell_size).double().to(device)
            # Create tensors on the same device
            zeros_tensor = -torch.ones_like(inputs, device=device)
            dialectric_only_tensor = torch.zeros_like(inputs, device=device)
            scint_only_tensor = torch.ones_like(inputs, device=device)
            indices = torch.randperm(inputs.size(0))[:10]
            corss_cycle_tensor=inputs.clone()
            corss_cycle_tensor[indices]=1-torch.abs(corss_cycle_tensor[indices]) #flips value of tensors
            # Zero the parameter gradients
            optimizer.zero_grad()
            #add dropout mechanism
            # Forward pass
            cloud_of_events = model(inputs,cell_size=cell_size)
            zero_emission = model(zeros_tensor,cell_size=cell_size)
            dialectric_only_emission = model(dialectric_only_tensor,cell_size=cell_size)
            scint_only_emission = model(scint_only_tensor,cell_size=cell_size)
            cross_cycle_emission=model(corss_cycle_tensor,cell_size=cell_size)
            # Compute errors
            no_material_error = torch.sum(zero_emission**2)
            dialectric_error = torch.sum(dialectric_only_emission**2)
            cross_cycle_error=criterion(cloud_of_events, cross_cycle_emission)**2
            scintilator_error = torch.abs(torch.sum(scint_only_emission**2) - 700**2)
            std_dev = torch.mean(torch.std(scint_only_emission, dim=0))+torch.mean(torch.std(dialectric_only_emission, dim=0))
            if mod =='2' or mod =='3':
                model_error = criterion(cloud_of_events[0,0,:,:], labels)
            else:
                model_error = criterion(cloud_of_events, labels)
            
            # Total loss
            loss = model_error + no_material_error + dialectric_error+cross_cycle_error+std_dev
            model_loss_arr_it.append(model_error.cpu().item())
            no_material_loss_arr_it.append(no_material_error.cpu().item())
            dialectric_loss_arr_it.append(dialectric_error.cpu().item())
            cross_cycle_loss_arr_it.append(cross_cycle_error.cpu().item())
            scintilator_loss_arr_it.append(std_dev.cpu().item())
        # Log validation results
        model_loss_arr.append(np.mean(model_loss_arr_it))
        no_material_loss_arr.append(np.mean(no_material_loss_arr_it))
        dialectric_loss_arr.append(np.mean(dialectric_loss_arr_it))
        cross_cycle_loss_arr.append(np.mean(cross_cycle_loss_arr_it))
        scintilator_loss_arr.append(np.mean(scintilator_loss_arr))

    print(f'Epoch [{epoch+1}/{num_epochs}]')

#%%
print('Training Finished')

torch.save(model.state_dict(), wieghts_path)
print('folder paths is '+train_result_path)
plt.figure()
plt.plot(np.array(model_loss_arr),'.',label ='model loss')
plt.plot(np.array(no_material_loss_arr),'.',label ='blank loss')
plt.plot(np.array(dialectric_loss_arr),'.',label ='dialectric loss')
plt.plot(np.array(cross_cycle_loss_arr),'.',label ='cross_cycle_loss_arr')

plt.title('loss function vs iterations')
plt.ylabel('MMSE')
plt.xlabel('interation #')
plt.savefig(os.path.join(train_result_path,'MSSE_loss.png'))


i=len(y_val)//2 #+Database_100samples_nonpriodic_sim_100000_photons_6layers
plt.figure()
plt.subplot(1,3,1)
u=X_val[i]==0
plt.plot(u,'.')
plt.title('layer structure')
plt.subplot(1,3,2)
plt.imshow(y_val[i],cmap='gray')
plt.title('measured photon emssion map')
plt.xlabel('zaxis bins')
plt.ylabel('radial bins')

plt.subplot(1,3,3)
X_val_tensor = torch.tensor(X_val[i], dtype=torch.double, device=device)  # Move tensor to the same device
cell_size=sample_size_val[i]/100
cell_size=torch.tensor(cell_size).double().to(device)
# Pass the input through the model
res = model(X_val_tensor,cell_size=cell_size)
if mod =='2'or mod =='3':
    res=res[0,0,:,:]
res_cpu = res.detach().cpu().numpy()
# Plot the result
plt.imshow(res_cpu, cmap='gray')
plt.title('Model 1 Result')
plt.xlabel('z-axis bins')
plt.ylabel('radial bins')
# Save the result

plt.savefig(os.path.join(train_result_path, 'prediction_visualisation.png'))
data = {
        'model error':model_loss_arr,
        'no_material_loss_arr':no_material_loss_arr,
        'dialectric_loss_arr':dialectric_loss_arr,
        'cross_cycle_loss_arr':cross_cycle_loss_arr,
        'scintilator_loss_arr':scintilator_loss_arr
}

# Save the dictionary as a pickle file
with open(os.path.join(train_result_path,"trainig loss arrays.pkl"), "wb") as file:
    pickle.dump(data, file)

print("Dictionary saved as pickle file successfully.")
#add comparison between netwrok 