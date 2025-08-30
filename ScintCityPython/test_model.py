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
from scipy.ndimage import median_filter
import os
from skimage.metrics import mean_squared_error
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
def MSE_ERR(arr1,arr2):
    return np.sum(np.power(arr1-arr2,2))/np.sum(np.power(arr1-np.mean(arr1),2))
def load_parameters(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
database_cutoff=0.8
num_epochs=500

#load dataset
mod='1'
#databse_path=r"/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers.pkl"

databse_path=r"/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2.pkl"
#databse_path=r"/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20240417132150509930Database_10samples_nonpriodic_sim_10000000_photons_6layers20240417132150516048.root"
weights_path=r"/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model1results/240416093038/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model1_parameters.pth"

with open(databse_path, 'rb') as f:
    datadict= pickle.load(f)
# for i in range(5):
#     datadict['binned'][i]=datadict['binned'][i][:-1]    
batch_size=len(datadict['binned'])
X_val=datadict['binned']
y_val=datadict['grid']
val_dataset = CustomDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)    
# For trainning
if mod=='1':
    model = Model1( datadict['binned'][0].size, datadict['binned'][0].size, datadict['grid'][0].size)
elif mod=='2':
    model = Model2( datadict['binned'][0].size, datadict['binned'][0].size, datadict['grid'][0].size)


# Step 4: Run Inference


# Load the parameters
load_parameters(model, weights_path)

# Predictions
i=len(y_val)//2 #+Database_100samples_nonpriodic_sim_100000_photons_6layers
plt.figure()
plt.suptitle('training results')
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
res = model(torch.tensor(X_val[i]).double())
if mod =='2':
    res=res[0,0,:,:]
res_im=res.detach().numpy()
label_im=y_val[i]
loss = mean_squared_error(label_im,res_im)
plt.imshow(res.detach().numpy(),cmap='gray')
plt.title('model 1 result')
plt.xlabel('zaxis bins')
plt.ylabel('radial bins')
plt.text(5, -1.5, f'loss function is {loss}', ha='center', transform=plt.gca().transAxes)
print(f'loss function is {loss}')
plt.savefig('model accuracy.png')
plt.savefig('model accuracy.svg')
filtered_res=median_filter(res.detach().numpy(), size=3)
filtered_image=median_filter(y_val[i], size=3)
# Calculate the dimensions of the downsampled array
downsampled_shape = (res_im.shape[0] // 4, res_im.shape[1] // 4)

# Create the downsampled array
downsampled_label = np.zeros(downsampled_shape)
downsampled_res = np.zeros(downsampled_shape)
# Loop through the original array and sum every 4x4 block
for i in range(downsampled_shape[0]):
    for j in range(downsampled_shape[1]):
        downsampled_label[i, j] = np.sum(label_im[i*4:(i+1)*4, j*4:(j+1)*4])
        downsampled_res[i, j] = np.sum(res_im[i*4:(i+1)*4, j*4:(j+1)*4])
loss = mean_squared_error(filtered_image,filtered_res)
plt.figure()
plt.suptitle('median filter')
plt.subplot(1,2,1)
plt.imshow(filtered_image,cmap='gray')
plt.title('filtered simulation')
plt.subplot(1,2,2)
plt.imshow(filtered_res,cmap='gray')
plt.title('filtered result')
plt.text(5, -1.5, f'loss function is {loss}', ha='center', transform=plt.gca().transAxes)
print(f'loss function is {loss}')
loss = mean_squared_error(downsampled_res,downsampled_label)
plt.figure()

plt.subplot(1,2,1)
plt.suptitle('lower resolution')

plt.imshow(downsampled_label,cmap='gray')
plt.title('squeesed simulation')
plt.subplot(1,2,2)
plt.imshow(downsampled_res,cmap='gray')
plt.title('squeesed result')
plt.text(5, -1.5, f'loss function is {loss}' ,fontsize=12, ha='center', va='center',color='red', transform=plt.gca().transAxes)
print(f'loss function is {loss}')
plt.show()
