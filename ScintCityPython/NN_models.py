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

class StochasticLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(StochasticLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for the mean and standard deviation of the weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_sigma_param = nn.Parameter(torch.randn(out_features, in_features) * -2.0)  # Log-space sigma
        
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma_param = nn.Parameter(torch.ones(out_features) * -2.0)  # Log-space sigma

    def forward(self, x,cell_size):
        # Ensure sigma is positive using softplus (sigma = log(1 + exp(param)))
        weight_sigma = F.softplus(self.weight_sigma_param)
        bias_sigma = F.softplus(self.bias_sigma_param)

        # Sample weights and biases from a Gaussian distribution
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = (self.bias_mu + bias_sigma * torch.randn_like(bias_sigma))*cell_size

        return F.linear(x, weight, bias)
class StochasticConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(StochasticConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Mean (\mu) and log-variance (\log\sigma^2)
        self.weight_mu = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * -2.0)
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_log_sigma = nn.Parameter(torch.ones(out_channels) * -2.0)
    
    def forward(self, x):
        # Sample weights and biases from the Gaussian distribution
        weight_sigma = torch.exp(0.5 * self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = torch.exp(0.5 * self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.conv2d(x, weight, bias, padding=self.padding)

class ScintProcessStochasticModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScintProcessStochasticModel, self).__init__()
        self.fc1 = StochasticLinear(input_size, hidden_size).double()
        self.fc2 = StochasticLinear(hidden_size, output_size).double()
        self.conv1 = StochasticConv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2).double()
        self.activation = nn.ReLU().double()
        self.grid_dim = input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(-1, 1, self.grid_dim, self.grid_dim)
        x = self.conv1(x)
        return x

class ScintProcessStochasticModel_no_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScintProcessStochasticModel_no_CNN, self).__init__()
        self.fc1 = StochasticLinear(input_size, hidden_size).double()
        self.fc2 = StochasticLinear(hidden_size, output_size).double()
        self.activation = nn.ReLU().double()
        #replace with GeLU?
        self.grid_dim = input_size

    def forward(self, x,cell_size=torch.Tensor(1)):
        x = self.fc1(x,cell_size)
        x = self.activation(x)
        x = self.fc2(x,cell_size)
        x = x.view(self.grid_dim, self.grid_dim)
        return x
        """
class ScintProcessStochasticModel_no_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.1):
        super(ScintProcessStochasticModel_no_CNN, self).__init__()
        self.fc1 = StochasticLinear(input_size, hidden_size).double()
        self.fc2 = StochasticLinear(hidden_size, output_size).double()
        self.activation = nn.ReLU().double()
        #self.dropout_prob = dropout_prob
        self.grid_dim = input_size

    def forward(self, x):
        x = self.fc1(x)
        # Apply dropout only during training
        #x = F.dropout(x, p=self.dropout_prob, training=self.training) 
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(self.grid_dim, self.grid_dim)
        return x
"""