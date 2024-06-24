
import numpy as np
import json
from scipy.special import legendre

import matplotlib.pyplot as plt
import os
import torch
script_name = os.path.basename(__file__)  # Gets the filename of the script
filename, file_extension = os.path.splitext(script_name)  # Separates file name and extension
save_filename = f"{filename}_figure.png"  # Appends '_figure.png' to the script name


from scipy import stats

class kernel_density():
    def __init__(self,dim,data):
        """ Initialize the KernelDensity class. Currently, the initializer does not do anything. """
        values = [[] for _ in range(dim)]
        for i in range(len(data)):
            for d in range(dim):
                values[d].append(data[i][d])

        # Create a Gaussian Kernel Density Estimator using the input data
        kernel = stats.gaussian_kde(values)
        self.kernel = kernel
        self.dim = dim

    def compute(self, X_new):
        """
        Computes the kernel density estimation.

        Parameters:
        dim (int): The number of dimensions in the data.
        data (list of lists): The input data for which the kernel density is to be computed. 
                              Each inner list represents a point in 'dim' dimensional space.
        X_new (list of lists): The new data points where the density is to be estimated.

        Returns:
        ndarray: The estimated density values for each point in X_new.
        """

        # Transpose X_new to fit the format expected by the kernel
        X_test = [[] for _ in range(self.dim)]
        for i in range(len(X_new)):
            for d in range(dim):
                X_test[d].append(X_new[i][d])

        # Return the estimated density values
        return self.kernel(X_test)
    
    def pdf(self, X_new):
        """
        Computes the kernel density estimation.

        Parameters:
        dim (int): The number of dimensions in the data.
        data (list of lists): The input data for which the kernel density is to be computed. 
                              Each inner list represents a point in 'dim' dimensional space.
        X_new (list of lists): The new data points where the density is to be estimated.

        Returns:
        ndarray: The estimated density values for each point in X_new.
        """

        # Transpose X_new to fit the format expected by the kernel
        X_test = [[] for _ in range(self.dim)]
        for i in range(len(X_new)):
            for d in range(dim):
                X_test[d].append(X_new[i][d])

        # Return the estimated density values
        return self.kernel.pdf(X_test) 
    
    def generate(self, sample_size):
        return self.kernel.resample(size=sample_size)
    



data_dict = {}
VRS_error_list = []

dim = 10 


data_input_dict = torch.load('GL_data_new_dim10.pt')

data_store_dict = {}
err_list = []
KL_list = []
KL_adjust_list = []
for n_data in [100000,200000,300000,400000,500000]:


    X_train = data_input_dict['x_train_data'][:n_data,:]
    right_domain = data_input_dict['x_boundary'] 
    X_test = data_input_dict['x_test_data'] 
    pdf_test = data_input_dict['pdf_test_data']
    
    x_data = X_train
    
    # X_train = x_data 
    print('n_data', n_data)
    print('dimension', dim)
    
    
    KDE = kernel_density(dim, X_train)
        
   
    '''
    test
    '''  
    y_new_test = np.array(KDE.pdf(X_test))   
    y_new_test = y_new_test.reshape(-1,1)
    pdf_test  = pdf_test.reshape(-1,1)
    
    err = np.linalg.norm(y_new_test - pdf_test,2)/np.linalg.norm(pdf_test,2)
    print('err', err)
    err_list.append(err)

      
    
    data_input_dict_new_train = torch.load('GL_data_additional_old_dim10.pt')
    x_data_new_train = data_input_dict_new_train['x_train_data']
    P_true_new_train = data_input_dict_new_train['pdf_train_data'] 
    
    
    y_new_train = np.array(KDE.pdf(x_data_new_train))

    
    y_new_train_adjust = []
    P_true_new_train_adjust = []
    for i in range(len(x_data_new_train)):
        if y_new_train[i] > 1e-7:
            y_new_train_adjust.append(np.log(y_new_train[i]))
            P_true_new_train_adjust.append(np.log(P_true_new_train[i]))
    
    KDE_KL_adjust = np.mean(P_true_new_train_adjust) - np.mean(y_new_train_adjust)
    print('Ptrue_adjust',np.mean(P_true_new_train_adjust))        
    print('KDE_KL_adjust', KDE_KL_adjust)
    KL_adjust_list.append(KDE_KL_adjust)
    
    data_store_dict['err_list'] = err_list
    data_store_dict['KL_adjust_list'] = KL_adjust_list
    torch.save(data_store_dict, f'{filename}_result.pt')











