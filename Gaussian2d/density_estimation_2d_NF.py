#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
KDE code

use the truncated gaussian and see the performance, non-local

the simple case one 
'''
import numpy as np
import json
# from generate_sin_data import Sin_distribution
# from generate_gaussian_data_2d_new import GMM_distribution_2d
from scipy.special import legendre
import zuko
import matplotlib.pyplot as plt
import os
import torch
script_name = os.path.basename(__file__)  # Gets the filename of the script
filename, file_extension = os.path.splitext(script_name)  # Separates file name and extension
save_filename = f"{filename}_figure.png"  # Appends '_figure.png' to the script name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from scipy import stats
from scipy.stats import multivariate_normal




dim = 2 
n_data = 20000


data_ref_dict = torch.load('gaussian_truncated_2d_data_20000.pt')
x_data_new = data_ref_dict['x_train_data_new']
y_true_new = data_ref_dict['x_train_pdf_new']






MAF_error = []
MAF_KL = []
MAF_adjust_KL = []

NAF_error = []
NAF_KL = []
NAF_adjust_KL = []

data_store_dict = {}

data_input_dict = torch.load('DE_2d_case2_data_all.pt')
for index in range(20):
    print(index)
    X_train = data_input_dict['x_train_data_'+str(index)]
    pdf_train = data_input_dict['pdf_train_data_'+str(index)]
    X_test = data_input_dict['x_test_data_'+str(index)]
    P_true = data_input_dict['pdf_test_data_'+str(index)]
    
    x_data = X_train
    ###MAF
    run_MAF = True
    if run_MAF == True:
        flow = zuko.flows.MAF(features=2, transforms=5, hidden_features=(128,128,128))
        flow = flow.to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
        
        
        for epoch in range(300):
            losses = []
        

            input_x = torch.tensor(X_train,device=device,dtype=torch.float32)
            loss = -flow().log_prob(input_x).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.detach())
        
            losses = torch.stack(losses)
            if epoch % 10 == 0:
        
                print(f'({epoch})', losses.mean().item())
        
        torch.save(flow.state_dict(), f'{filename}_data_MAF_'+str(index)+'.pth')
        
        device = torch.device("cpu")

        flow = zuko.flows.MAF(features=2, transforms=5, hidden_features=(128,128,128))
        # print(flow)
        flow = flow.to(device)
        ### loading
        model_weights = torch.load(f'{filename}_data_MAF_'+str(index)+'.pth')
        flow.load_state_dict(model_weights)        
        

        x_test_torch = torch.tensor(X_test,device=device)
        log_prob = flow().log_prob(x_test_torch)
        NN_pdf = torch.exp(log_prob)
        NN_pdf = NN_pdf.cpu().detach().numpy()
        
        error = np.linalg.norm(P_true - NN_pdf,2)/np.linalg.norm(P_true,2)
        print('MAF L2 error', error)
        MAF_error.append(error)

        
        x_data_new_train_torch = torch.tensor(x_data_new,device=device,dtype=torch.float32)
        log_prob = flow().log_prob(x_data_new_train_torch)
        NN_pdf = torch.exp(log_prob)
        NN_pdf = NN_pdf.cpu().detach().numpy()

      
        NN_pdf_adjust = []
        pdf_true_adjust = []
        
        for i in range(x_data_new_train_torch.shape[0]):
          if NN_pdf[i] > 1e-7:
            NN_pdf_adjust.append(np.log(NN_pdf[i]))
            pdf_true_adjust.append(np.log(y_true_new[i]))
        log_prob_KL = np.mean(pdf_true_adjust)-np.mean(NN_pdf_adjust)
        print('KL_MAF_adjust', log_prob_KL)
        MAF_adjust_KL.append(log_prob_KL)
        

        data_store_dict['MAF_error'] = MAF_error
        data_store_dict['MAF_adjust_KL'] = MAF_adjust_KL
        torch.save(data_store_dict, 'DE_2d_case2_MAF_result.pt')
        
        
    #### NAF    
        
    flow = zuko.flows.NAF(features=2, transforms=3, hidden_features=(128,128,128))
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    
    
    for epoch in range(300):
        losses = []
    

        input_x = torch.tensor(X_train,device=device,dtype=torch.float32)
        loss = -flow().log_prob(input_x).mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())
    
        losses = torch.stack(losses)
        if epoch % 10 == 0:
    
            print(f'({epoch})', losses.mean().item())
    
    torch.save(flow.state_dict(), f'{filename}_data_NAF_'+str(index)+'.pth')
    
    device = torch.device("cpu")

    flow = zuko.flows.NAF(features=2, transforms=3, hidden_features=(128,128,128))
    # print(flow)
    flow = flow.to(device)
    ### loading
    model_weights = torch.load(f'{filename}_data_NAF_'+str(index)+'.pth')
    flow.load_state_dict(model_weights)        
    

    x_test_torch = torch.tensor(X_test,device=device)
    log_prob = flow().log_prob(x_test_torch)
    NN_pdf = torch.exp(log_prob)
    NN_pdf = NN_pdf.cpu().detach().numpy()
    
    error = np.linalg.norm(P_true - NN_pdf,2)/np.linalg.norm(P_true,2)
    print('NAF L2 error', error)
    NAF_error.append(error)

    
    x_data_new_train_torch = torch.tensor(x_data_new,device=device,dtype=torch.float32)
    log_prob = flow().log_prob(x_data_new_train_torch)
    NN_pdf = torch.exp(log_prob)
    NN_pdf = NN_pdf.cpu().detach().numpy()

    
    NN_pdf_adjust = []
    pdf_true_adjust = []
    
    for i in range(x_data_new_train_torch.shape[0]):
      if NN_pdf[i] > 1e-7:
        NN_pdf_adjust.append(np.log(NN_pdf[i]))
        pdf_true_adjust.append(np.log(y_true_new[i]))
    log_prob_KL = np.mean(pdf_true_adjust)-np.mean(NN_pdf_adjust)
    print('KL_MAF_adjust', log_prob_KL)
    NAF_adjust_KL.append(log_prob_KL)
    

    data_store_dict['NAF_error'] = NAF_error
    data_store_dict['NAF_adjust_KL'] = NAF_adjust_KL
    torch.save(data_store_dict, 'DE_2d_case2_NAF_result.pt')
    


