
import torch
import numpy as np
import zuko

import matplotlib.pyplot as plt
import os
script_name = os.path.basename(__file__)  # Gets the filename of the script
filename, file_extension = os.path.splitext(script_name)  # Separates file name and extension
save_filename = f"{filename}_figure.png"  # Appends '_figure.png' to the script name


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




n_data = 100000
data_input_dict = torch.load('GL_data_new_dim10.pt')
x_data = data_input_dict['x_train_data'][:n_data,:] 
right_domain = data_input_dict['x_boundary'] 
X_test = data_input_dict['x_test_data'] 
pdf_test = data_input_dict['pdf_test_data']

NN_error_list = []
data_dict = {}

x_train = torch.tensor(x_data,device=device,dtype=torch.float32)

trainset = torch.utils.data.TensorDataset(x_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5000, shuffle=True)

input_size = 10
flow = zuko.flows.MAF(features=input_size, transforms=5, hidden_features=(128,128,128))
print(flow)
flow = flow.to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)




### training
for epoch in range(300):
    losses = []

    for x in trainloader:
        input_x = x[0]
        loss = -flow().log_prob(input_x).mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())

    losses = torch.stack(losses)
    if epoch % 10 == 0:

        print(f'({epoch})', losses.mean().item())

torch.save(flow.state_dict(), f'{filename}_data_'+str(n_data)+'.pth')




device = torch.device("cpu")
input_size = 10
flow = zuko.flows.MAF(features=input_size, transforms=5, hidden_features=(128,128,128))
print(flow)
flow = flow.to(device)
### loading
model_weights = torch.load('MAF_GL_density_10d_data_'+str(n_data)+'.pth')
flow.load_state_dict(model_weights)

print('n_data', n_data)

X_test_torch = torch.tensor(X_test,device=device)
log_prob = flow().log_prob(X_test_torch)
NN_pdf = torch.exp(log_prob)
NN_pdf = NN_pdf.cpu().detach().numpy()

NN_pdf = NN_pdf.reshape(-1,1)
pdf_test = pdf_test.reshape(-1,1)

err = np.linalg.norm(NN_pdf - pdf_test,2)/np.linalg.norm(pdf_test,2)
print('err', err)


data_input_dict_new_train = torch.load('GL_data_additional_dim10.pt')
x_data_new_train = data_input_dict_new_train['x_train_data']
pdf_new_train = data_input_dict_new_train['pdf_train_data'] 



x_data_new_train_torch = torch.tensor(x_data_new_train,device=device)
log_prob = flow().log_prob(x_data_new_train_torch)
NN_pdf = torch.exp(log_prob)
NN_pdf = NN_pdf.cpu().detach().numpy()


NN_pdf_adjust = []
pdf_true_adjust = []
for i in range(x_data_new_train_torch.shape[0]):
  if NN_pdf[i] > 1e-7:
    NN_pdf_adjust.append(np.log(NN_pdf[i]))
    pdf_true_adjust.append(np.log(pdf_new_train[i]))


log_prob_KL = np.mean(pdf_true_adjust)-np.mean(NN_pdf_adjust)
print('mean pdf true adjust',np.mean(pdf_true_adjust))
print('adjust KL divergence',log_prob_KL)