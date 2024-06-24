
import numpy as np
import torch
import time
### Ginzburg-Landau distribution
import matplotlib.pyplot as plt

device = torch.device("cpu")


 
class gen_GL_data:
    def __init__(self, d, epsilon, left_domain, right_domain, Ttotal_1, tau, lamda, h_list,normal_type):
        self.d = d
        self.n_dims = d
        self.epsilon = epsilon
        self.left_domain = left_domain
        self.right_domain = right_domain
        self.Ttotal_1 = Ttotal_1
        self.tau = tau
        self.lamda = lamda 
        self.h_list = h_list
        self.normal_type = normal_type
        self.h = h_list[0]
        self.beta = 2/self.epsilon**2
        
        ### domain of the underlying density [left, right]
        ### generate density, always [-1,1]
    
    ### input corporate a h_list here, this could be different
    def grad(self, X):

        
        B = (X**3 - X)/self.lamda
        
        A = torch.zeros_like(X,device=device)
        A[:, 0, :] = (X[:,0,:])*self.lamda/self.h_list[0]**2 + (X[:,0,:] - X[:,1,:])*self.lamda/self.h_list[1]**2
        for i in range(1,self.d-1):
            A[:, i, :] = (X[:,i,:] - X[:,i-1,:])*self.lamda/self.h_list[i]**2 +\
                         (X[:,i,:] - X[:,i+1,:])*self.lamda/self.h_list[i+1]**2
                    
        A[:, -1, :] = (X[:,-1,:] - X[:,-2,:])*self.lamda/self.h_list[self.d-1]**2 +\
                      (X[:,-1,:])*self.lamda/self.h_list[self.d-1]**2

        DeltaU = A + B 
       
        return DeltaU
    
    def generate(self, N):

        Xinit = torch.zeros([N, self.d],device=device)
        # Xinit = torch.rand([N, self.d],device=device) * (self.right_domain - self.left_domain) + self.left_domain
        ## [left_domian, right_domain]

        
        tau_num_1 = int(self.Ttotal_1 / self.tau)

        num_repeat = 1 
        Xtmp = Xinit.unsqueeze(2)


        for _ in range(tau_num_1):
            noise = torch.randn([N, self.d, num_repeat],device=device)
            Xtmp = Xtmp - (self.grad(Xtmp)) * self.tau + self.epsilon * np.sqrt(self.tau) * noise

        X_output1 = Xtmp.mean(dim=2)
        
        
        X_train = X_output1
        X_train = X_train.cpu().detach().numpy()
        
        X_train_min = np.min(X_train)
        X_train_max = np.max(X_train)
        if X_train_min < self.left_domain or X_train_max > self.right_domain:
            raise Exception('Out of Boundary')

        print('success')
        X_shift = X_train/self.right_domain ## [-1,1]

        return X_shift ## always [-1,1]

    def GLpdf(self,X): ## the density into [-1,1]
        # without normalization yet
        
        X = X*self.right_domain ## shift to [left, right]
        tmp = 0
        tmp += (self.lamda/2*(X[:,0])**2/self.h**2) # + 1/4/self.lamda*(1)**2) 
        for i in range(self.n_dims):
            if i == self.n_dims - 1:
                tmp += (self.lamda/2*(X[:,i])**2/self.h**2 + 1/4/self.lamda*(X[:,i]**2-1)**2) 
            else:
                tmp += (self.lamda/2*(X[:,i]-X[:,i+1])**2/self.h**2 + 1/4/self.lamda*(X[:,i]**2-1)**2)
                
        pdf = np.exp(-self.beta*tmp)
        return pdf*(self.right_domain**self.n_dims)

    def Normal_const(self,normal_type):
        if normal_type=='Known':

            # rec = [0.0,0.52152,0.3967,0.3012,0.22975,0.1749,0.1319,0.0996,0.07587,0.05832]
            # rec = [1.0]*20
            # return rec[self.n_dims-1]
            if self.n_dims == 10:
                return 13.13
            else:
                raise Exception('Not valid dimension')
                
    
        if normal_type == "Uniform":       
            ## [-1,1]
            # Normal_const = 0
            X_total = np.random.uniform(low=-1.0,high=1.0,size=[4000000, self.n_dims])
            
            Normal_const = np.mean(self.GLpdf(X_total))
              
            Normal_const = Normal_const*(2**self.n_dims)
            return Normal_const
    
    def GLpdf_normal(self,x):
        return self.GLpdf(x)/self.Normal_const(self.normal_type)
'''
final setting for GL
'''

dim = 10 

epsilon = 4  
left_domain = -2.5
right_domain = 2.5
Ttotal_1 = 10.0 
tau = 1e-3 

lamda = 0.02 
h = 1/(dim+1) 
hlist = [h for _ in range(dim)]

normal_type = 'Known'
Gen_data = gen_GL_data(dim, epsilon, left_domain, right_domain, Ttotal_1, tau, lamda, hlist, normal_type)


data_input_dict = {}

for k in range(10):
    start = time.time()
    print('K',k)
    N_train = 500000 
    X_train = Gen_data.generate(N_train)
    pdf_train = np.array(Gen_data.GLpdf_normal(X_train))

    X_test = np.random.uniform(-1.0, 1.0, size=(500000, dim)).astype(np.float32)
    pdf_test = np.array(Gen_data.GLpdf_normal(X_test))

    data_input_dict['x_train_data_'+str(k)] = X_train
    data_input_dict['pdf_train_data_'+str(k)] = pdf_train
    data_input_dict['x_boundary'] = right_domain
    data_input_dict['x_test_data_'+str(k)] = X_test
    data_input_dict['pdf_test_data_'+str(k)] = pdf_test

    end = time.time()
    print('time', end-start)
    torch.save(data_input_dict, 'GL_data_new_dim'+str(dim)+'.pt')
    # torch.save(data_input_dict, 'GL_data_additional_dim'+str(dim)+'.pt')


print('finish')


