import numpy as np
import json
from scipy.special import legendre

import matplotlib.pyplot as plt
import os
import torch
script_name = os.path.basename(__file__)  # Gets the filename of the script
filename, file_extension = os.path.splitext(script_name)  # Separates file name and extension
save_filename = f"{filename}_figure.png"  # Appends '_figure.png' to the script name


class generate_legendre:
    def __init__(self):
        pass
    def generate(self,tensor_shape):
        legendre_mat=[]
        for d in range(len(tensor_shape)):
            legendre_mat.append([])
            for m in range(tensor_shape[d]):
                legendre_mat[d].append(  list(legendre(m) * np.sqrt( (2*m+1)/2) )[::-1] +[0 for _ in range( tensor_shape[d]-m-1)] )
            legendre_mat[d]=np.array(legendre_mat[d])
        return legendre_mat


class generate_poly_tensor:
    def __init__(self):
        pass
    def compute_x_power(self, dim, x, tensor_shape):
        rec=[]
        for i in range(dim):
            rec.append([1])
            for _ in range(1,tensor_shape[i]):
                rec[i].append(rec[i][-1]*x[i])
        return rec

    #basis_coef[d] are the coefficients of the polynomials for x[d]
    def compute_basis_val(self, dim,x,tensor_shape,basis_coef):
        x_power=self.compute_x_power(dim,x,tensor_shape)
        basis_val=[]
        for d in range(dim):
            basis_val.append([])
            for m in range(tensor_shape[d]):
                basis_val[d].append(np.dot(basis_coef[d][m],x_power[d]))
        return basis_val
    
 
    
    
class Tensor:
    """
    This class handles operations related to tensors in the wavelet basis.
    """
    def __init__(self):
        self.generate_x_tensor = generate_poly_tensor()
        
        

    def compute_rank_one(self, dim, basis_val):
        """
        Computes a rank-one tensor.

        Parameters:
        dim (int): The dimensionality.
        basis_val (list): The basis values.

        Returns:
        array: The computed rank-one tensor.
        """
        temp_tensor = 1
        for dd in range(dim):
            temp_tensor = np.tensordot([temp_tensor], [basis_val[dd]], axes=[[0], [0]])
        return temp_tensor

    def compute_tensor(self, x_train, tensor_shape, dim):
        """
        Computes the total tensor for the training data.

        Parameters:
        x_train (array): The training data.
        tensor_shape (list): The shape of the tensor.
        dim (int): The dimensionality.

        Returns:
        array: The computed tensor.
        """
        legendre_coefficients = generate_legendre().generate(tensor_shape)
        
        A_temp = np.zeros(tensor_shape, dtype=np.float32)
        
        for i in range(len(x_train)):
            basis_val = self.generate_x_tensor.compute_basis_val(dim, x_train[i], tensor_shape,legendre_coefficients)
            A_temp = A_temp + self.compute_rank_one(dim, basis_val)
        A_temp = A_temp / len(x_train)
        return A_temp

    def compute_range_basis(self, index, x_train, tensor_shape, dim):
        """
        Performs range estimation for the tensor.

        Parameters:
        index (int): The index for range estimation.
        x_train (list): Training data.
        tensor_shape (list): Shape of the tensor.
        dim (int): Dimension of the tensor.

        Returns:
        ndarray: Computed basis for the range.
        """
        legendre_coefficients = generate_legendre().generate(tensor_shape)
        
        A_temp = np.zeros(tensor_shape, dtype=np.float32)
        for i in range(len(x_train)):
            x_train[i][0], x_train[i][index] = x_train[i][index], x_train[i][0]
            basis_val = self.generate_x_tensor.compute_basis_val(dim, x_train[i], tensor_shape,legendre_coefficients)
            A_temp += Tensor().compute_rank_one(dim, basis_val)
            x_train[i][0], x_train[i][index] = x_train[i][index], x_train[i][0]

        A_temp /= len(x_train)
        dim_right = int(np.prod(tensor_shape[1:]))
        A_temp = A_temp.reshape(tensor_shape[0], dim_right)

        P, D = np.linalg.svd(A_temp, full_matrices=False, hermitian=False)[:2]
        cur_rank, cur_max_ratio = 0, 0
        for rank in range(1, len(D)):
            if D[rank - 1] / D[rank] > cur_max_ratio:
                cur_max_ratio, cur_rank = D[rank - 1] / D[rank], rank
                
        print('rank', cur_rank)
        P_transpose = P.transpose()
        # P_basis = P_transpose[:cur_rank]
        P_basis= P_transpose[:  min(max(2,cur_rank),4) ]
        
        #(rank, M)
        return P_basis


class VRSModel:
    """
    This class represents the VRS model.
    """
    def __init__(self, tensor_shape, dim, X_train, load_parameter, P_x_basis, A_predict):
        """
        Initializes the VRSModel class.

        Parameters:
        tensor_shape (list): Shape of the tensor.
        dim (int): Dimension of the tensor.
        X_train (list): Training data.
        """

        self.tensor_shape = tensor_shape
        self.tensor = Tensor()
        self.dim = dim
        self.tensor_shape_M = [tensor_shape[0][0]]*self.dim

        self.generate_basis = generate_poly_tensor()
        self.legendre_coefficients = generate_legendre().generate(self.tensor_shape_M)

        if load_parameter == False:
            self.new_shape = []
            self.P_x_basis = []
            for dd in range(dim):
                self.P_x_basis.append(self.tensor.compute_range_basis(dd, X_train, tensor_shape[dd], dim))
                self.new_shape.append(len(self.P_x_basis[-1]))
            print(self.new_shape)

            self.A_predict = np.zeros(self.new_shape, dtype=np.float32)
            for i in range(len(X_train)):
                self.A_predict += self.generate_new_rank_one(X_train[i])/len(X_train)
        else:
             self.P_x_basis = P_x_basis
             self.A_predict = A_predict
             self.new_shape = self.A_predict.shape


    def VRS_parameter(self,):
        return self.A_predict, self.P_x_basis

    def generate_new_rank_one(self, xx):
        """
        Computes the basis function evaluated on a new sample.

        Parameters:
        xx (list): A new sample.

        Returns:
        ndarray: A new rank one tensor.
        """
        
        legendre_basis = self.generate_basis.compute_basis_val(self.dim, xx, self.tensor_shape_M,self.legendre_coefficients)
        
        new_basis_val = []
        for d in range(self.dim):
            temp = legendre_basis[d]
            # temp = np.array([self.wavelet_basis.compute(l, xx[d]) for l in range(self.tensor_shape[0])])
            new_basis_val.append([np.dot(temp, self.P_x_basis[d][ll]) for ll in range(self.new_shape[d])])

        temp_tensor = 1
        for dd in range(self.dim):
            temp_tensor = np.tensordot([temp_tensor], [new_basis_val[dd]], axes=[[0], [0]])
        return temp_tensor

    def predict(self, x_test):
        """
        Performs prediction on a new sample.

        Parameters:
        x_test (list): Test data.

        Returns:
        float: Predicted value.
        """
        temp = self.generate_new_rank_one(x_test)
        return np.sum(np.multiply(self.A_predict, temp))
    

    def VRS_sample(self,index_i, index_j, num_sample = 1000,step_size=0.5):
        '''
        try to generate samples from the VRS model
        for given index i and j, since other component is const (chosen the first)
        we only consider f(x1,x2) = \sum_{i,j} phi_i(x1)phi_j(x2) B[i,j,0,...0]
        '''

        Core_total = np.zeros((self.new_shape[index_i],self.new_shape[index_j]))
        for i in range(self.new_shape[index_i]):
            for j in range(self.new_shape[index_j]):
                index_total = np.zeros((self.dim,),dtype=np.int8)
                index_total[index_i] = i
                index_total[index_j] = j
                index_total = index_total.tolist()
                # print(index_total)
                # print('self.A_predict',np.shape(self.A_predict))                
                Core_tmp = self.A_predict[tuple(index_total)]
                # print('Core_tmp', Core_tmp)
                Core_total[i,j] = Core_tmp
        coeff_i = self.P_x_basis[index_i] ##(r,M)
        coeff_j = self.P_x_basis[index_j] ##(r,M)
        Basis_total = np.matmul(np.matmul(coeff_i.T,Core_total),coeff_j) ##(M,M)
        ## sample from [-1,1]
        tensor_shape_M_2d = [self.tensor_shape_M[index_i],self.tensor_shape_M[index_j]]
        legendre_coefficients_2d = [self.legendre_coefficients[index_i],self.legendre_coefficients[index_j]]
        
        def target_pdf(xx):
            ## two-dimension
            legendre_basis = self.generate_basis.compute_basis_val(2, xx, tensor_shape_M_2d,legendre_coefficients_2d)
            legendre_basis_1 = np.array(legendre_basis[0])
            legendre_basis_1 = np.expand_dims(legendre_basis_1,axis=1) #(M,1)
            legendre_basis_2 = np.array(legendre_basis[1])
            legendre_basis_2 = np.expand_dims(legendre_basis_2,axis=1) #(M,1)
            # print(legendre_basis_1.shape)
            # print(legendre_basis_2.shape)
            out = np.matmul(np.matmul(legendre_basis_1.T,Basis_total),legendre_basis_2)
            
            return out[0,0]

        def proposal_sample(x,step_size):
            
            """
            Proposal distribution for Metropolis-Hastings sampling. Uses a uniform distribution.

            Parameters:
            x (array): The current sample. 2d sample

            Returns:
            array: The proposed next sample.
            """

            x_prime = x + np.random.uniform(-step_size, step_size, size=x.shape)
            
            # Ensure the proposed sample is within the bounds [-1, 1]
            num_try = 0
            while np.min(x_prime) < -1 or np.max(x_prime) > 1:
                x_prime = x + np.random.uniform(-step_size, step_size, size=x.shape)
                num_try += 1
                if num_try > 10:
                    return x
            
            return x_prime
        

        #### make the MCMC workable for double peaks model
        def MHsampling(target_pdf, proposal_sample, x_init, num_iter,step_size):
            """
            The Metropolis-Hastings algorithm for sampling from a target distribution.

            Parameters:
            target_pdf (function): The target probability density function.
            proposal_sample (function): The proposal distribution function.
            x_init (array): The initial sample.
            num_iter (int): The number of iterations.

            Returns:
            array: An array of accepted samples.
            """
            x = x_init
            accepted = []
            for iteration in range(6 * num_iter):
                x_new = proposal_sample(x,step_size)
                # Compute the acceptance probability
                alpha = min(1, target_pdf(x_new) / target_pdf(x))
                if np.random.rand() < alpha:
                    x = x_new
                # Accept the second half of samples
                if iteration >= 5*num_iter:
                    accepted.append(x)
                    
            return np.array(accepted)
        
        x_init = np.zeros((2,))
        num_iter = num_sample
        samples = MHsampling(target_pdf, proposal_sample, x_init, num_iter,step_size)
        return samples        


    def VRS_2marginal_pdf(self,index_i, index_j, xx_total):
        '''
        try to generate samples from the VRS model
        for given index i and j, since other component is const (chosen the first)
        we only consider f(x1,x2) = \sum_{i,j} phi_i(x1)phi_j(x2) B[i,j,0,...0]
        '''
        # index_total = np.zeros((self.dim,),dtype=np.int8)
        Core_total = np.zeros((self.new_shape[index_i],self.new_shape[index_j]))
        for i in range(self.new_shape[index_i]):
            for j in range(self.new_shape[index_j]):
                index_total = np.zeros((self.dim,),dtype=np.int8)
                index_total[index_i] = i
                index_total[index_j] = j
                index_total = index_total.tolist()
                # print(index_total)
                # print('self.A_predict',np.shape(self.A_predict))                
                Core_tmp = self.A_predict[tuple(index_total)]
                # print('Core_tmp', Core_tmp)
                Core_total[i,j] = Core_tmp
        coeff_i = self.P_x_basis[index_i] ##(r,M)
        coeff_j = self.P_x_basis[index_j] ##(r,M)
        Basis_total = np.matmul(np.matmul(coeff_i.T,Core_total),coeff_j) ##(M,M)
        ## sample from [-1,1]
        tensor_shape_M_2d = [self.tensor_shape_M[index_i],self.tensor_shape_M[index_j]]
        legendre_coefficients_2d = [self.legendre_coefficients[index_i],self.legendre_coefficients[index_j]]
        

        ## two-dimension
        [N_xx,d_xx] = np.shape(xx_total)
        out_total = np.zeros((N_xx,))
        for k in range(N_xx):
            xx = xx_total[k]
            legendre_basis = self.generate_basis.compute_basis_val(2, xx, tensor_shape_M_2d,legendre_coefficients_2d)
            legendre_basis_1 = np.array(legendre_basis[0])
            legendre_basis_1 = np.expand_dims(legendre_basis_1,axis=1) #(M,1)
            legendre_basis_2 = np.array(legendre_basis[1])
            legendre_basis_2 = np.expand_dims(legendre_basis_2,axis=1) #(M,1)

            out = np.matmul(np.matmul(legendre_basis_1.T,Basis_total),legendre_basis_2)
            out_total[k] = out[0,0]
            

        return out_total
    



data_dict = {}
VRS_error_list = []

dim = 10 
n_data = 100000 

data_input_dict = torch.load('GL_data_new_dim10.pt')


X_train = data_input_dict['x_train_data'][:n_data,:]
right_domain = data_input_dict['x_boundary'] 

x_data = X_train


# X_train = x_data 
print('n_data', n_data)
print('dimension', dim)



# Set tensor shape parameters
for MM in [25]:
    for LL in [4]:

        #### indepedent on each case
        tensor_shape = []
        for d_index in range(dim):
            tensor_shape_temp = 2*np.ones((dim,),dtype=np.int8)
            tensor_shape_temp[0] = MM
            if d_index - 1 >= 1:
                tensor_shape_temp[d_index-1] = LL
            if d_index+1 <= dim - 1:
                tensor_shape_temp[d_index+1] = LL
                
            tensor_shape.append(tensor_shape_temp)
                
        load_parameter = False
        
        if load_parameter == False:
            P_x_basis = []
            A_predict = []
        if load_parameter == True:
            data_parameter_dict = torch.load(f'{filename}_data_'+str(n_data)+'.pt')
            A_predict = data_parameter_dict['A_predict']
            P_x_basis = data_parameter_dict['P_x_basis']
            
        model = VRSModel(tensor_shape, dim, X_train, load_parameter, P_x_basis, A_predict)



if load_parameter == False:
    A_predict, P_x_basis = model.VRS_parameter()
    data_dict['A_predict'] = A_predict
    data_dict['P_x_basis'] = P_x_basis
    torch.save(data_dict, f'{filename}_data_'+str(n_data)+'.pt')






data_input_dict_new_train = torch.load('GL_data_additional_dim10.pt')
X_test = data_input_dict_new_train['x_test_data_1'] 
pdf_test = data_input_dict_new_train['pdf_test_data_1']

y_new_test = np.array([model.predict(xx) for xx in X_test])
y_new_test = y_new_test.reshape(-1,1)
pdf_test  = pdf_test.reshape(-1,1)

print('n_data', n_data)
err = np.linalg.norm(y_new_test - pdf_test,2)/np.linalg.norm(pdf_test,2)
print('err', err)



x_data_new_train = data_input_dict_new_train['x_train_data_1']
P_true_new_train = data_input_dict_new_train['pdf_train_data_1'] 

y_new_train = np.array([model.predict(xx) for xx in x_data_new_train])

tol = 1e-7
y_new_train_adjust = []
P_true_new_train_adjust = []
for i in range(len(x_data_new_train)):
    # if y_new_train[i] > 1e-7:
    if y_new_train[i] > tol:
        y_new_train_adjust.append(np.log(y_new_train[i]))
        P_true_new_train_adjust.append(np.log(P_true_new_train[i]))

VRS_KL = np.mean(P_true_new_train_adjust) - np.mean(y_new_train_adjust)
print('Ptrue adjust',np.mean(P_true_new_train_adjust))       
print('VRS_KL', VRS_KL)

















