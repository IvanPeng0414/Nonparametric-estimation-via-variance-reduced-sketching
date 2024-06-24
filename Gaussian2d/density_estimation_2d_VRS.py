'''
this code summarizes the 2d example
'''
import numpy as np
import json
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


class GMM_distribution_2d():


    def __init__(self, n_dims=2, normal_type="Uniform"):
        """
        Initializes the SinDistribution class.

        Parameters:
        n_dims (int): The number of dimensions.
        normal_type (str): The type of normalization to use. Options are "Uniform" or "Known".
        """
        self.n_dims = n_dims
        self.means_1 = np.array([-0.5,-0.5])
        self.means_2 = np.array([0.5,0.5])
        self.means_3 = np.array([-0.5,0.5])
        self.means_4 = np.array([0.5,-0.5])
        self.means = [self.means_1,self.means_2,self.means_3,self.means_4]        

        
        self.cov_1 = np.array([[0.25**2,0.03**2],[0.03**2,0.25**2]])    
        self.cov_2 = np.array([[0.25**2,0.03**2],[0.03**2,0.25**2]])
        
        self.cov_3 = np.array([[0.1**2,-0.05**2],[-0.05**2,0.1**2]])   
        self.cov_4 = np.array([[0.1**2,-0.05**2],[-0.05**2,0.1**2]])
        
        self.covariances = [self.cov_1,self.cov_2,self.cov_3,self.cov_4]

        
        self.normal_type = normal_type
        self.weight_list = [0.25,0.25,0.25,0.25]
        self.lower_bound = np.array([-1, -1])
        self.upper_bound = np.array([1, 1])

    # Function to generate samples from a d-dimensional Gaussian mixture model
    def generate_gmm_samples(self,num_samples):
        samples = []
        while len(samples) < num_samples:
            # Choose a component based on weights
            component = np.random.choice(len(self.weight_list), p=self.weight_list)
            # Sample from the chosen component
            sample = multivariate_normal.rvs(mean=self.means[component], cov=self.covariances[component])
            # Check if the sample is within the bounds
            if np.all(sample >= self.lower_bound) and np.all(sample <= self.upper_bound):
                samples.append(sample)
        return np.array(samples)
    
    
    # Function to compute the density of the Gaussian mixture model at a point
    ## the function has been normalized
    def gmm_pdf(self,x):
        '''
        input a array/list
        x in [-1,1]

        '''
       
        density_1 = self.weight_list[0]*stats.multivariate_normal.pdf(x[:,:2], mean=self.means_1, cov=self.cov_1) +\
                    self.weight_list[1]*stats.multivariate_normal.pdf(x[:,:2], mean=self.means_2, cov=self.cov_2) +\
                    self.weight_list[2]*stats.multivariate_normal.pdf(x[:,:2], mean=self.means_3, cov=self.cov_3) +\
                    self.weight_list[3]*stats.multivariate_normal.pdf(x[:,:2], mean=self.means_4, cov=self.cov_4) 
        density = density_1


        return density
    

    def Normal_const(self, normal_type):
        """
        Computes the normalization constant based on the specified normal type.

        Parameters:
        normal_type (str): The type of normalization ("Known" or "Uniform").

        Returns:
        float: The normalization constant.
        """
        if normal_type == 'Known':
            # Predefined normalization constants for known distributions
            rec = [0.974]*10
            return rec[self.n_dims]

        if normal_type == "Uniform":
            # Estimate normalization constant for a Uniform distribution
            Normal_const = 0
            X_total = np.random.uniform(low=-1.0, high=1.0, size=[50000, self.n_dims])
            
            Normal_const = np.mean(self.gmm_pdf(X_total))
      
            return Normal_const * 2 ** self.n_dims


    def gmm_pdf_normal(self, x):

        return self.gmm_pdf(x) / self.Normal_const(self.normal_type)
    
    
    

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

########################################
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
        P_basis = P_transpose[:cur_rank]
        # P_basis= P_transpose[:  max(2,cur_rank) ]
        
        #(rank, M)
        return P_basis


class VRSModel:
    """
    This class represents the VRS (Variable Rank Spline) model.
    """
    def __init__(self, tensor_shape, dim, X_train):
        """
        Initializes the VRSModel class.

        Parameters:
        tensor_shape (list): Shape of the tensor.
        dim (int): Dimension of the tensor.
        X_train (list): Training data.
        """
        self.P_x_basis = []
        self.new_shape = []
        self.tensor_shape = tensor_shape
        self.tensor = Tensor()
        self.dim = dim
        self.tensor_shape_M = [tensor_shape[0][0]]*self.dim

        self.generate_basis = generate_poly_tensor()
        self.legendre_coefficients = generate_legendre().generate(self.tensor_shape_M)
        
        
        
        for dd in range(dim):
            self.P_x_basis.append(self.tensor.compute_range_basis(dd, X_train, tensor_shape[dd], dim))
            self.new_shape.append(len(self.P_x_basis[-1]))
        print(self.new_shape)

        self.A_predict = np.zeros(self.new_shape, dtype=np.float32)
        for i in range(len(X_train)):
            self.A_predict += self.generate_new_rank_one(X_train[i])/len(X_train)


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
        # index_total = np.zeros((self.dim,),dtype=np.int8)
        Core_total = np.zeros((self.new_shape[index_i],self.new_shape[index_j]))
        for i in range(self.new_shape[index_i]):
            for j in range(self.new_shape[index_j]):
                index_total = np.zeros((self.dim,),dtype=np.int8)
                index_total[index_i] = i
                index_total[index_j] = j
                index_total = index_total.tolist()              
                Core_tmp = self.A_predict[tuple(index_total)]

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


flow_MAF = zuko.flows.MAF(features=2, transforms=5, hidden_features=(128,128,128))

flow_MAF = flow_MAF.to(device)
model_weights_MAF = torch.load('NF_MAF_Gaussian_truncated_2d.pth', map_location=torch.device('cpu'))
flow_MAF.load_state_dict(model_weights_MAF)

flow_NAF = zuko.flows.NAF(features=2, transforms=5, hidden_features=(128,128,128))
flow_NAF = flow_NAF.to(device)
model_weights_NAF = torch.load('NF_NAF_Gaussian_truncated_2d.pth', map_location=torch.device('cpu'))
flow_NAF.load_state_dict(model_weights_NAF)


'''
main part
'''
data_dict = {}
VRS_error_list = []

dim = 2 
n_data = 20000 

gmm_model = GMM_distribution_2d(n_dims=dim, normal_type="Known")

X_test = np.random.uniform(-1.0, 1.0, size=(40000, dim)).astype(np.float32)
P_true = np.array(gmm_model.gmm_pdf_normal(X_test))

# x_data = gmm_model.generate_gmm_samples(n_data)
# X_train = x_data 
# x_data_new = gmm_model.generate_gmm_samples(n_data)
# y_true_new = np.array(gmm_model.gmm_pdf_normal(x_data_new))


print('n_data', n_data)
print('dimension', dim)
Normal_const = gmm_model.Normal_const("Uniform")
print('Normal_const',Normal_const)


# data_input_dict = {}
# data_input_dict['x_train_data'] = x_data
# data_input_dict['x_train_data_new'] = x_data_new
# data_input_dict['x_train_pdf_new'] = y_true_new
# data_input_dict['x_test_data'] = X_test
# data_input_dict['x_test_pdf'] = P_true
# torch.save(data_input_dict, 'gaussian_truncated_2d_data_20000.pt')

data_input_dict = torch.load('gaussian_truncated_2d_data_20000.pt')

x_data = data_input_dict['x_train_data']
X_train = x_data  
x_data_new = data_input_dict['x_train_data_new']
y_true_new = data_input_dict['x_train_pdf_new']
X_test = data_input_dict['x_test_data']
P_true = data_input_dict['x_test_pdf']



###### 
## KDE part
#####
KDE = kernel_density(dim, X_train)
                
y_kde = KDE.compute(X_test)

kde_sample = KDE.generate(len(x_data))
kde_sample = kde_sample.T

err_kde = np.linalg.norm(y_kde - P_true, 2) / np.linalg.norm(P_true, 2)
print('KDE error', err_kde)



y_kde_data_new = KDE.pdf(x_data_new)
err_KL = np.mean(np.log(y_true_new)) - np.mean(np.log(y_kde_data_new))
print('KL KDE error', err_KL)
y_KDE_adjust = []
y_true_adjust_KDE = []
for i in range(len(x_data_new)):
    if y_kde_data_new[i] > 1e-7:
        y_KDE_adjust.append(np.log(y_kde_data_new[i]))
        y_true_adjust_KDE.append(np.log(y_true_new[i]))
err_KL_KDE_adjust = np.mean(y_true_adjust_KDE) - np.mean(y_KDE_adjust)        
print('KL KDE error', err_KL_KDE_adjust)                
############ VRS
M = 20
L = 12
tensor_shape = [[M,L]]*2
model = VRSModel(tensor_shape, dim, X_train)
y_vrs = np.array([model.predict(xx) for xx in X_test])
err_vrs = np.linalg.norm(y_vrs - P_true, 2) / np.linalg.norm(P_true, 2)
print('VRS error', err_vrs)

y_VRS_data_new = np.array([model.predict(xx) for xx in x_data_new])
y_true_new = np.array(gmm_model.gmm_pdf_normal(x_data_new))
y_VRS_adjust = []
y_true_adjust = []
for i in range(len(x_data_new)):
    if y_VRS_data_new[i] > 1e-7:
        y_VRS_adjust.append(np.log(y_VRS_data_new[i]))
        y_true_adjust.append(np.log(y_true_new[i]))
err_KL_VRS = np.mean(y_true_adjust) - np.mean(y_VRS_adjust)
print('KL error', err_KL_VRS)


#####
## MAF
#####
x_test_torch = torch.tensor(X_test,device=device)
log_prob = flow_MAF().log_prob(x_test_torch)
NN_pdf = torch.exp(log_prob)
NN_pdf = NN_pdf.cpu().detach().numpy()

error = np.linalg.norm(P_true - NN_pdf,2)/np.linalg.norm(P_true,2)
print('MAF L2 error', error)


x_data_new_train_torch = torch.tensor(x_data_new,device=device,dtype=torch.float32)
log_prob = flow_MAF().log_prob(x_data_new_train_torch)
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


#####
## NAF
#####
x_test_torch = torch.tensor(X_test,device=device)
log_prob = flow_NAF().log_prob(x_test_torch)
NN_pdf = torch.exp(log_prob)
NN_pdf = NN_pdf.cpu().detach().numpy()

error = np.linalg.norm(P_true - NN_pdf,2)/np.linalg.norm(P_true,2)
print('NAF L2 error', error)


x_data_new_train_torch = torch.tensor(x_data_new,device=device,dtype=torch.float32)
log_prob = flow_NAF().log_prob(x_data_new_train_torch)
NN_pdf = torch.exp(log_prob)
NN_pdf = NN_pdf.cpu().detach().numpy()


NN_pdf_adjust = []
pdf_true_adjust = []

for i in range(x_data_new_train_torch.shape[0]):
  if NN_pdf[i] > 1e-7:
    NN_pdf_adjust.append(np.log(NN_pdf[i]))
    pdf_true_adjust.append(np.log(y_true_new[i]))
log_prob_KL = np.mean(pdf_true_adjust)-np.mean(NN_pdf_adjust)
print('KL_NAF_adjust', log_prob_KL)





#######
'''
plot the figure
'''
###########
index_i = 0
index_j = 1
n_bins = 100
N_mesh = n_bins
x = np.linspace(-1,1,N_mesh)
X_1, X_2 = np.meshgrid(x,x)
xx_total = np.concatenate([X_1.reshape(-1,1), X_2.reshape(-1,1)],axis=1)

Gaussian_pdf = np.array(gmm_model.gmm_pdf_normal(xx_total))
Gaussian_pdf = Gaussian_pdf/np.sum(Gaussian_pdf)
Gaussian_pdf_matrix = Gaussian_pdf.reshape(N_mesh,N_mesh)


marginal_12pdf = model.VRS_2marginal_pdf(index_i, index_j, xx_total)
marginal_12pdf = marginal_12pdf/np.sum(marginal_12pdf)
marginal_12pdf_matrix = marginal_12pdf.reshape(N_mesh,N_mesh)

KDE_pdf = KDE.pdf(xx_total)
KDE_pdf = KDE_pdf/np.sum(KDE_pdf)
KDE_pdf_matrix = KDE_pdf.reshape(N_mesh,N_mesh)



xx_total_torch = torch.tensor(xx_total,device=device,dtype=torch.float32)
log_prob_MAF = flow_MAF().log_prob(xx_total_torch)
MAF_pdf = torch.exp(log_prob_MAF)
MAF_pdf = MAF_pdf.cpu().detach().numpy()
MAF_pdf = MAF_pdf/np.sum(MAF_pdf)
MAF_pdf_matrix = MAF_pdf.reshape(N_mesh,N_mesh)



log_prob_NAF = flow_NAF().log_prob(xx_total_torch)
NAF_pdf = torch.exp(log_prob_NAF)
NAF_pdf = NAF_pdf.cpu().detach().numpy()
NAF_pdf = NAF_pdf/np.sum(NAF_pdf)
NAF_pdf_matrix = NAF_pdf.reshape(N_mesh,N_mesh)




import matplotlib.gridspec as gridspec

plt.figure(figsize=(31,6))
gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05])

plt.subplot(gs[0])
plt.imshow(Gaussian_pdf_matrix,origin='lower',
           extent=[-1,1,-1,1],
           vmin=0.0, vmax=0.0016)
plt.xticks([])
plt.yticks([])
plt.xlabel('$x_1$',fontsize=36)
plt.ylabel('$x_2$',fontsize=36)
plt.title('True density',fontsize=32)


plt.subplot(gs[1])
plt.imshow(marginal_12pdf_matrix,origin='lower',
           extent=[-1,1,-1,1],
           vmin=0.0, vmax=0.0016)
plt.xticks([])
plt.yticks([])
plt.xlabel('$x_1$',fontsize=36)
plt.ylabel('$x_2$',fontsize=36)
plt.title('VRS',fontsize=32)

# plt.subplot(1,5,3)
plt.subplot(gs[2])
plt.imshow(KDE_pdf_matrix,origin='lower',
           extent=[-1,1,-1,1],
           vmin=0.0, vmax=0.0016)
plt.xticks([])
plt.yticks([])
plt.xlabel('$x_1$',fontsize=36)
plt.ylabel('$x_2$',fontsize=36)
plt.title('KDE',fontsize=32)    


# plt.subplot(1,5,4)
plt.subplot(gs[3])
plt.imshow(MAF_pdf_matrix,origin='lower',
           extent=[-1,1,-1,1],
           vmin=0.0, vmax=0.0016)
plt.xticks([])
plt.yticks([])
plt.xlabel('$x_1$',fontsize=36)
plt.ylabel('$x_2$',fontsize=36)
plt.title('NN-MAF',fontsize=32)       


plt.subplot(gs[4])
plt.imshow(NAF_pdf_matrix,origin='lower',
           extent=[-1,1,-1,1],
           vmin=0.0, vmax=0.0016)
plt.xticks([])
plt.yticks([])
plt.xlabel('$x_1$',fontsize=36)
plt.ylabel('$x_2$',fontsize=36)
plt.title('NN-NAF',fontsize=32)

cbar = plt.colorbar(cax=plt.subplot(gs[5]))
cbar.ax.tick_params(labelsize=24)  
plt.tight_layout()
plt.savefig('fig-summary/Gaussian_truncated_2d_hard_result.pdf',bbox_inches='tight') 
 


