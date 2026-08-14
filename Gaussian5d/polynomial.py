import numpy as np
from scipy.special import legendre



#legendre polynomial on [0,1]
#need to add a change of basis
class polynomial:
    def __init__(self,max_degree=30) -> None:
        self.max_degree=max_degree
        self.coefficients=[]
        
        for i in range(max_degree+1):
            self.coefficients.append(np.array(legendre(i))* np.sqrt(2*i+1))
    
    def compute_single_x_single_degree(self, degree,powers_x):
        return np.dot( self.coefficients[degree], powers_x[-degree-1:]) 
    
    def compute_single_x_all_degree(self,  cur_shape,x):
        powers_x= np.ones(cur_shape)
        new_x=2*x-1
        for i in range(cur_shape-2, -1,-1):
            powers_x[i]=powers_x[i+1]*new_x
        
        return np.array([ self.compute_single_x_single_degree(degree, powers_x) for degree  in range(cur_shape)  ])

 
class generate_basis_vector_given_coordinates:
    def __init__(self,dim):
        self.basis=polynomial()
        self.dim=dim
    #basis_val[d][m] is the m-th basis evaluated at x[d]

    def compute_basis_val_new_coordinates(self,x_input,tensor_shape, new_coordinates):
        
        basis_val=[self.basis.compute_single_x_all_degree(tensor_shape[d],x_input[new_coordinates[d]]) for d in range(self.dim)]

        return basis_val

######################################################

class new_basis:
    def __init__(self,M, U) -> None:
        self.M=M
        self.coefficients=[]
        
        for i in range(M):
            self.coefficients.append(np.concatenate ((np.zeros(M-i-1) , np.array(legendre(i))* np.sqrt(2*i+1) )))
        self.new_coefficients= np.matmul(U,self.coefficients)
        self.new_length=len(U)
        

    def compute_single_x_all_new_basis(self,x):
        
        powers_x= np.ones(self.M)
        new_x=2*x-1
        for i in range(self.M-2, -1,-1):
            powers_x[i]=powers_x[i+1]*new_x
        
        return np.array([  np.dot(self.new_coefficients[index], powers_x) for index  in range(self.new_length)  ])






class generate_new_basis_vector:
    def __init__(self, dim,M ,P_x_basis):
        self.basis=[]
        for dd in range(dim):
            self.basis.append(new_basis(M,P_x_basis[dd]))
        self.dim=dim

    #basis_val[d][m] is the m-th basis evaluated at x[d]
    def compute_basis_val(self,x_input):
        
        basis_val=[self.basis[d].compute_single_x_all_new_basis (x_input[d]) for d in range(self.dim)]

        return basis_val
    def compute_basis_val_given_coordinates(self, x_input, new_coordinates):
        
        basis_val=[self.basis[d].compute_single_x_all_new_basis(x_input[new_coordinates[d]]) for d in range(self.dim)]

        return basis_val