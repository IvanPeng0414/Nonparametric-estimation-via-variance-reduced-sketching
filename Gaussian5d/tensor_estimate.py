from polynomial import generate_basis_vector_given_coordinates,generate_new_basis_vector
import numpy as np

 



class compute_rank_one:
    #we compute rank 1 tensor given basis_val
        #the rank one tensor is 
        #[p_0(x[0]), p_1(x[0]) , ...  ]\otimes [p_0(x[1]),p_1(x[1]),... ]\otimes [p_0(x[d-1]), p_1(x[d-1]),... ]
    def __init__(self,dim):
        self.dim=dim

    def compute(self,basis_val):
        temp_tensor=1
        
        for dd in range(self.dim-1, -1, -1):
            temp_tensor= np.tensordot( np.reshape(basis_val[dd], (len(basis_val[dd]),1)), [temp_tensor],axes=(1,0))

        #for dd in range(dim):
        #    temp_tensor= np.tensordot([temp_tensor], [basis_val[dd]],axes=[[0],[0]])
        return temp_tensor 


class SVD_adaptive_thresholding:
    def __init__(self):
        pass
    def compute(self, A_temp):
        P,D = np.linalg.svd(A_temp, full_matrices=True, hermitian=False)[:2]
        
        #thresholding the rank 
        #should be the maximum ratio!!
        #we keep at least one singular function

        cur_rank=1
        cur_square_sum= D[0]*D[0]
        #print(D)
        #for rank in range(1,len(D)):
        for rank in range(1,len(D)):
            temp=D[rank]*D[rank]
            if temp/cur_square_sum<1/100:
                break
            else:
                cur_rank=rank+1
                cur_square_sum+=temp
        #we keep everything  D[1,...,rank-1]
        #print chosen_rank to ensure that the threshold we choose is good
        
        ##################################### this is essential  for debugging 
        ###omitted for now to run large experiments
        #print(D,cur_rank)
        # cur_rank=min(max(cur_rank,2),4)
        cur_rank=max(cur_rank,2)
        # cur_rank = 3 #3
        #print(D)
        print('selected rank =', cur_rank)
        P_transpose=P.transpose()
        
        P_basis= P_transpose[: cur_rank]
        #print(P_basis)
        
        
        return P_basis





class tensor:
    def __init__(self, dim,index):
        self.dim=dim
        self.generate_basis_vector_given_coordinates = generate_basis_vector_given_coordinates(dim)
        self.new_coordinate=[ ii  for ii in range(self.dim)]
        self.new_coordinate[0]=index
        self.new_coordinate[index]=0
        self.compute_rank_one=compute_rank_one(dim)


    def compute_tensor(self,x_train,tensor_shape, new_coordinates):

            A_temp=np.zeros(tensor_shape, dtype=float)
            
            for i in range(len(x_train)):
                basis_val=self.generate_basis_vector_given_coordinates.compute_basis_val_new_coordinates(x_train[i],tensor_shape, new_coordinates)

                
                A_temp=A_temp+self.compute_rank_one.compute(basis_val)             
                
            A_temp=A_temp/len(x_train)
            return A_temp





    def compute_range_basis(self,x_train,tensor_shape):

        

        A_temp = self.compute_tensor(x_train,tensor_shape, self.new_coordinate)
        #print(A_temp)
        
 
        A_temp= A_temp.reshape( tensor_shape[0],  -1) 
        return SVD_adaptive_thresholding().compute(A_temp)





from transform_domain import domain


class vrs_prediction:
    def __init__(self, tensor_shape, dim,M,X_train):

        
        self.compute_rank_one=compute_rank_one(dim)
        self.dim=dim 
        self.new_domain=domain(dim, X_train)
        self.X_train_transform=self.new_domain.compute_data(X_train)
        self.X_train_transform=np.clip(self.X_train_transform, 0, 1)
        N_train=len(X_train)

        #compute projection basis matrix using iteration function
        self.cur_basis=[]
        self.cur_shape=[]
        for dd in range(dim):
            print('coordinate =',dd)
            
            self.cur_basis.append( tensor(dim,dd).compute_range_basis(self.X_train_transform,tensor_shape) ) 
            self.cur_shape.append(len(self.cur_basis[-1]))
        
        
        self.generate_new_basis_vector=generate_new_basis_vector(dim,M,self.cur_basis)
        
        
         
        self.A_predict=np.zeros(self.cur_shape, dtype=float)
        for i in range(N_train):
            basis_val=self.generate_new_basis_vector.compute_basis_val( self.X_train_transform[i])

            self.A_predict=self.A_predict+self.compute_rank_one.compute(basis_val)   
            
        self.A_predict=self.A_predict/N_train

    
    def predict_one_x(self,X_test):
        X_test_transform= self.new_domain.compute_data(X_test)
        X_test_transform=np.clip(X_test_transform, 0, 1)
        basis_val=self.generate_new_basis_vector.compute_basis_val(X_test_transform)
        
        
        
        ans= np.tensordot(basis_val[0],self.A_predict,axes=(0,0) )
        for dd in range(1,self.dim):
            ans= np.tensordot(basis_val[dd],ans,axes=(0,0) )
        ans=self.new_domain.transform_density_val(ans)
        return ans

    def predict(self,X_new):
        y_lr=np.array([self.predict_one_x(xx) for xx in X_new])
        return y_lr
