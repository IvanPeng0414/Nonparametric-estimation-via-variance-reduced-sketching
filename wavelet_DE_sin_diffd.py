#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
this code is for wavelet, we need to shift the data from [-1,1] to [0,1] first
'''
import numpy as np
import json
from json import JSONEncoder
from generate_sin_data import Sin_distribution
from kde import kernel_density

class indexing:
    def __init__(self) -> None:
        pass
    def generate(self,max_level):
        seen={}
        for index in range(2**max_level):
            temp=len(bin(index+1))-2
            seen[index+1]=(temp-1,index+1-2**(temp-1))
        return seen



class wavelet:
    def __init__(self) -> None:
        self.seen=indexing().generate(5)
        self.power=[np.power(2,nn/2) for nn in range(11)]
    def generator(self,x):
        if x>1 or x<0:
            return 0
        if x<0.5:
            return 1
        return -1
    def compute(self,l_index,x):
        if l_index==0:
            return 1
        level, k= self.seen[l_index]

        return  self.power[level]*self.generator((self.power[level]**2)*x-k)






class generate_x_tensor:
    def __init__(self):
        self.wavelet_basis=wavelet()

    #basis_val[d][m] is the m-th wavelet basis evaluated at x[d]
    def compute_basis_val(self, dim,x,tensor_shape):

        basis_val=[]
        for d in range(dim):
            basis_val.append([])
            basis_val[d]=[self.wavelet_basis.compute(index_l,x[d]) for index_l in range(tensor_shape[d]) ]
        return basis_val


class tensor:
    def __init__(self):

        self.generate_x_tensor= generate_x_tensor()
    def compute_rank_one(self,dim,basis_val):
        #we compute rank 1 tensor here
        #the rank one tensor is
        #[p_0(x[0]), p_1(x[0]) , ...  ]\otimes [p_0(x[1]),p_1(x[1]),... ]\otimes [p_0(x[d-1]), p_1(x[d-1]),... ]
        temp_tensor=1


        for dd in range(dim):
            temp_tensor= np.tensordot([temp_tensor], [basis_val[dd]],axes=[[0],[0]])
        return temp_tensor
    def compute_tensor(self,x_train,tensor_shape, dim):

            A_temp=np.zeros(tensor_shape, dtype=float)

            for i in range(len(x_train)):
                basis_val=self.generate_x_tensor().compute_basis_val(dim,x_train[i],tensor_shape)


                A_temp=A_temp+self.compute_rank_one(dim,basis_val)

            A_temp=A_temp/len(x_train)
            return A_temp

    def compute_range_basis(self,index,x_train,tensor_shape, dim):
        #index is the x, everything else is y


        A_temp=np.zeros(tensor_shape, dtype=float)

        for i in range(len(x_train)):
            x_train[i][0], x_train[i][index]= x_train[i][index],x_train[i][0]
            basis_val=generate_x_tensor().compute_basis_val(dim,x_train[i],tensor_shape)


            A_temp=A_temp+tensor().compute_rank_one(dim,basis_val)
            x_train[i][0], x_train[i][index]= x_train[i][index],x_train[i][0]
        A_temp=A_temp/len(x_train)
        #print(A_temp)

        dim_right= int(np.prod(tensor_shape[1:]) )
        A_temp= A_temp.reshape( tensor_shape[0], dim_right )


        P,D = np.linalg.svd(A_temp, full_matrices=False, hermitian=False)[:2]


        #thresholding the rank
        #should be the maximum ratio!!
        cur_rank=0
        cur_max_ratio=0
        for rank in range(1,len(D)):
            if  D[rank-1]/D[rank]>cur_max_ratio:
                cur_max_ratio=D[rank-1]/D[rank]
                cur_rank=rank
        #we keep everything  D[1,...,rank-1]
        #print chosen_rank to ensure that the threshold we choose is good

        ##################################### this is essential  for debugging
        ###omitted for now to run large experiments
        #print(D,cur_rank)

        print(cur_rank)
        #P_transpose gets the colunms into the rows
        P_transpose=P.transpose()







        P_basis= P_transpose[:  min(max(2,cur_rank),3) ]
        #P_basis= P_transpose[: max(cur_rank,3)]



        return P_basis




class tensor_new_basis:
    def __init__(self, tensor_shape, dim,X_train):
        self.P_x_basis=[]
        self.new_shape=[]
        self.tensor_shape=tensor_shape
        self.tensor=tensor()
        self.dim=dim
        self.wavelet_basis=wavelet()
        for dd in range(dim):
            print('range',dd)

            self.P_x_basis.append( self.tensor.compute_range_basis(dd,X_train,tensor_shape, dim) )
            self.new_shape.append(len(self.P_x_basis[-1]))


        self.A_predict=np.zeros(self.new_shape, dtype=float)
        for i in range(len(X_train)):

            self.A_predict=self.A_predict+self.generate_new_rank_one(X_train[i])

        self.A_predict=self.A_predict/len(X_train)


    def generate_new_rank_one(self,xx):
        new_basis_val=[]
        for d in range(self.dim):
            new_basis_val.append([])
            temp= np.array( [self.wavelet_basis.compute(__,xx[d])  for __ in range(self.tensor_shape[0]) ])
            new_basis_val[d]=[np.dot(temp,self.P_x_basis[d][ll])  for ll in range(self.new_shape[d]) ]

        temp_tensor=1
        for dd in range(self.dim):
            temp_tensor= np.tensordot([temp_tensor], [new_basis_val[dd]],axes=[[0],[0]])
        return temp_tensor
    def predict(self,x_test):


        temp=self.generate_new_rank_one(x_test)
        return np.sum(np.multiply(self.A_predict, temp))
        ans=self.A_predict
        temp=self.generate_new_rank_one(x_test)

        for dd in range(self.dim):
            ans=(np.tensordot(ans,temp[dd],axes=([0],[0])) )

        return ans

















dims = [2,3,4,5,6,7,8,9,10]


step_size_list_new = [0.0, 0.0, 0.25, 0.3,0.3,0.35,0.35,0.35,0.4,0.4,0.4]
n_data_list = [1000000]
data_dict = {}
MM_list = [0,0,10,8,6,4,4,4,4,4,4]
LL_list = [0,0,10,5,3,3,2,2,2,2,2]

wavelet_error_list = []
KDE_error_list = []
for dim in dims:

    sin_model = Sin_distribution(n_dims=dim, lamda=1.0, beta=1.0, h=1.0, normal_type="known")

    step_size = step_size_list_new[dim]
    X_test = np.random.uniform(-1.0,1.0,size=(100000,dim)).astype(np.float32)
    X_new = X_test/2 + 1/2  
    
    for n_data in n_data_list:
        x_data = sin_model.sample(n_data, step_size)
        X_train = x_data/2 + 1/2
        print('n_data', n_data)
        print('dimension', dim)

        MM,LL = MM_list[dim], LL_list[dim]   #4,2
        tensor_shape=[LL for _ in range(dim)]
        tensor_shape[0]=MM
        model=tensor_new_basis(tensor_shape,dim,X_train)




        y_lr=np.array([model.predict(xx) for xx in X_new])
        P_true = np.array([sin_model.sinpdf_normal(xx) for xx in X_test])


        err_lr = np.linalg.norm(y_lr - P_true,2) /np.linalg.norm(P_true,2)
        print('LR error',err_lr)
        err_lr = np.linalg.norm(y_lr - P_true*2**dim,2) /np.linalg.norm(P_true*2**dim,2)
        print('LR error',err_lr)
        wavelet_error_list.append(err_lr)

        y_kde=kernel_density().compute(dim, X_train, X_new)
        err_kde = np.linalg.norm(y_kde - P_true,2)/np.linalg.norm(P_true,2)
        print('KDE error', err_kde)
        err_kde = np.linalg.norm(y_kde - P_true*2**dim,2)/np.linalg.norm(P_true*2**dim,2)
        print('KDE error', err_kde)
        KDE_error_list.append(err_kde)

        data_dict['wavelet_error_list'] = wavelet_error_list
        data_dict['KDE_error_list'] = KDE_error_list

        with open('data_wavelet_DE_sin_diffd.json', 'w') as json_file:
            # Serialize the dictionary to JSON and save it to the file
            json.dump(data_dict, json_file)
            
print('wavelet_error_list', wavelet_error_list)
print('KDE_error_list', KDE_error_list)


