#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
this code is for wavelet basis, we need to shift the data from [-1,1] to [0,1] first
'''
import numpy as np
import json
from generate_sin_data import Sin_distribution



class Indexing:
    """
    This class sets up indexing for wavelet basis.
    """
    def __init__(self):
        pass

    def generate(self, max_level):
        """
        Generates indexing for a given maximum level.

        Parameters:
        max_level (int): The maximum level for indexing.

        Returns:
        dict: A dictionary with the generated indices.
        """
        seen = {}
        for index in range(2 ** max_level):
            temp = len(bin(index + 1)) - 2
            seen[index + 1] = (temp - 1, index + 1 - 2 ** (temp - 1))
        return seen


class Wavelet:
    """
    This class represents a wavelet with methods to generate wavelet functions.
    """
    def __init__(self):
        self.seen = Indexing().generate(5)
        self.power = [np.power(2, nn / 2) for nn in range(11)]

    def generator(self, x):
        """
        Generates a single wavelet function.

        Parameters:
        x (float): The input value for the wavelet function.

        Returns:
        int: Wavelet function value at x.
        """
        if x > 1 or x < 0:
            return 0
        if x < 0.5:
            return 1
        return -1

    def compute(self, l_index, x):
        """
        Computes the wavelet function for a given level index and input x.

        Parameters:
        l_index (int): The level index.
        x (float): The input value.

        Returns:
        float: The computed wavelet function value.
        """
        if l_index == 0:
            return 1
        level, k = self.seen[l_index]
        return self.power[level] * self.generator((self.power[level] ** 2) * x - k)


class GenerateXTensor:
    """
    This class generates tensors for the wavelet basis.
    """
    def __init__(self):
        self.wavelet_basis = Wavelet()

    def compute_basis_val(self, dim, x, tensor_shape):
        """
        Computes the wavelet basis function for sketching.

        Parameters:
        dim (int): The dimensionality.
        x (array): The input values.
        tensor_shape (list): The shape of the tensor.

        Returns:
        list: Wavelet basis values.
        """
        basis_val = []
        for d in range(dim):
            basis_val.append([self.wavelet_basis.compute(index_l, x[d]) for index_l in range(tensor_shape[d])])
        return basis_val


class Tensor:
    """
    This class handles operations related to tensors in the wavelet basis.
    """
    def __init__(self):
        self.generate_x_tensor = GenerateXTensor()

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
        A_temp = np.zeros(tensor_shape, dtype=np.float32)
        for i in range(len(x_train)):
            basis_val = self.generate_x_tensor.compute_basis_val(dim, x_train[i], tensor_shape)
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
        A_temp = np.zeros(tensor_shape, dtype=np.float32)
        for i in range(len(x_train)):
            x_train[i][0], x_train[i][index] = x_train[i][index], x_train[i][0]
            basis_val = self.generate_x_tensor.compute_basis_val(dim, x_train[i], tensor_shape)
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
        #P_basis = P_transpose[:cur_rank]
        P_basis= P_transpose[:  max(2,cur_rank) ]
        
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
        self.wavelet_basis = Wavelet()

        for dd in range(dim):
            self.P_x_basis.append(self.tensor.compute_range_basis(dd, X_train, tensor_shape, dim))
            self.new_shape.append(len(self.P_x_basis[-1]))

        self.A_predict = np.zeros(self.new_shape, dtype=np.float32)
        for i in range(len(X_train)):
            self.A_predict += self.generate_new_rank_one(X_train[i])
        self.A_predict /= len(X_train)

    def generate_new_rank_one(self, xx):
        """
        Computes the basis function evaluated on a new sample.

        Parameters:
        xx (list): A new sample.

        Returns:
        ndarray: A new rank one tensor.
        """
        new_basis_val = []
        for d in range(self.dim):
            temp = np.array([self.wavelet_basis.compute(l, xx[d]) for l in range(self.tensor_shape[0])])
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
    
        # ans=self.A_predict
        # temp=self.generate_new_rank_one(x_test)

        # for dd in range(self.dim):
        #     ans=(np.tensordot(ans,temp[dd],axes=([0],[0])) )

        # return ans


# List of dimensions to process
dims = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Parameters obtained by cross-validation
step_size_list_new = [0.0, 0.0, 0.25, 0.3, 0.3, 0.35, 0.35, 0.35, 0.4, 0.4, 0.4]
MM_list = [0, 0, 10, 8, 6, 4, 4, 4, 4, 4, 4]
LL_list = [0, 0, 10, 5, 3, 3, 2, 2, 2, 2, 2]

# Number of data points to process
n_data_list = [1000000]
data_dict = {}
VRS_error_list = []


# Iterate over each dimension
for dim in dims:
    # Initialize the sine model for the current dimension
    sin_model = Sin_distribution(n_dims=dim, normal_type="Known")

    # Determine the step size for the current dimension
    step_size = step_size_list_new[dim]

    # Generate test data and normalize
    X_test = np.random.uniform(-1.0, 1.0, size=(1000000, dim)).astype(np.float32)
    X_new = X_test / 2 + 1 / 2

    # Calculate true probability values for test data
    P_true = np.array([sin_model.sinpdf_normal(xx) for xx in X_test])

    # Iterate over the number of data points
    for n_data in n_data_list:
        # Sample training data from the sine model
        x_data = sin_model.sample(n_data, step_size)
        X_train = x_data / 2 + 1 / 2
        print('n_data', n_data)
        print('dimension', dim)

        # Set tensor shape parameters
        MM, LL = MM_list[dim], LL_list[dim]
        tensor_shape = [LL for _ in range(dim)]
        tensor_shape[0] = MM

        # Initialize the VRS model and predict
        model = VRSModel(tensor_shape, dim, X_train)
        y_vrs = np.array([model.predict(xx) for xx in X_new])

        # Calculate VRS error
        err_vrs = np.linalg.norm(y_vrs - P_true * 2**dim, 2) / np.linalg.norm(P_true * 2**dim, 2)
        print('VRS error', err_vrs)
        VRS_error_list.append(err_vrs)



        # Update the data dictionary with error lists
        data_dict['VRS_error_list'] = VRS_error_list


        # Write error data to a JSON file
        with open('data_density_estimation_sin_diffd.json', 'w') as json_file:
            json.dump(data_dict, json_file)

# Print final error lists
print('VRS_error_list', VRS_error_list)























