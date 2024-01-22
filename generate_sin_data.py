#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

class Sin_distribution():
    """
    This class implements a sine distribution, allowing for sampling from this distribution
    using the Metropolis-Hastings algorithm.
    """

    def __init__(self, n_dims, normal_type="Uniform"):
        """
        Initializes the SinDistribution class.

        Parameters:
        n_dims (int): The number of dimensions.
        normal_type (str): The type of normalization to use. Options are "Uniform" or "Known".
        """
        self.n_dims = n_dims  
        self.normal_type = normal_type

    def sinpdf(self, x):
        """
        Computes the sine probability density function.

        Parameters:
        x (array): An array of values.

        Returns:
        float: The sine of the mean of x plus pi/4, shifted up by 1.
        """
        return np.sin(np.pi * np.mean(x) + np.pi / 4) + 1
        # Note: This PDF is not normalized.

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
            rec = [0] * 11
            rec[1:11] = [0, 1.2855, 1.4007, 1.4644, 1.5070, 1.5363, 1.5582, 1.5749, 1.5890, 1.5996]
            return rec[self.n_dims] * 2 ** self.n_dims

        if normal_type == "Uniform":
            # Estimate normalization constant for a Uniform distribution
            Normal_const = 0
            X_total = np.random.uniform(low=-1.0, high=1.0, size=[1000000, self.n_dims])

            for i in range(np.shape(X_total)[0]):
                Normal_const += self.sinpdf(X_total[i, :]) / np.shape(X_total)[0]
            return Normal_const * 2 ** self.n_dims

    def sinpdf_normal(self, x):
        """
        Normalizes the sine probability density function.

        Parameters:
        x (array): An array of values.

        Returns:
        float: The normalized sine probability density.
        """
        return self.sinpdf(x) / self.Normal_const(self.normal_type)

    def sample(self, num_iter, step_size):
        """
        Generates samples from the sine distribution using the Metropolis-Hastings algorithm.

        Parameters:
        num_iter (int): The number of iterations to perform.
        step_size (float): The step size for the proposal distribution.

        Returns:
        array: An array of accepted samples.
        """
        x_init = np.zeros((self.n_dims,))

        def proposal_sample(x):
            """
            Proposal distribution for Metropolis-Hastings sampling. Uses a uniform distribution.

            Parameters:
            x (array): The current sample.

            Returns:
            array: The proposed next sample.
            """
            x_prime = x + np.random.uniform(-step_size, step_size, size=x.shape)
            
            # Ensure the proposed sample is within the bounds [-1, 1]
            if np.min(x_prime) < -1 or np.max(x_prime) > 1:
                return x
            else:
                return x_prime

        def MHsampling(target_pdf, proposal_sample, x_init, num_iter):
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
            for iteration in range(2 * num_iter):
                x_new = proposal_sample(x)
                # Compute the acceptance probability
                alpha = min(1, target_pdf(x_new) / target_pdf(x))
                if np.random.rand() < alpha:
                    x = x_new
                # Accept the second half of samples
                if iteration >= num_iter:
                    accepted.append(x)
            return np.array(accepted)

        samples = MHsampling(self.sinpdf, proposal_sample, x_init, num_iter)
        return samples

## test    
dim = 5
sin_model = Sin_distribution(n_dims=dim, normal_type="Known")