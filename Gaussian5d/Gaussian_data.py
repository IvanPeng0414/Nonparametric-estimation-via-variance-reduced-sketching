#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 17:49:52 2025

@author: yifanpeng
"""

import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal

class GMMDistributionHighDTwoModes:
    def __init__(self, n_dims=5, normal_type="Uniform"):
        """
        High-dimensional Gaussian mixture model with only two modes.

        Parameters
        ----------
        n_dims : int
            Number of dimensions (d).
        normal_type : str
            Type of normalization ("Uniform" or "Known").
            "Uniform" estimates the normalizing constant over [-1,1]^d by Monte Carlo.
        """
        self.n_dims = n_dims
        self.normal_type = normal_type

        # Means: two opposite corners in high dimensions
        self.means_1 = np.full(self.n_dims, -0.5)
        self.means_2 = np.full(self.n_dims,  0.5)
        self.means = [self.means_1, self.means_2]

        # Covariances: isotropic for simplicity (you can customize)
        sigma_1 = 0.25  # std dev for component 1
        sigma_2 = 0.15  # std dev for component 2

        self.cov_1 = (sigma_1 ** 2) * np.eye(self.n_dims)
        self.cov_2 = (sigma_2 ** 2) * np.eye(self.n_dims)
        self.covariances = [self.cov_1, self.cov_2]

        # Mixture weights (two modes)
        self.weight_list = [0.5, 0.5]

        # Box constraints for sampling & normalization domain
        self.lower_bound = np.full(self.n_dims, -1.0)
        self.upper_bound = np.full(self.n_dims,  1.0)

    # Generate samples from the high-D GMM, restricted to the box [-1,1]^d
    def generate_gmm_samples(self, num_samples):
        samples = []
        while len(samples) < num_samples:
            # Choose a component based on weights
            component = np.random.choice(len(self.weight_list), p=self.weight_list)
            # Sample from the chosen Gaussian
            sample = multivariate_normal.rvs(
                mean=self.means[component], 
                cov=self.covariances[component]
            )
            # Keep only if within bounds
            if np.all(sample >= self.lower_bound) and np.all(sample <= self.upper_bound):
                samples.append(sample)
        return np.array(samples)

    # Unnormalized GMM pdf evaluated at x (shape: [N, d])
    def gmm_pdf(self, x):
        """
        Compute the (unnormalized over [-1,1]^d) GMM density at points x.

        Parameters
        ----------
        x : array-like, shape (N, d) or (d,)
            Points where the density is evaluated.

        Returns
        -------
        density : np.ndarray, shape (N,)
        """
        # x = np.atleast_2d(x)
        density = (
            self.weight_list[0] * stats.multivariate_normal.pdf(x, mean=self.means_1, cov=self.cov_1)
            + self.weight_list[1] * stats.multivariate_normal.pdf(x, mean=self.means_2, cov=self.cov_2)
        )
        return density

    def Normal_const(self, normal_type=None):
        """
        Computes the normalization constant over [-1,1]^d.

        Parameters
        ----------
        normal_type : str, optional
            "Known" or "Uniform". If None, uses self.normal_type.

        Returns
        -------
        float
            Approximate integral of gmm_pdf over [-1,1]^d.
        """
        if normal_type is None:
            normal_type = self.normal_type

        if normal_type == "Known":
            # For a true GMM on R^d, each Gaussian is already normalized.
            # But here we want the mass inside [-1,1]^d, which is not analytic in general.
            # So for "Known", you could precompute or store approximations if you want.
            if self.n_dims == 5:
                return 0.95
            else:
                raise NotImplementedError("Known normal_type not implemented for high-d case.")

        if normal_type == "Uniform":
            # Monte Carlo estimate of integral over [-1,1]^d
            num_mc = 800000
            X_total = np.random.uniform(
                low=-1.0, 
                high=1.0, 
                size=(num_mc, self.n_dims)
            )
            # Average pdf over uniform samples, times volume of the box
            avg_density = np.mean(self.gmm_pdf(X_total))
            volume = 2.0 ** self.n_dims
            Nonst = avg_density * volume
            # print('Nonst', Nonst)
            return Nonst

        raise ValueError(f"Unknown normal_type: {normal_type}")

    def gmm_pdf_normal(self, x):
        """
        Normalized pdf over [-1,1]^d.

        Parameters
        ----------
        x : array-like, shape (N, d) or (d,)

        Returns
        -------
        np.ndarray
            Normalized density values at x.
        """
        return self.gmm_pdf(x) / self.Normal_const(self.normal_type)

'''
# Example: 
gmm_hd = GMMDistributionHighDTwoModes(n_dims=5, normal_type="Known")
samples = gmm_hd.generate_gmm_samples(num_samples=1000)  # shape (1000, 10)

# Evaluate (unnormalized) pdf at some points
x_test = np.random.uniform(-1, 1, size=(10, 5))
test_pdf = gmm_hd.gmm_pdf_normal(x_test)
'''