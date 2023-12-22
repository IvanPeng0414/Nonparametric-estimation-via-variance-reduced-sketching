#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

#import matplotlib.pyplot as plt


class Sin_distribution():
    ##1. initialization
    def __init__(self, n_dims, lamda, beta, h, normal_type="Uniform"):
        self.n_dims = n_dims  
        self.lamda = lamda 
        self.beta = beta
        self.h = h
        self.normal_type = normal_type
    
    
    def sinpdf(self,x):

        #return np.sin(np.sum(x)) + 1
        return np.sin(np.pi*np.mean(x)+np.pi/4) + 1
        
        # without normalization yet

    
    
    def Normal_const(self,normal_type):
        if normal_type=='known':
            #np.sin(np.sum(x))+1
            rec=[0]*11
            #rec[1]=2.188838122556386
            rec[1]=0
            rec[2]=1.2855363994141558
            rec[3]=1.4007828994218285
            rec[4]=1.464490421436803
            rec[5]=1.5070709600953234 
            rec[6]=1.536330450295231
            rec[7]=1.5582564976796813
            rec[8]=1.5749823100508795
            rec[9]=1.589034729690488
            rec[10]=1.599601627957483
            #####
            #np.sin(pi*np.sum(x)/2)+1
            #rec[1]=2.188838122556386
            #rec[2]=1.8105917401146714
            #rec[1]=1.49675
            #rec[3]=1.516050053510251
            #rec[2]=20
            #rec[4]=1
            #rec[5]=0.5816639369330584
            #rec[6]=0.46758027486591575
            #rec[7]=0.6610216727685047
            #rec[8]=1
            #rec[9]=1.2750539055982193
            #rec[10]=1.3501491981707892

            return rec[self.n_dims]*2**self.n_dims
            


        
        if normal_type == "Uniform":         
            Normal_const = 0
            X_total = np.random.uniform(low=-1.0,high=1.0,size=[1000000, self.n_dims])

            for i in range(np.shape(X_total)[0]):
                Normal_const += self.sinpdf(X_total[i,:])/np.shape(X_total)[0]                
            return Normal_const*2**self.n_dims
    
    def sinpdf_normal(self,x):
        return self.sinpdf(x)/self.Normal_const(self.normal_type)
        

    ##3. sample
    def sample(self, num_iter, step_size):

        x_init = np.zeros((self.n_dims,))
        
        def proposal_sample(x):
            ## sampling kernel in MH sampling, use uniform distribution
            x_prime = x + np.random.uniform(-step_size, step_size, size=x.shape)
            
            if np.min(x_prime) < -1 or np.max(x_prime) > 1:
                return x
            else:
                return x_prime
            
        def MHsampling(target_pdf, proposal_sample, x_init, num_iter):
            ## Metropolis–Hastings algorithm                   
            x = x_init
            accepted = []        
            for iteration in range(2*num_iter):
                x_new = proposal_sample(x)                
                # Compute acceptance probability
                alpha = min(1, target_pdf(x_new)/target_pdf(x) )
                if np.random.rand() < alpha:
                    x = x_new
                if iteration >= num_iter:
                    accepted.append(x)
                    
            return np.array(accepted)


        
        samples = MHsampling(self.sinpdf, proposal_sample, x_init, num_iter)

        return samples
    

###############  
dim = 5
sin_model = Sin_distribution(n_dims=dim, lamda=1.0, beta=1.0, h=1.0, normal_type="known")

