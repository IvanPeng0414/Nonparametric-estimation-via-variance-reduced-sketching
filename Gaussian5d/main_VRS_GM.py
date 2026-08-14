import numpy as np
import time
from Gaussian_data import GMMDistributionHighDTwoModes


from tensor_estimate  import vrs_prediction
  
## wait for the polish....

dim=5
N_train= 40000 

##############tuning parameter selection
MM=15 
LL = 3 

print(MM,LL)
tensor_shape=[LL for _ in range(dim)]
tensor_shape[0]=MM

tensor_shape_full=[MM for _ in range(dim)]

#########################################

lr_rec=0
kde_rec=0



gmm_hd = GMMDistributionHighDTwoModes(n_dims=dim, normal_type="Known")
for rr in range(1):
    
    

    X_train = gmm_hd.generate_gmm_samples(num_samples=N_train)  # shape (1000, 10)

    N_test=100000
    X_test = gmm_hd.generate_gmm_samples(num_samples=N_test)  # shape (1000, 10)

    # y_true = np.array([distribution.density_value(xx) for xx in X_test])
    y_true = gmm_hd.gmm_pdf_normal(X_test)

    #############density transform 
    start  = time.time()
    vrs_model=vrs_prediction(tensor_shape, dim, MM, X_train)
    y_lr= vrs_model.predict(X_test)   
    lr_err = np.linalg.norm(y_lr - y_true,2)/np.linalg.norm(y_true,2)
    print('lr error', lr_err)

    tol = 1e-16 #1e-7
    y_test_adjust = []
    y_true_adjust = []
    for i in range(len(X_test)):
        if y_lr[i] > tol:
            y_test_adjust.append(np.log(y_lr[i]))
            y_true_adjust.append(np.log(y_true[i]))

    VRS_KL = np.mean(y_true_adjust) - np.mean(y_test_adjust)    
    print('VRS_KL', VRS_KL)

    end  = time.time()
    print('lr time', end-start)
    
    start  = time.time()
    vrs_model_full=vrs_prediction(tensor_shape_full, dim, MM, X_train)
    y_lr_full = vrs_model_full.predict(X_test)
    lr_full_err = np.linalg.norm(y_lr_full - y_true,2)/np.linalg.norm(y_true,2)
    print('Tucker full error', lr_full_err)

    tol = 1e-16
    y_test_adjust = []
    y_true_adjust = []
    for i in range(len(X_test)):
        if y_lr_full[i] > tol:
            y_test_adjust.append(np.log(y_lr_full[i]))
            y_true_adjust.append(np.log(y_true[i]))

    full_KL = np.mean(y_true_adjust) - np.mean(y_test_adjust)    
    print('full_KL', full_KL)

    end  = time.time()
    print('Tucker full time', end-start)  


