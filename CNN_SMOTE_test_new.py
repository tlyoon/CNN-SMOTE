import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

with open('n_oversample.txt', 'r') as f:
    content = f.read().strip()
    n_oversample = int(content)
print('n_oversample', n_oversample, type(n_oversample))
#n_oversample = 3000  ### default = 500
nk_neighbors = 2    #default = 2

Y=np.load('label_test_new.npy',allow_pickle=True)
X=np.load('config_test_new.npy',allow_pickle=True)

#OVERSAMPLE THE DATASET
sample_strategy= Counter(Y)#creates dict to get the classes of the label file
sample_strategy= dict.fromkeys(sample_strategy,n_oversample) #sets the number of augmented samples
sm = SMOTE(sampling_strategy = sample_strategy,  k_neighbors=nk_neighbors) 
x_res, y_res = sm.fit_resample(X,Y)

print('Before SMOOTING, number of samples in config_test_new.npy:',X.shape[0])
print('After SMOOTING, number of samples in X_test_new.npy:',x_res.shape[0])


np.save("X_test_new",x_res)
print("File X_test_new.npy has been saved")
np.save("Y_test_new",y_res)
print("File Y_test_new.npy has been saved")

