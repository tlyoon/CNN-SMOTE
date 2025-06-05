import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

Y=np.load('label_train_new.npy',allow_pickle=True)
X=np.load('config_train_new.npy',allow_pickle=True)
#OVERSAMPLE THE DATASET
sample_strategy= Counter(Y)#creates dict to get the classes of the label file
sample_strategy= dict.fromkeys(sample_strategy,500) #sets the number of augmented samples
sm = SMOTE(sampling_strategy = sample_strategy,  k_neighbors=2) 
x_res, y_res = sm.fit_resample(X,Y)
np.save("X_train_new",x_res)

print('Before SMOOTING, number of samples in config_train_new.npy:',X.shape[0])
print('After SMOOTING, number of samples in X_train_new.npy:',x_res.shape[0])


print("File X_train_new.npy has been saved")
np.save("Y_train_new",y_res)
print("File Y_train_new.npy has been saved")
