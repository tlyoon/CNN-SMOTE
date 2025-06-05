from keras import models 
from keras import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os, re
import seaborn as sns
import matplotlib.pyplot as plt

############LOAD Validating DATASET FORM FILE##################
X_test = np.load('X_test_new.npy', allow_pickle=True)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
Y_test = np.load('Y_test_new.npy', allow_pickle=True)

#####ENCODE LOADED FILE TO WORK WITH NN OUTPUT#######
encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y = encoder.transform(Y_test)
dummy_y = utils.to_categorical(encoded_Y)

##########LOAD TRAINED MODEL##################
#fold_no = 0 #### choose which saved model to use

##
with open("mycnn_model_v2.py", "r") as f:
    lines = f.readlines()

# Try to find the n_splits argument
for line in lines:
    match = re.search(r"StratifiedKFold\s*\(\s*n_splits\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*|\d+)", line)
    if match:
        value = match.group(1)
        break

# Now resolve the value
if value.isdigit():
    print(f"Value of n_splits: {value}")
else:
    # Look for variable assignment earlier in the file
    pattern = re.compile(rf"{value}\s*=\s*(\d+)")
    for line in lines:
        assign_match = pattern.search(line)
        if assign_match:
            print(f"Value of n_splits: {assign_match.group(1)}")
            n_split = int(assign_match.group(1))
            break
    else:
        print(f"Could not resolve value of variable '{value}'")


for fold_no in range(n_split):
    ##
    mpath = f"model_fold{fold_no}.h5" ## 'models/fold_5.keras'
    model = models.load_model(mpath)
    y_test = model.predict(X_test)
    yhat = np.argmax(y_test, axis=-1).astype('int')
    acc = accuracy_score(encoded_Y, yhat) * 100
    print('Accuracy: %.3f' % acc)
    
    ##########CONFUSION MATRIX & CLASSIFICATION REPORT##################
    cm = confusion_matrix(encoded_Y, yhat)
    report = classification_report(encoded_Y, yhat, target_names=encoder.classes_)
    
    print('Confusion Matrix')
    print(cm)
    print('Classification Report')
    print(report)
    
    ##########SAVE REPORT TO FILE##################
    with open(f'REPORT_{fold_no}.txt', 'a') as file:
        file.write('Confusion Matrix\n' + str(cm) + '\n' + '\nClassification Report \n' + report + '\n')
    
    ########## PLOT CONFUSION MATRIX ##################
    plt.figure(figsize=(9, 7))
    
    # Ensure correct class names
    target_names = list(encoder.classes_)  # Convert to list for indexing
    
    # Compute confusion matrix
    cm = confusion_matrix(encoded_Y, yhat)
    
    # Plot heatmap with correctly mapped labels
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=True, linewidths=1, linecolor="black")
    
    ax.set_title('Classification Confusion Matrix\n')
    ax.set_xlabel('Predicted Classification')
    ax.set_ylabel('Ground Truth Classification')
    
    # Correctly map class labels
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names, rotation=0)
    
    # Save CM plot
    path = os.path.join(os.getcwd(), f'cm_plot_{fold_no}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()