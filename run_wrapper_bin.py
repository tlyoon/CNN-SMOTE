import subprocess, pandas as pd, os, glob, shutil
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('raw.csv')
keywords = list(set(df.iloc[:,1].tolist()))
pegged=os.getcwd()

script0 = "random_split.py"
print(f"Running script {script0}")
result = subprocess.run(["python", script0 ], capture_output=True, text=True)
print(f"--- STDOUT {script0} ---")
print(result.stdout.strip())
if result.stderr:
    print(f"--- STDERR {script0} ---")
    print(result.stderr.strip())
if result.returncode != 0:
    print(f"Error: {script0} exited with code {result.returncode}")
print('')

##
for keyword in keywords:    
    os.makedirs(keyword, exist_ok=True) 
    py_files = glob.glob('*.py')
    csv_files = glob.glob('*_new.csv')
    txt_files = glob.glob('*.txt')
    for file in py_files + csv_files + [ 'n_oversample.txt' ] + [ 'txt_files' ] + [ 'raw.csv' ]:
        shutil.copy(file, keyword)
    print(f"Copied {len(py_files + csv_files)} *.py and *.csv files to '{keyword}/'")
print('')
##

#####
scripts = ["convert_2_binary.py",
           "preprocess_data_new_train.py",
           "preprocess_data_new_test.py",
           "CNN_SMOTE_train_new.py",
           "CNN_SMOTE_test_new.py",
           "mycnn_model_v2.py",
           "stresstest_new_v2.py"]

for keyword in keywords:    
    os.chdir(keyword)
    ###
    ### work here 
    print(f'I am now in {os.getcwd()}')
    ###    
    for i, file in enumerate(scripts, start=1):
        print(f"Running script {i}: {file}")
        result = subprocess.run(["python", file], capture_output=True, text=True)
        print(f"--- STDOUT ({file}) ---")
        print(result.stdout.strip())
        if result.stderr:
            print(f"--- STDERR ({file}) ---")
            print(result.stderr.strip())
        if result.returncode != 0:
            print(f"Error: {file} exited with code {result.returncode}")
            break  # Optional: stop execution on error
    ###
    ### end work here 
    os.chdir(pegged)
    print('')
