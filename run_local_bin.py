import subprocess, pandas as pd, os, glob, shutil
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

#####
scripts = ["random_split.py",
           "convert_2_binary.py",
           "preprocess_data_new_train.py",
           "preprocess_data_new_test.py",
           "CNN_SMOTE_train_new.py",
           "CNN_SMOTE_test_new.py",
           "mycnn_model_v2.py",
           "stresstest_new_v2.py"]

print(f'I am now in {os.getcwd()}')
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
