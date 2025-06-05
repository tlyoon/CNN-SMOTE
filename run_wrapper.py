import subprocess

file="random_split.py"
result = subprocess.run(["python", file], capture_output=True, text=True)
print(f'1. {file}',result.stdout)


file="preprocess_data_new_train.py"
result = subprocess.run(["python", file], capture_output=True, text=True)
print(f'2. {file}',result.stdout)

file="CNN_SMOTE_train_new.py"
result = subprocess.run(["python", file], capture_output=True, text=True)
print(f'3. {file}',result.stdout)

file="preprocess_data_new_test.py"
result = subprocess.run(["python", file], capture_output=True, text=True)
print(f'4. {file}',result.stdout)

file="CNN_SMOTE_test_new.py"
result = subprocess.run(["python", file], capture_output=True, text=True)
print(f'5. {file}',result.stdout)

'''
file="mycnn_model_v2.py"
result = subprocess.run(["python", file], capture_output=True, text=True)
print(f'6. {file}',result.stdout)

file="stresstest_new.py"
result = subprocess.run(["python", file], capture_output=True, text=True)
print(f'7. {file}',result.stdout)
'''
