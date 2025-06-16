import subprocess

scripts = [
    "random_split.py",
    "preprocess_data_new_train.py",
    "CNN_SMOTE_train_new.py",
    "mycnn_model_v2.py",
    "preprocess_data_new_test.py",
    "CNN_SMOTE_test_new.py",
    "stresstest_new_v2.py"
]

for i, file in enumerate(scripts, start=1):
    print(f"\n▶️ Running script {i}: {file}")
    result = subprocess.run(["python", file], capture_output=True, text=True)

    print(f"--- STDOUT ({file}) ---")
    print(result.stdout.strip())

    if result.stderr:
        print(f"--- STDERR ({file}) ---")
        print(result.stderr.strip())

    if result.returncode != 0:
        print(f"❌ Error: {file} exited with code {result.returncode}")
        break  # Optional: stop execution on error
