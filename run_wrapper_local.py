import subprocess, os, shutil
from pathlib import Path

run_nmax = 20

hostname = subprocess.check_output("hostname", shell=True).decode().strip()
source_dir = os.getcwd()
print('source_dir', source_dir)
for i in range(1, run_nmax + 1):
    destination_dir = os.path.join(source_dir, 'run_' + str(i))
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate over items in the source directory
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        destination_item = os.path.join(destination_dir, item)
        # Check if the item is a file (not a directory)
        if os.path.isfile(source_item):
            # Copy the file to the destination directory
            shutil.copy2(source_item, destination_item)
    os.chdir(destination_dir)
    #### do work in destination_dir
    file = 'run_local_bin.py' ##'run_once.py'
    print(f'Now running python {file} in {os.getcwd()} in {hostname}')
    
    result = subprocess.run(["python", file], capture_output=True, text=True)

    print(f"--- STDOUT ({file}) ---")
    print(result.stdout.strip())

    if result.stderr:
        print(f"--- STDERR ({file}) ---")
        print(result.stderr.strip())

    if result.returncode != 0:
        print(f"❌ Error: {file} exited with code {result.returncode}")
        break  # Optional: stop execution on error

    print('')
    #### finish doing work in destination_dir
    os.chdir(source_dir)


