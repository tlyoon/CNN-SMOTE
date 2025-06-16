import pandas as pd, os, shutil, glob, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
    
def convert_2_binary(keyword):
    f1= 'train_new.csv'
    df = pd.read_csv(f1)       
    df['Type'] = df['Type'].apply(lambda x: x if x == keyword else f'others')    
    output_filename = f'{keyword}_{f1}'
    df.to_csv(output_filename, index=False)
    print(f"Converted {f1} into {output_filename}")
    shutil.copy(f1,'orig_'+f1)
    print(f"Copied {f1} to orig_{f1}")
    
    shutil.move(output_filename,f1)
    print(f"Moved {output_filename} to {f1}")
    
    f2= 'test_new.csv'
    df = pd.read_csv(f2)       
    df['Type'] = df['Type'].apply(lambda x: x if x == keyword else f'others')    
    output_filename = f'{keyword}_{f2}'
    df.to_csv(output_filename, index=False)
    print(f"Converted {f2} into {output_filename}")
    shutil.copy(f2,'orig_'+f2)
    print(f"Copied {f2} to orig_{f2}")
    shutil.move(output_filename,f2)
    print(f"Moved {output_filename} to {f2}")

keyword = os.path.basename(os.getcwd())
convert_2_binary(keyword)

'''    
df = pd.read_csv('raw.csv')
keywords = list(set(df.iloc[:,1].tolist()))
for keyword in keywords:
    #print(f'converting raw.csv into csv binary version with keyword {keyword}, {keyword}_bin.csv')
    convert_2_binary(keyword)
    os.makedirs(keyword, exist_ok=True)  # Create directory if it doesn't exist
    # Find all .py files in the current directory
    py_files = glob.glob('*.py')
    # Copy each .py file to the output directory
    for file in py_files:
        shutil.copy(file, keyword)
    print(f"Copied {len(py_files)} *.py files to '{keyword}/'")
    targetpath = os.path.join(keyword,'raw.csv')
    shutil.move(f'{keyword}_bin.csv', targetpath)
    print(f"Copied {keyword}_bin.csv to {targetpath}")
    print(' ')
'''