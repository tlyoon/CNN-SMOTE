import runpy
loop=1
for i in range(loop):
    runpy.run_path("preprocess_data_v2.py")
    runpy.run_path("CNN_SMOTE.py")  
    runpy.run_path("application_v2.py")       
