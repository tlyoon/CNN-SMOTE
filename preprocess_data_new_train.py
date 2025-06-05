import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
  
# making data frame from csv file 
# define the csv file to read.

#df = pd.read_csv("sample.csv")
#df = pd.read_csv("Farizah DSC raw data.csv")
#df=pd.read_csv("../DSC VALIDATION NORM.csv")
df=pd.read_csv("train_new.csv")
## Identity of the column (counting from 0 and begining from the left of the input *.csv file)
ncol_HONEYTYPE=2
ncol_SUGARSYRUP=3
ncol_ADULTERANT=4

### specify which ncol to use as classes. Only 1 is permitted.
ncol=1   #ncol_SUGARSYRUP
#ncol=ncol_ADULTERANT
### end of specify which ncol to use as classes. Only 1 is permitted.

## end of Identity of the column (counting from 0 and begining from the left of the input *.csv file)

### the column marking the begining of the features .a.k.a. descriptor .a.k.a. fingerprint
nbf=2

### the column used for identifying the lable of a particular sample (e.g., file name) 
nbl=0



nofrow=df.shape[0]
nofcolumn=df.shape[1]
print('nofrow:',nofrow)
print('nofcolumn:',nofcolumn)

dictrow={};label={};cclass={}
#classes=set(df.iloc[:,1].to_numpy().tolist())
classes=set(df.iloc[:,ncol].to_numpy().tolist())
count=0;fig={}
ax={}

dfnconfig='config_train_new.npy'
dfnlabel='label_train_new.npy'
if os.path.isfile(dfnconfig):
    os.remove(dfnconfig)
if os.path.isfile(dfnlabel):
    os.remove(dfnlabel)    

print(f"Samples in {dfnconfig}:")
#classes.remove('Mix')
for j in classes:
#    print('j:',j)
    #fig[count] = plt.figure(count)
    fig = plt.figure(count)
    #fig.set_size_inches(8,4)
    ax = fig.add_subplot(1, 1, 1)    
    countj=0
    #print('j=',j)
    for i in range(nofrow):
        j=str(j)
        #if df.iloc[i][1] == j:
        if str(df.iloc[i][ncol]) == j:
            countj=countj+1
            #ax[count].set_title(j)
            #print("***",df.iloc[i][0], df.iloc[i][1])
            #line=str(df.iloc[i][0])+ ';' + df.iloc[i][1]
            #print('line',line)
            #label[i]=str(df.iloc[i][0]) + ';' + df.iloc[i][1]
            label[i]=df.iloc[i][nbl]
            
            dictrow[i]=df.iloc[i][nbf:]
            #ax[count].plot(range(len(dictrow[i])),dictrow[i],label=label[i])
            ax.plot(range(len(dictrow[i])),dictrow[i],label=label[i])
            
            dictrow[i]=np.reshape(df.iloc[i][nbf:].to_numpy().tolist(),(1,len(df.iloc[i][nbf:])))
            if os.path.isfile(dfnconfig):
                config = np.load(dfnconfig,allow_pickle=True)
                np.save(dfnconfig,np.vstack((config, dictrow[i])))
            else :
                config=dictrow[i]
                np.save(dfnconfig,dictrow[i])

#            if os.path.isfile(dfnlabel):
#                config = np.load(dfnlabel,allow_pickle=True)
#                np.save(dfnlabel,np.vstack((label, j)))
#            else :
#                datalabel=j
#                np.save(dfnlabel,j)                
            
            datalabel = np.load(dfnlabel,allow_pickle=True) if os.path.isfile(dfnlabel) else [] #get data if exist
            np.save(dfnlabel,np.append(datalabel,j))
            
    print('j,countj:',j,countj)
    count=count+1
    plt.rcParams["figure.figsize"] = (6,4)
    ax.set_title(j+' (No. of samples: '+str(countj)+')')
    ax.set_ylabel('ATR-FTIR reading')
    #ax.set_xlabel('Temperature (degree Celcius)')
    ax.set_xlabel('Wavelength (nm)')
    #plt.legend(loc='lower left',bbox_to_anchor=(-0.5, 0.0, 0.5, 1))
    plt.legend(loc='upper left',bbox_to_anchor=(0.9, 1.2))
    plt.savefig(j+'.png',dpi=150)
    plt.gcf()
    
