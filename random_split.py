import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV
filein='raw.csv'

df = pd.read_csv(filein)
print(f'The number of samples in {filein}:',df.shape[0])

# Container for train and test DataFrames
train_list = []
test_list = []

# Set the split ratio
test_ratio = 0.5  # 50% train, 50% test per Type

# Group by each unique Type and split each group
for type_class, group in df.groupby('Type'):
    train, test = train_test_split(group, test_size=test_ratio, random_state=None, shuffle=True)
    train_list.append(train)
    test_list.append(test)

# Concatenate the per-Type splits
train_df = pd.concat(train_list).sample(frac=1).reset_index(drop=True)  # shuffle again
test_df = pd.concat(test_list).sample(frac=1).reset_index(drop=True)    # shuffle again

# Save to CSV
train_df.to_csv('train_new.csv', index=False)
test_df.to_csv('test_new.csv', index=False)

print(f'The number of samples in train_new.csv:',train_df.shape[0])
print(f'The number of samples in test_new.csv:',test_df.shape[0])
print("train_new.csv and test_new.csv have been generated with approximately equal samples per Type.")
print('')
