# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""
from os import chdir
import pandas as pd
from sklearn.model_selection import train_test_split

chdir('dataset')
file_path = 'PCBA-686978.txt'
CF_path = 'CF_PCBA-686978.txt'
data1 = pd.read_csv(file_path, delimiter='\t')
data2 = pd.read_csv(CF_path, delimiter='\t')
# Combine the datasets
data_combined = pd.concat([data1, data2], axis=0)
data_combined.reset_index(drop=True, inplace=True)
# Separate features, labels, and chemical names
chem_names = data_combined.iloc[:, 0]
features = data_combined.iloc[:, 1:-1]
labels = data_combined.iloc[:, -1]

seed_list = [13,17,21,24,29]
for i in seed_list:
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test, train_chem_names, test_chem_names = train_test_split(
        features, labels, chem_names, test_size=0.2, stratify=labels, random_state=i)
    # Combine sample names, features, and labels for train and test datasets
    train_data = pd.concat([train_chem_names, X_train, y_train], axis=1)
    test_data = pd.concat([test_chem_names, X_test, y_test], axis=1)
    # File paths for train and test datasets
    train_output_path = f'PCBA-686978_train{i}.txt'
    test_output_path = f'PCBA-686978_test{i}.txt'
    #
    train_data.to_csv(train_output_path, sep='\t', index=False)
    test_data.to_csv(test_output_path, sep='\t', index=False)
    print(f"Train dataset {i} saved to {train_output_path}")
    print(f"Test dataset {i} saved to {test_output_path}")
