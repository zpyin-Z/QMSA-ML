# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""
from os import chdir
import pandas as pd
from scipy.stats import ttest_ind

chdir('dataset')
file_path = 'PCBA-686978.txt'
CF_path = 'CF_PCBA-686978.txt'
data1 = pd.read_csv(file_path, delimiter='\t')
data2 = pd.read_csv(CF_path, delimiter='\t')
# Combine the datasets
data_combined = pd.concat([data1, data2], axis=0)
data_combined.reset_index(drop=True, inplace=True)
# Check for NaN values and drop rows with NaN
initial_shape = data_combined.shape
data_combined.dropna(inplace=True)
dropped_rows = initial_shape[0] - data_combined.shape[0]
print(f"Dropped {dropped_rows} rows with NaN values.")

# Separate features and labels
features = data_combined.iloc[:, 1:-1]
labels = data_combined.iloc[:, -1]
chem_names = data_combined.iloc[:, 0]
# Separate classes
class_0 = features[labels == 0]
class_1 = features[labels == 1]

p_values = {}
feature_data = {}

# t-tests
for feature in features.columns:
    stat, p = ttest_ind(class_0[feature], class_1[feature], equal_var=False)
    mean_0 = class_0[feature].mean()
    mean_1 = class_1[feature].mean()
    sd_0 = class_0[feature].std()
    sd_1 = class_1[feature].std()
    p_values[feature] = (p, mean_0, sd_0, mean_1, sd_1)
    feature_data[feature] = {
        'class_0_values': class_0[feature].tolist(),
        'class_1_values': class_1[feature].tolist()
    }

with open('t_test_p_values.txt', 'w') as f:
    for feature, (p, mean_0, sd_0, mean_1, sd_1) in p_values.items():
        f.write(f"{feature}:\n")
        f.write(f"  p-value = {p:.4e}\n")
        f.write(f"  Class 0 - Mean: {mean_0:.4f}, SD: {sd_0:.4f}\n")
        f.write(f"  Class 1 - Mean: {mean_1:.4f}, SD: {sd_1:.4f}\n\n")

feature_data_df = pd.DataFrame({
    f"{feature}_Class 0 values": data['class_0_values']
    for feature, data in feature_data.items()
}).join(pd.DataFrame({
    f"{feature}_Class 1 values": data['class_1_values']
    for feature, data in feature_data.items()
}))

feature_data_df.to_csv('feature_data.csv', index=False)

print("Results saved to 't_test_p_values.txt' and 'feature_data.csv'.")
