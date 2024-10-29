# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""
import numpy as np 
import pandas as pd
from os import chdir

chdir('dataset')
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    initial_shape = data.shape
    data.dropna(inplace=True)
    dropped_rows = initial_shape[0] - data.shape[0]
    print(f"Dropped {dropped_rows} rows with NaN values from {file_path}")
    return data

seed_list = [13, 17, 21, 24, 29]
results = []

for seed in seed_list:
    training_data = load_data(f'PCBA-686978_train{seed}.txt')
    validation_data = load_data(f'PCBA-686978_test{seed}.txt')
    features_training = training_data.iloc[:, 1:-1].values
    features_validation = validation_data.iloc[:, 1:-1].values
    # Calculate centroid of training data
    centroid = np.mean(features_training, axis=0)
    # Calculate Euclidean distances from validation data to the centroid
    distances_to_centroid = np.linalg.norm(features_validation - centroid, axis=1)
    # Thresholds
    euclidean_threshold_95 = np.percentile(distances_to_centroid, 95)
    euclidean_threshold_3avg = 3 * np.mean(distances_to_centroid)
    # Identify compounds inside the AD using both thresholds
    validation_data['AD_Euclidean_95'] = distances_to_centroid <= euclidean_threshold_95
    validation_data['AD_Euclidean_3avg'] = distances_to_centroid <= euclidean_threshold_3avg
    # Calculate the percentage of compounds inside AD for both methods
    total_compounds = validation_data.shape[0]
    true_inside_ad_euclidean_95 = validation_data['AD_Euclidean_95'].sum()
    true_inside_ad_euclidean_3avg = validation_data['AD_Euclidean_3avg'].sum()
    results.append({
        'Seed': seed,
        'Total Compounds': total_compounds,
        'AD_Euclidean_95%': true_inside_ad_euclidean_95 / total_compounds * 100 if total_compounds > 0 else 0,
        'AD_Euclidean_3Avg': true_inside_ad_euclidean_3avg / total_compounds * 100 if total_compounds > 0 else 0
    })

results_df = pd.DataFrame(results)
print(results_df)
