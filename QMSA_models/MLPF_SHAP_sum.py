# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""

import pandas as pd
import numpy as np
from os import chdir
import os

def calculate_mean_std_feature_importance(model_name, seed_list):
    feature_importance_list = []
    for seed in seed_list:
        file_path = f"Feature_importance_SHAP_{model_name}_PCBA-686978_split{seed}.csv"
        if os.path.exists(file_path):
            importance_data = pd.read_csv(file_path)
            feature_importance_list.append(importance_data['Importance'].values)
        else:
            print(f"Warning: {file_path} not found.")
    feature_importance_array = np.array(feature_importance_list)
    mean_importances = np.mean(feature_importance_array, axis=0)
    std_importances = np.std(feature_importance_array, axis=0)

    feature_names = importance_data['Feature'].values  # Get feature names from the last loaded file
    result_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Importance': mean_importances,
        'Standard Deviation': std_importances
    })
    return result_df

def main():
    chdir('dataset\\data')
    seed_list = [13, 17, 21, 24, 29]
    model_names = ['Random Forest', 'XGBoost']
    for model_name in model_names:
        result_df = calculate_mean_std_feature_importance(model_name, seed_list)
        result_df.to_csv(f"Mean_Std_Feature_Importance_{model_name}.csv", index=False)

if __name__ == "__main__":
    main()
