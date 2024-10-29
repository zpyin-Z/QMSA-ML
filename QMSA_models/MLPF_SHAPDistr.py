# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""

from os import chdir
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from scipy.stats import spearmanr

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    data.dropna(inplace=True)
    X = data.iloc[:, 1:-1]  # Features
    y = data.iloc[:, -1]    # Labels
    feature_names = X.columns.tolist()  #Feature names
    scaler = StandardScaler()  # Normalize
    X_normalized = scaler.fit_transform(X)
    X = X.reset_index(drop=True)
    return X_normalized, y, feature_names, X

# Oversampling
def balance_data(X_train, y_train, method='random'):
    if method == 'random':
        oversampler = RandomOverSampler(random_state=42)
    elif method == 'smote':
        oversampler = SMOTE(random_state=42)
    elif method == 'borderline_smote':
        oversampler = BorderlineSMOTE(random_state=42)
    else:
        raise ValueError("Invalid oversampling method.")
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def save_data_to_csv(data, filename):
    directory = "data"
    if not os.path.exists(directory):
        os.makedirs(directory)
    pd.DataFrame(data).to_csv(os.path.join(directory, filename), index=False)

# Spearman correlation
def spearman_correlation_analysis(shap_data, original_data, feature_names, model_name, original_name):
    correlation_results = {}
    for feature in feature_names:
        r, p = spearmanr(shap_data[feature], original_data[feature])
        correlation_results[feature] = {'r': r, 'p': p}

    with open(f'Spearman_correlation_{model_name}_{original_name}.txt', 'w') as f:
        for feature, results in correlation_results.items():
            f.write(f"{feature}: r = {results['r']:.4f}, p = {results['p']:.4e}\n")

# SHAP values
def calculate_and_save_shap(model, X_train, original_X_train, feature_names, model_name, original_name):
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):  # Binary classification
            shap_values_class_1 = shap_values[1]  # Get SHAP values for the positive class
        else:
            shap_values_class_1 = shap_values  # Use the full SHAP values
        # SHAP values and original feature values
        shap_data = pd.DataFrame(shap_values_class_1, columns=feature_names)
        original_data = pd.DataFrame(original_X_train, columns=feature_names)

        combined_data = pd.concat([shap_data, original_data], axis=1)
        save_data_to_csv(combined_data, f"Shap_and_feature_{model_name}_{original_name}.csv")

        spearman_correlation_analysis(shap_data, original_data, feature_names, model_name, original_name)

# Train models and calculate SHAP values
def train_and_calculate(X_train, y_train, original_X_train, feature_names, original_name):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=20, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Evaluating {name}...")
        calculate_and_save_shap(model, X_train, original_X_train, feature_names, name, original_name)

def main():
    chdir('dataset')
    seed_list = [13, 17, 21, 24, 29]
    for i in seed_list:
        print(f"Processing split {i}...")
        train_file = f'PCBA-686978_train{i}.txt'
        test_file = f'PCBA-686978_test{i}.txt'
        X_train, y_train, feature_names, original_X_train = load_and_preprocess_data(train_file)
        train_and_calculate(X_train, y_train, original_X_train, feature_names, f'PCBA-686978_split{i}')

if __name__ == "__main__":
    main()
