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
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
import numpy as np

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    data.dropna(inplace=True)
    X = data.iloc[:, 1:-1]  # Features
    y = data.iloc[:, -1]    # Labels
    feature_names = X.columns.tolist()
    scaler = StandardScaler()  # Normalize
    X_normalized = scaler.fit_transform(X)
    return X_normalized, y, feature_names

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

def plot_feature_importance_shap(shap_values, feature_names, model_name, original_name):
    # Calculate mean absolute SHAP values for feature importance
    importance = np.abs(shap_values).mean(axis=0)
    feature_importance_data = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    save_data_to_csv(feature_importance_data, f"Feature_importance_SHAP_{model_name}_{original_name}.csv")
    plt.figure()
    plt.title(f"Feature Importance (SHAP) for {model_name}")
    plt.barh(feature_names, importance, align='center')
    plt.xlabel("Mean Absolute SHAP Value")
    plt.savefig(f"Feature_importance_SHAP_{model_name}_{original_name}.svg", bbox_inches='tight', transparent=False, dpi=800)
    plt.show()
    plt.close()

# SHAP values
def calculate_and_plot_shap(model, X_train, feature_names, model_name, original_name):
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):  # Binary classification
            shap_values_class_1 = shap_values[1]  # Get SHAP values for the positive class
        else:
            shap_values_class_1 = shap_values  # Use the full SHAP values
        shap_data = pd.DataFrame(shap_values_class_1, columns=feature_names)
        save_data_to_csv(shap_data, f"Shap_values_{model_name}_{original_name}.csv")

        feature_data = pd.DataFrame(X_train, columns=feature_names)
        save_data_to_csv(feature_data, f"Feature_values_{model_name}_{original_name}.csv")

        plt.figure()
        shap.summary_plot(shap_values_class_1, X_train, feature_names=feature_names, show=False, sort=False)
        plt.title(f"SHAP Summary Plot for {model_name}")
        plt.savefig(f"Shap_summary_{model_name}_{original_name}.svg", bbox_inches='tight', transparent=False, dpi=800)
        plt.show()
        plt.close()
        plot_feature_importance_shap(shap_values_class_1, feature_names, model_name, original_name)
    else:
        print(f"Skipping SHAP calculation for {model_name} (not applicable)")

# Train models and visualize feature importance and SHAP values
def train_and_visualize(X_train, y_train, X_test, y_test, feature_names, original_name):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=20, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Evaluating {name}...")
        calculate_and_plot_shap(model, X_train, feature_names, name, original_name)

def main():
    chdir('dataset')
    seed_list = [13, 17, 21, 24, 29]
    for i in seed_list:
        print(f"Processing split {i}...")
        train_file = f'PCBA-686978_train{i}.txt'
        test_file = f'PCBA-686978_test{i}.txt'
        X_train, y_train, feature_names = load_and_preprocess_data(train_file)
        X_test, y_test, _ = load_and_preprocess_data(test_file)
        X_resampled, y_resampled = balance_data(X_train, y_train, method='random')
        train_and_visualize(X_resampled, y_resampled, X_test, y_test, feature_names, f'PCBA-686978_split{i}')

if __name__ == "__main__":
    main()

