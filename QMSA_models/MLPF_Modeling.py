# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""
from os import chdir
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
import numpy as np
from sklearn.metrics import matthews_corrcoef

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    initial_shape = data.shape
    data.dropna(inplace=True)
    dropped_rows = initial_shape[0] - data.shape[0]
    print(f"Dropped {dropped_rows} rows with NaN values from {file_path}")
    X = data.iloc[:, 1:-1]  # Features
    y = data.iloc[:, -1]    # Labels
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, y

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

# Evaluate
def evaluate_model(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    conf_matrix = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'report': report,
        'confusion_matrix': conf_matrix
    }

# Train
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=42),
        'SVM': SVC(C=100, kernel="rbf", probability=True, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=20, random_state=42),
        'MLP': MLPClassifier(alpha=0.001, hidden_layer_sizes=(100, 100), learning_rate="constant", activation="relu", solver="adam", max_iter=10000, random_state=42)
    }
    # Train individual model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = {
            'train': evaluate_model(model, X_train, y_train),
            'test': evaluate_model(model, X_test, y_test)
        }
    # Stacking classifier
    estimators = [(name, model) for name, model in models.items()]
    stacking_model = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(random_state=42), cv=5
    )
    # Train stacking model
    print("Training Stacking Classifier...")
    stacking_model.fit(X_train, y_train)
    # Evaluate stacking model
    results['Stacking'] = {
        'train': evaluate_model(stacking_model, X_train, y_train),
        'test': evaluate_model(stacking_model, X_test, y_test)
    }
    return results

# Plot ROC
def plot_roc_curves(results, dataset_type):
    plt.figure()
    for name, result in results.items():
        plt.plot(result[dataset_type]['fpr'], result[dataset_type]['tpr'], label=f'{name} (AUC = {result[dataset_type]["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({dataset_type})')
    plt.legend(loc="lower right")
    plt.show()

# The detailed results
def print_results(results):
    for name, result in results.items():
        print(f'===== {name} =====')
        for dataset_type in ['train', 'test']:
            print(f'--- {dataset_type} ---')
            print(f'Precision: {result[dataset_type]["precision"]:.2f}')
            print(f'Recall: {result[dataset_type]["recall"]:.2f}')
            print(f'F1-Score: {result[dataset_type]["f1"]:.2f}')
            print(f'ROC AUC: {result[dataset_type]["roc_auc"]:.2f}')
            print(f'MCC: {result[dataset_type]["mcc"]:.2f}')
            print('Classification Report:')
            print(result[dataset_type]['report'])
            print('Confusion Matrix:')
            print(result[dataset_type]['confusion_matrix'])
            print('\n')

def save_results(results, file_path):
    with open(file_path, 'w') as file:
        for name, result in results.items():
            for dataset_type in ['train', 'test']:
                file.write(f"Model: {name} ({dataset_type})\n")
                file.write(f"Precision: {result[dataset_type]['precision']:.2f}\n")
                file.write(f"Recall: {result[dataset_type]['recall']:.2f}\n")
                file.write(f"F1-Score: {result[dataset_type]['f1']:.2f}\n")
                file.write(f"ROC AUC: {result[dataset_type]['roc_auc']:.2f}\n")
                file.write(f"MCC: {result[dataset_type]['mcc']:.2f}\n")
                file.write('Classification Report:\n')
                file.write(result[dataset_type]['report'] + '\n')
                file.write('Confusion Matrix:\n')
                file.write(np.array2string(result[dataset_type]['confusion_matrix']) + '\n')
                file.write('FPR:\n')
                file.write(','.join(map(str, result[dataset_type]['fpr'])) + '\n')
                file.write('TPR:\n')
                file.write(','.join(map(str, result[dataset_type]['tpr'])) + '\n')
                file.write('\n')

def main():
    chdir('dataset')
    seed_list = [13,17,21,24,29]
    for i in seed_list:
        print(f"Processing split {i}...")
        train_file = f'PCBA-686978_train{i}.txt'
        test_file = f'PCBA-686978_test{i}.txt'
        X_train, y_train = load_and_preprocess_data(train_file)
        X_test, y_test = load_and_preprocess_data(test_file)
        # Balance
        X_resampled, y_resampled = balance_data(X_train, y_train, method='random')
        # Train and evaluate models
        results = train_and_evaluate_models(X_resampled, y_resampled, X_test, y_test)
        # Plot ROC
        for dataset_type in ['train', 'test']:
            plot_roc_curves(results, dataset_type)
        print_results(results)
        save_results(results, f"PCBA-686978_model_result_split_{i}.txt")

if __name__ == "__main__":
    main()
