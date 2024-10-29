# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""
import pandas as pd
from os import chdir
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


chdir('dataset')
train_template = 'PCBA-686978_train{}.txt'

all_results = []

# Resampling methods
resamplers = {
    'Random Oversampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42)
}

param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 150, 200,300],
        'max_depth': [2, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear']
    },
    'XGBoost': {
        'n_estimators': [50, 100, 150,200,300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [2, 5, 10, 20, 30]
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,),(200,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive']
    }
}

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=10000,random_state=42)
}


def tune_model_for_split(split_num):
    try:
        train_file = train_template.format(split_num)
        train_data = pd.read_csv(train_file, delimiter='\t')
        # Check for NaN values and drop rows with NaN
        initial_shape_train = train_data.shape
        train_data.dropna(inplace=True)
        dropped_rows_train = initial_shape_train[0] - train_data.shape[0]
        logging.info(f"Split {split_num}: Dropped {dropped_rows_train} rows with NaN values from the training set.")

        # Separate features, labels, and chemical names
        X_train = train_data.iloc[:, 1:-1]  # Features
        y_train = train_data.iloc[:, -1]    # Labels

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        split_results = []

        # Evaluate each resampler + model combination
        for resampler_name, resampler in resamplers.items():
            for model_name, model in models.items():
                X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
                # Grid Search
                grid_search = GridSearchCV(model, param_grids[model_name], scoring='roc_auc', cv=10, n_jobs=-1)
                grid_search.fit(X_resampled, y_resampled)
                # Best Model
                best_model = grid_search.best_estimator_
                # Best hyperparameters and scores
                split_results.append({
                    'Split': split_num,
                    'Resampler': resampler_name,
                    'Model': model_name,
                    'Best_Params': grid_search.best_params_,
                    'Best_Score': grid_search.best_score_
                })
        return split_results

    except Exception as e:
        logging.error(f"Error during processing split {split_num}: {str(e)}")
        return []

# Tuning for each split in parallel
num_cores = cpu_count()
seed_list = [13,17,21,24,29]
results_list = Parallel(n_jobs=num_cores)(delayed(tune_model_for_split)(i) for i in seed_list)

# Flatten the list of results
results_list_flat = [item for sublist in results_list for item in sublist]

# Convert results to DataFrame
results_df = pd.DataFrame(results_list_flat)

# Save tuning results to CSV
results_df.to_csv('Tuning_results_all_splits.txt', index=False)
# Print results
print(results_df)
