# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
"""
from os import chdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

chdir('dataset')
def load_output_file(filepath):
    model_metrics = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        current_model = None
        in_report = False
        for line in lines:
            line = line.strip()
            if line.startswith("Model:"):
                current_model = line.split(":")[1].strip()
                model_metrics[current_model] = { "Precision": [], "Recall": [], "F1-Score": [], "ROC AUC": [], "Accuracy": [], "MCC": [] }
                continue
            if line.startswith("Classification Report:"):
                in_report = True
                continue
            if in_report:
                # Extract accuracy value from the report
                if "accuracy" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            accuracy = float(parts[1])
                            model_metrics[current_model]["Accuracy"].append(accuracy)
                        except ValueError:
                            pass
                elif line.startswith("macro avg") or line.startswith("weighted avg"):
                    in_report = False
                continue
            # Extract metrics
            if current_model:
                if "Precision" in line:
                    model_metrics[current_model]["Precision"].append(float(line.split(":")[1].strip()))
                elif "Recall" in line:
                    model_metrics[current_model]["Recall"].append(float(line.split(":")[1].strip()))
                elif "F1-Score" in line:
                    model_metrics[current_model]["F1-Score"].append(float(line.split(":")[1].strip()))
                elif "ROC AUC" in line:
                    model_metrics[current_model]["ROC AUC"].append(float(line.split(":")[1].strip()))
                elif "MCC" in line:
                    model_metrics[current_model]["MCC"].append(float(line.split(":")[1].strip()))
    return model_metrics

seed_list = [13,17,21,24,29]
all_model_metrics = {}
for i in seed_list:
    filepath = f'PCBA-686978_model_result_split_{i}.txt'
    file_metrics = load_output_file(filepath)
    for model, metrics in file_metrics.items():
        if model not in all_model_metrics:
            all_model_metrics[model] = { "Precision": [], "Recall": [], "F1-Score": [], "ROC AUC": [], "Accuracy": [], "MCC": [] }
        for metric, values in metrics.items():
            all_model_metrics[model][metric].extend(values)

# Calculate averages and standard deviations
model_averages = {model: {metric: np.mean(values) for metric, values in metrics.items()} for model, metrics in all_model_metrics.items()}
model_std_devs = {model: {metric: np.std(values) for metric, values in metrics.items()} for model, metrics in all_model_metrics.items()}

summary_data = {
    'Model': [],
    'Precision (Mean)': [], 'Precision (Std Dev)': [],
    'Recall (Mean)': [], 'Recall (Std Dev)': [],
    'F1-Score (Mean)': [], 'F1-Score (Std Dev)': [],
    'ROC AUC (Mean)': [], 'ROC AUC (Std Dev)': [],
    'Accuracy (Mean)': [], 'Accuracy (Std Dev)': [],
    'MCC (Mean)': [], 'MCC (Std Dev)': []
}

for model in all_model_metrics.keys():
    summary_data['Model'].append(model)
    summary_data['Precision (Mean)'].append(model_averages[model]['Precision'])
    summary_data['Precision (Std Dev)'].append(model_std_devs[model]['Precision'])
    summary_data['Recall (Mean)'].append(model_averages[model]['Recall'])
    summary_data['Recall (Std Dev)'].append(model_std_devs[model]['Recall'])
    summary_data['F1-Score (Mean)'].append(model_averages[model]['F1-Score'])
    summary_data['F1-Score (Std Dev)'].append(model_std_devs[model]['F1-Score'])
    summary_data['ROC AUC (Mean)'].append(model_averages[model]['ROC AUC'])
    summary_data['ROC AUC (Std Dev)'].append(model_std_devs[model]['ROC AUC'])
    summary_data['Accuracy (Mean)'].append(model_averages[model]['Accuracy'])
    summary_data['Accuracy (Std Dev)'].append(model_std_devs[model]['Accuracy'])
    summary_data['MCC (Mean)'].append(model_averages[model]['MCC'])
    summary_data['MCC (Std Dev)'].append(model_std_devs[model]['MCC'])

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('model_performance_summary.csv', index=False)

# Plot
models = list(all_model_metrics.keys())
metrics = list(all_model_metrics[models[0]].keys())
n_models = len(models)
n_metrics = len(metrics)
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(n_metrics)
width = 0.1
for i, model in enumerate(models):
    avg_values = [model_averages[model][metric] for metric in metrics]
    std_values = [model_std_devs[model][metric] for metric in metrics]
    ax.bar(x + i * width - ((n_models - 1) / 2) * width, avg_values, width, 
           yerr=std_values, label=model, capsize=5)
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics for Different Models Across Splits')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.savefig('model_performance_plot.png')
plt.show()
