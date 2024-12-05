# QMSA-ML
QMSA-ML is an explainable machine learning framework for the prediction of PFAS bioactivity using quantitative molecular surface analysis-derived representation.<br>
## Workflow
- First, the molecular electrostatic potential (ESP) of compounds is obtained using density functional theory (DFT) calculations, and then the quantitative molecular surface analysis (QMSA) descriptors were calculated.
- Next, random forest (RF), support vector machine (SVM), extreme gradient boosting (XGB), multilayer perceptron (MLP), and stacking models were trained, tunned, and evaluated.
- Finally, the contribution of QMSA descriptors to PFAS bioactivity were evaluated through Shapley analysis.
## Dataset
The original PFAS bioactivity datasets (referred to as the C3F6 and CF datasets) were obtained from a previous publication by Cheng et al (DOI:10.1021/acs.est.9b04833). These datasets were constructed using data from PubChem’s BioAssay (PCBA), Maximum Unbiased Validation (MUV), human β-secretase 1 (BACE), blood-brain barrier penetration (BBBP), and Toxicology in the 21st Century (Tox21) datasets. The C3F6 dataset consists of 1,012 compounds containing a perfluoroalkyl moiety (-CnF2n-, n ≥ 3) with three or more carbons, which are considered as PFAS according to current definition, and the CF dataset contains 62,043 compounds with ≥ 1 C-F bond.<br>
## Overview of the modules
- Datasets: The training sets and validation sets for target PCBA-686978. Random data splitting was performed five times.
- Models: Scripts for modeling, hyperparameters tunning, model evaluation, Shapley analysis.
- Results: The results of model performance evaluation using multiple metrics (accuracy, precision, recall (sensitivity), AUC-ROC, F1-score, and MCC).
## Installation
All scripts were written in Python 3.9. The raw scripts have been tested on both Windows and Linux systems. A minimum of 2 GB of memory and 10 GB of free disk space are recommended. The required Python packages include Scikit-learn, pandas, imblearn, matplotlib, and NumPy.
