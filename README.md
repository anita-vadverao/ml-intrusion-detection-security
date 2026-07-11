ML-Based Network Intrusion Detection

A machine learning project that trains and evaluates three classifiers — K-Nearest Neighbors, Random Forest, and Logistic Regression — to detect malicious network traffic using the UNSW-NB15 dataset. Includes automated model evaluation and CI/CD integration via Jenkins.

Project Goal

Use AI/ML to enhance intrusion detection systems (IDS) by training classifiers to distinguish normal network traffic from attacks, then comparing model performance to identify the most effective approach.

Dataset

UNSW-NB15 — a network intrusion dataset containing both normal traffic and 9 categories of attacks (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shell Code, Worms). Traffic is labeled as attack (1) or normal (0).


Dataset not included in this repo due to file size. Download from the UNSW-NB15 official source and place UNSW-NB15.csv in the project root before running.



Approach


Preprocessing — custom label encoding for categorical columns (proto, service, state), feature selection down to 16 relevant columns
Exploratory Data Analysis — histograms, correlation matrix, class distribution, per-feature density plots, scatter matrices
Model Training — three classifiers trained and compared:

K-Nearest Neighbors (tested across neighbor counts 1–4)
Random Forest (tested across estimator counts: 1, 12, 20, 33, 40, 50)
Logistic Regression (tested across regularization strength C, 10⁻⁵ to 10³)



Evaluation — confusion matrices, normalized confusion matrices, ROC curves, precision-recall curves, cross-validation


Results

ModelPredictor AccuracyCross-ValidationFalse Negative RateKNN98.95%97%0.385Random Forest99.78%99%0.055Logistic Regression98.10%98%0.856

Random Forest performed best overall — highest accuracy and, critically, the lowest false negative rate (missed attacks), which matters most for a security use case where an undetected attack is more costly than a false alarm.

Project Structure

├── data_and_models.py      # Data loading, preprocessing, and model training
├── main.py                 # Runs the full pipeline: EDA, training, evaluation, saves best model
├── requirements.txt
├── plots/                  # Generated charts (created when main.py runs)
├── Cyber1.pickle           # Saved best-performing model (generated on run)
└── FinalResult.csv         # Model predictions (generated on run)

How to Run

bashpip install -r requirements.txt
python main.py

This will:


Preprocess the dataset
Generate all EDA and evaluation plots into plots/
Train and compare all three models
Save the best-performing model as Cyber1.pickle
Save predictions to FinalResult.csv


CI/CD

This project is integrated with Jenkins to automatically run the pipeline on each commit, validating that the model continues to meet minimum performance thresholds before being considered passing.

Background

Originally developed as a group project for a graduate-level AI Security course, since adapted as an individual portfolio project with restructured code, headless plotting support, and CI/CD integration.

References


Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. Military Communications and Information Systems Conference (MilCIS).
Kasongo, S. M. (2020). Performance Analysis of Intrusion Detection Systems Using a Feature Selection Method on the UNSW-NB15 Dataset. Journal of Big Data, 7(1).