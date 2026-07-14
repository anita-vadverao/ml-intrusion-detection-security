import os
import pickle as pk
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

run_folder = f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_folder, exist_ok=True)


# Preprocessing on the dataset
df = pd.concat(
    map(pd.read_csv, ['UNSW-NB15.csv']), ignore_index=True)
print(df.head())
print(df)

#Label encoding
def custom_label_encoding(column):
    unique_values = column.unique()  # Get unique values in the column
    encoding_dict = {val: idx + 1 for idx, val in enumerate(unique_values)}  # Create encoding dictionary
    return column.map(encoding_dict)  # Map the values to their encoded counterparts

columns_to_encode = ['proto', 'state', 'service']
for column in columns_to_encode:
    new_column_name = f'{column}_encoded'
    df[new_column_name] = custom_label_encoding(df[column])
# Drop the original columns
df.drop(columns_to_encode, axis=1, inplace=True)
df.to_csv('updated_UNSW-NB15.csv', index=False)

# columns_to_keep =['proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','attack']
columns_to_keep =['dur','Dload','swin','dwin','dbytes','stcpb','dtcpb','smeansz','dmeansz','trans_depth','ct_srv_src','ct_srv_dst','attack','proto_encoded','state_encoded','service_encoded']
df.drop(columns=df.columns.difference(columns_to_keep), inplace=True)
df.to_csv('updated_UNSW-NB15.csv', index=False)

col_name_list = list(df.columns)
print("col_name_list = ", col_name_list)
size_col_name_list = len(col_name_list)
print("size_col_name_list = ", size_col_name_list)
print("df.shape ", df.shape)
X = df.drop(labels = ["attack"],axis = 1)
Y = df["attack"].values

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
print("X_train ", X_train)
print("X_test ", X_test)
print("y_train ", y_train)
print("y_test ", y_test)

# for standardizing the dataset
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

iris = load_iris()
X1, y1 = iris.data, iris.target

#Exploratory Data Analysis
#histogram
print("Histogram analysis: ")
df.hist(figsize=(16, 16),layout=(8,8), color="green", alpha = 0.35, bins=10)
plt.tight_layout(h_pad=3.0)
plt.subplots_adjust(top=0.92)
plt.suptitle('Histograms Analysis', fontsize=10)
plt.savefig(f"{run_folder}/histogram_analysis.png")
plt.close()

# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(f"{run_folder}/Correlation Matrix.png")
plt.close()

# Plotting a pie chart for normal vs. attack data distribution
label_counts = df['attack'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=['Green', 'Red'])
plt.title('Distribution of Normal vs. Attack traffic')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.savefig(f"{run_folder}/Distribution of Normal vs Attack traffic.png")
plt.close()

#Numerical features analysis
def plot_dis_number (ds_list):
    g = sns.FacetGrid(df, hue = "attack", aspect = 2.5, palette={1:"Red", 0:"Green"})
    g = g.map(sns.kdeplot,ds_list, fill = True, alpha = 0.8 )
    g.set(xlim = (0, df[ds_list].max()))
    g.add_legend()
    g.set_axis_labels(ds_list, "proportion")
    plt.savefig(f"{run_folder}/Numerical features analysis.png")
    plt.close()

print("Individual numerical features analysis with attack")
plot_dis_number("dur")
plot_dis_number("Dload")
plot_dis_number("swin")
plot_dis_number("dwin")
plot_dis_number("dbytes")
plot_dis_number("stcpb")
plot_dis_number("dtcpb")
plot_dis_number("smeansz")
plot_dis_number("dmeansz")
plot_dis_number("trans_depth")
plot_dis_number("ct_srv_src")
plot_dis_number("ct_srv_dst")
plot_dis_number("proto_encoded")
plot_dis_number("state_encoded")
plot_dis_number("service_encoded")

#supervised learning algorithm - KNN
knn_range = range(1,5)
accuracy_scores = []
for n_neighbors in knn_range:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred_classifier_knn= knn.predict(X_test)
    y_score_knn = knn.predict_proba(X_test)[:, 1]
    accuracy_knn=accuracy_score(y_test,y_pred_classifier_knn) # calculate the accuracy of KNN and stores that data
    accuracy_scores.append(accuracy_knn)
plt.plot(knn_range,accuracy_scores, color='b', linestyle='--', marker='o', label='Predictor Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Predictor Accuracy vs. Number of neighbors KNN')
plt.legend()
plt.savefig(f"{run_folder}/Predictor Accuracy vs Number of neighbors KNN.png")
plt.close()

#Cross validation code for KNN classifier
X2, y2 = make_classification(n_samples=100, n_features=4, n_classes=2,
                           n_informative=2, n_redundant=1, n_repeated=0,
                           random_state=42)

# cross validation predictor accuracy for KNN for Numbers of Neighbors
mean_accuracy_knn = []
for k in knn_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X2, y2, cv=5)
    mean_accuracy_knn.append(np.mean(cv_scores))
# Plotting number of neighbors vs. mean accuracy
plt.figure(figsize=(8, 6))
plt.plot(list(knn_range), mean_accuracy_knn, marker='o', linestyle='-', color='r', label='Predictor Accuracy')
plt.title('Cross Validation Predictor Accuracy for KNN for Numbers of Neighbors')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean Accuracy')
plt.xticks(list(knn_range))
plt.legend()
plt.grid(True)
plt.savefig(f"{run_folder}/Cross Validation Predictor Accuracy for KNN for Numbers of Neighbors.png")
plt.close()

# confusion matrix for KNN
cm_knn=confusion_matrix(y_test, y_pred_classifier_knn)
print("Confusion matrix for KNN",cm_knn)
labels = ['Not attacked', 'attacked']
plt.figure(figsize=(7,5))
ax= plt.subplot()
sns.heatmap(cm_knn,cmap="Reds",annot=True,fmt='.1f', ax = ax)
# labels, title and ticks
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predictor labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix KNN')
plt.savefig(f"{run_folder}/Confusion Matrix KNN.png")
plt.close()

#Normalized confusion matrix for KNN
cm_knn_nor = cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis]
print("Normalized Confusion matrix for KNN",cm_knn_nor)
labels = ['Not attacked', 'attacked']
plt.figure(figsize=(7,5))
ax= plt.subplot()
# plt.imshow(cm_knn_nor,interpolation='nearest',cmap="Greens")
sns.heatmap(cm_knn_nor,cmap="Greens", annot=True,fmt='.3f', ax = ax)
# # labels, title and ticks
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predictor labels');ax.set_ylabel('True labels')
ax.set_title('Normalized Confusion Matrix KNN')
plt.savefig(f"{run_folder}/Normalized Confusion Matrix KNN.png")
plt.close()

# supervised learning algorithm - Random Forest classifier for accuracy
estimator_range = [1,12,20,33,40,50]
accuracy_scores = []
for n_estimator in estimator_range:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimator, random_state=150)
    rf_classifier.fit(X_train, y_train)
    y_pred_classifier_rfc= rf_classifier.predict(X_test)
    y_score_rfc = rf_classifier.predict_proba(X_test)[:, 1]
    accuracy_rf=accuracy_score(y_test,y_pred_classifier_rfc) # calculate the accuracy of Random forest classifier and stores that data
    accuracy_scores.append(accuracy_rf)
plt.plot(estimator_range,accuracy_scores, color='b', linestyle='--', marker='o', label='Predictor Accuracy')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Predictor Accuracy vs. Number of Estimators Random Forest')
plt.legend()
plt.savefig(f"{run_folder}/Predictor Accuracy vs. Number of Estimators Random Forest.png")
plt.close()

#Cross validation code for Random forest classifier
mean_accuracy_rf = []
for n_estimator in estimator_range:
    rf_classifier_cross = RandomForestClassifier(n_estimators=n_estimator, random_state=42)
    rf_classifier_cross.fit(X_train, y_train)
    cross_validation_scores = cross_val_score(rf_classifier_cross, X1, y1, cv=8, scoring='accuracy')
    print("Random Forest classifier cross validation score", cross_validation_scores)
    mean_accuracy = np.mean(cross_validation_scores)
    mean_accuracy_rf.append(mean_accuracy)
plt.plot(estimator_range, mean_accuracy_rf, marker='o', linestyle='-', color='r',label='Predictor Accuracy')
plt.xlabel('Number of Estimators (Trees)')
plt.ylabel('Accuracy')
plt.title('Cross Validation Predictor Accuracy for Random Forest Classifier')
plt.xticks(np.arange(0, max(estimator_range) + 1, 50))  # Adjust x-axis ticks if needed
plt.tight_layout()
plt.savefig(f"{run_folder}/Cross Validation Predictor Accuracy for Random Forest Classifier.png")
plt.close()

# confusion matrix for Random forest
cm_rfc=confusion_matrix(y_test, y_pred_classifier_rfc)
print("Confusion matrix for Random Forest",cm_rfc)
labels = ['Not attacked', 'attacked']
plt.figure(figsize=(7,5))
ax= plt.subplot()
sns.heatmap(cm_rfc,cmap="Blues",annot=True,fmt='.1f', ax = ax)
# labels, title and ticks
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predictor labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix Random Forest')
plt.savefig(f"{run_folder}/Confusion Matrix Random Forest")
plt.close()

#Normalized confusion matrix for Random forest
cm_rfc_nor = cm_rfc.astype('float') / cm_rfc.sum(axis=1)[:, np.newaxis]
print("Normalized Confusion matrix for Random Forest",cm_rfc_nor)
labels = ['Not attacked', 'attacked']
plt.figure(figsize=(7,5))
ax= plt.subplot()
sns.heatmap(cm_rfc_nor,cmap="Greens", annot=True,fmt='.3f', ax = ax)
# # labels, title and ticks
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predictor labels')
ax.set_ylabel('True labels')
ax.set_title('Normalized Confusion Matrix Random Forest')
plt.savefig(f"{run_folder}/Normalized Confusion Matrix Random Forest")
plt.close()

#supervised learning algorithm - Logistics regression for accuracy
accuracy_scores_lr = []
weights, params = [], []
for c in np.arange(-5., 4.):
    lr_classifier = LogisticRegression(C=10.**c, random_state=50,max_iter=1000)
    lr_classifier.fit(X_train_std, y_train)
    weights.append(lr_classifier.coef_[0])
    params.append(10**c)
    y_pred_lr = lr_classifier.predict(X_test_std)
    y_score_log = lr_classifier.predict_proba(X_test)[:, 1]
    print('Accuracy1: %.2f' % accuracy_score(y_test, y_pred_lr))
    accuracy_scores_lr.append(accuracy_score(y_test, y_pred_lr))
plt.plot(params, accuracy_scores_lr,color='r', marker='o', linestyle='--',label='Predictor Accuracy')
plt.title('Logistics Regression Predicator Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
plt.savefig(f"{run_folder}/Logistics Regression Predicator Accuracy")
plt.close()

#Cross validation code for Logistic Regression classifier
mean_accuracy_lr = []
weights, params = [], []
for c in np.arange(-5., 4.):
    lr = LogisticRegression(C=10.**c, random_state=50, max_iter=1000)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[0])
    params.append(10**c)
    cross_validation_scores = cross_val_score(lr, X1, y1, cv=8, scoring='accuracy')
    print("Logistics Regression Cross Validation Score", cross_validation_scores)
    mean_accuracy = np.mean(cross_validation_scores)
    mean_accuracy_lr.append(mean_accuracy)
plt.plot(params,mean_accuracy_lr,color='b', marker='o', linestyle='--',label='Predictor Accuracy')
plt.title('Logistics Regression Cross Validation Predictor Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
plt.savefig(f"{run_folder}/Logistics Regression Cross Validation Predictor Accuracy")
plt.close()

# confusion matrix for Logistics Regression
conf_matrix = confusion_matrix(y_test, y_pred_lr)
print("Confusion matrix for Logistics Regression",conf_matrix)
labels = ['Not attacked', 'attacked']
plt.figure(figsize=(7,5))
ax= plt.subplot()
sns.heatmap(conf_matrix,cmap="Reds",annot=True,fmt='.1f', ax = ax);
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Logistics Regression');
plt.savefig(f"{run_folder}/Confusion Matrix Logistics Regression")
plt.close()

# Normalized confusion matrix for Logistics Regression
conf_matrix_nor = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
print("Normalized Confusion matrix for Logistics Regression",conf_matrix_nor)
labels = ['Not attacked', 'attacked']
plt.figure(figsize=(7,5))
ax= plt.subplot()
sns.heatmap(conf_matrix_nor,cmap="Greens", annot=True,fmt='.3f', ax = ax);
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Normalized Confusion Matrix Logistics');
plt.savefig(f"{run_folder}/Normalized Confusion Matrix Logistics")
plt.close()

#RoC curve for KNN, Random Forest and Logistics Regression
#RoC curve for KNN
fpr_knn, tpr_knn,_ = roc_curve(y_test, y_score_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
print(roc_auc_knn)

#RoC curve for Random Forest
fpr_rf, tpr_rf,_ = roc_curve(y_test, y_score_rfc)
roc_auc_forests = auc(fpr_rf, tpr_rf)
print(roc_auc_forests)

#RoC curve for Logistic regression
fpr_lr, tpr_lr,_ = roc_curve(y_test, y_score_log)
roc_auc_log = auc(fpr_lr, tpr_lr)
print(roc_auc_log)

# ROC curves analysis
ns_probs = [0 for _ in range(len(y_test))]

# lstm_fpr, lstm_tpr, _ = roc_curve(y_test, yhat)
no_fpr, no_tpr, _ = roc_curve(y_test, ns_probs)

print("fpr_knn, tpr_knn", fpr_knn, "\n", tpr_knn)
print("fpr_lr, tpr_lr", fpr_knn, "\n", tpr_knn)
print("fpr_dt, tpr_dt", fpr_knn, "\n", tpr_knn)
print("fpr_rf, tpr_rf", fpr_knn, "\n", tpr_knn)

t = np.arange(0, 110, 10)
fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(lstm_fpr, lstm_tpr, linestyle='--', color='r',  lw=2.0, label='LSTM')
ax.plot(fpr_knn, tpr_knn, linestyle='--', color='b',  lw=2.0, label='KNN')
ax.plot(fpr_lr, tpr_lr, linestyle='--', color='y',  lw=2.0, label='LR')
ax.plot(fpr_rf, tpr_rf, linestyle='--', color='m',  lw=2.0, label='RF')

ax.plot(no_fpr, no_tpr, marker='.', color='g', lw=2.0, label='No skill')
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC curve', fontsize = 18)
plt.legend(loc='best',fontsize = 18)
ax.grid(True)
ticklines = ax.get_xticklines() + ax.get_yticklines()
gridlines = ax.get_xgridlines()
ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-')

for line in gridlines:
    line.set_linestyle('-')

for label in ticklabels:
    label.set_color('black')
    label.set_fontsize('large')
plt.savefig(f"{run_folder}/ROC curve")
plt.close()

#Precision Recall curve for KNN, Random Forest and Logistics Regression
#Precision Recall curve for KNN
knn_precision, knn_recall, _ = precision_recall_curve(y_test, y_pred_classifier_knn)

#Precision Recall curve for Random Forest
rf_precision, rf_recall, _ = precision_recall_curve(y_test, y_pred_classifier_rfc)

#Precision Recall curve for Logistic regression
lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred_lr)

# Precision recall curves analysis
no_skill = len(y_test[y_test==1]) / len(y_test)

t = np.arange(0, 110, 10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(knn_recall, knn_precision, linestyle='--', color='b',  lw=2.0, label='KNN')
ax.plot(lr_recall, lr_precision, linestyle='--', color='y',  lw=2.0, label='LR')
ax.plot(rf_recall, rf_precision, linestyle='--', color='m',  lw=2.0, label='RF')

ax.plot([0, 1], [no_skill, no_skill], marker='.', color='g', lw=2.0, label='No skill')
plt.xlabel('Recall', fontsize = 18)
plt.ylabel('Precision', fontsize = 18)
plt.title('Precision-Recall curve', fontsize = 18)
plt.legend(loc='best',fontsize = 18)
ax.grid(True)
ticklines = ax.get_xticklines() + ax.get_yticklines()
gridlines = ax.get_xgridlines()
ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-')

for line in gridlines:
    line.set_linestyle('-')

for label in ticklabels:
    label.set_color('black')
    label.set_fontsize('large')
plt.savefig(f"{run_folder}/Precision-Recall curve")
plt.close()

#Save Best model(rfc)
def save_model(model):
        with open('Cyber1.pickle', 'wb') as f:
            pk.dump(model, f)

#Save Best model(rfc)
save_model(rf_classifier)

# Load and execute Saved Model Execution
def read_and_predict(test_data):
    # load model
    f = open('Cyber1.pickle', 'rb')
    model = pk.load(f)
    f.close()
    predicted_attack = model.predict(test_data)
    print("predicted_attack ", predicted_attack)
    confmat_out = confusion_matrix(y_true=y_test, y_pred=predicted_attack)
    print("Confusion Matrix:")
    print(confmat_out)
    labels = ['Not attacked', 'attacked']
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    sns.heatmap(confmat_out, cmap="Blues", annot=True, fmt='.1f', ax=ax)
    # labels, title and ticks
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predictor labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix Random Forest')
    plt.savefig(f"{run_folder}/Confusion Matrix Random Forest")
    df = pd.DataFrame(predicted_attack, columns=['predicted_attack'])
    df.to_csv('FinalResult.csv', index=False)
    plt.close()

# Load and execute Saved Model Execution
read_and_predict(X_test)

# print("Scatter matrix analysis: ")
# selected_columns1 = ['dur', 'dbytes', 'Dload', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz']
# selected_columns2 = ['dmeansz', 'trans_depth', 'ct_srv_src', 'ct_srv_dst', 'attack', 'proto_encoded', 'state_encoded', 'service_encoded']
# # Create a scatter matrix for the selected columns
# scatter_matrix(df[selected_columns1],figsize= (16,16), color="red", alpha =0.5)
# scatter_matrix(df[selected_columns2],figsize= (16,16), color="red", alpha =0.5)
# plt.ylabel(12)
# plt.savefig(f"{run_folder}/Scatter matrix analysis ")
# plt.close()

html_report = f"""
<html>
<head><title>Intrusion Detection Model Report</title></head>
<body>
<h1>Model Evaluation Report</h1>
<h2>Accuracy Comparison</h2>
<table border="1" cellpadding="5">
<tr><th>Model</th><th>Accuracy</th></tr>
<tr><td>KNN</td><td>{accuracy_score(y_test, y_pred_classifier_knn):.4f}</td></tr>
<tr><td>Random Forest</td><td>{accuracy_score(y_test, y_pred_classifier_rfc):.4f}</td></tr>
<tr><td>Logistic Regression</td><td>{accuracy_score(y_test, y_pred_lr):.4f}</td></tr>
</table>
<h2>Confusion Matrices</h2>
<img src="confusion matrix KNN.png" width="400">
<img src="Confusion Matrix Random Forest.png" width="400">
<img src="Confusion Matrix Logistics Regression.png" width="400">
<h2>ROC Curve</h2>
<img src="ROC curve.png" width="400">
</body>
</html>
"""

with open(f'{run_folder}/report.html', 'w') as f:
    f.write(html_report)
