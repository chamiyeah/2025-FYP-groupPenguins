import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, 
    precision_score, f1_score, ConfusionMatrixDisplay
)

"""
BASELINE MODEL WITH ABC FEATURES - Decision tree

This file provides cross validation code that we used do determine the optimal depth 
of the decision tree, and a code for training and testing the classifier

Input: csv file with extracted features 'enhanced_feature_dataset.csv'
Output: cross validation figure for depth selection, 
metrics for test set after chosing depth, csv file with predicted probabilities of test data
"""


#loading dataframe with extracted features 
data = pd.read_csv('../result/enhanced_feature_dataset.csv') 
data.dropna(inplace=True) #drop rows with Nan in features dataframe
data['cancer'] = data['cancer'].astype(int) #convert bool to binary

X = data[['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy']]#features - baseline
y = data['cancer']#labels

#grouping 
groups = data['patient_id']
unique_patients = data['patient_id'].unique() #number of unique patients

#split into training and testing considering patient id, 80% patients in training and 20% in test (patients distribution tested before)
train_ids, test_ids = train_test_split(unique_patients, test_size=0.2, random_state=42) 

####################################################################################################
#CROSS VALIDATION - tunning depth hyperparameter

#using training set for cross validation
train_idx = data['patient_id'].isin(train_ids)#selecting patients in that are in training set

X_train, y_train = X[train_idx], y[train_idx] #getting training set patients features and labels
groups_train = groups[train_idx] #saving patients ids for GroupKFold?

#checking balance
print(f'Cancer cases: {round(y_train.sum()/len(X_train)*100)}% of the training data set')

max_depth_values = range(1, 21) 
cv = GroupKFold(n_splits=5)

summary = [] #max_depth values with mean AUC and STDs

#cross validation loop
for depth in max_depth_values:
    clf = make_pipeline(
        StandardScaler(),  #not crutial for decision tree, but added for consistency
        DecisionTreeClassifier(max_depth=depth, random_state=42)
    )

    auc_scores = cross_val_score(
        clf, X_train, y_train, cv=cv, groups=groups_train, scoring='roc_auc'
    )
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"max_depth={depth}, AUC={mean_auc:.4f} ± {std_auc:.4f}")
    summary.append({'max_depth': depth, 'mean_auc': mean_auc, 'std_auc': std_auc})

#result for cross validation
summary_df = pd.DataFrame(summary) 
summary_df.to_csv("../result/cross_val_DT_baseline.csv", index=False)

plt.errorbar(summary_df['max_depth'], summary_df['mean_auc'], yerr=summary_df['std_auc'], fmt='-o')
plt.xlabel("Tree Depth")
plt.ylabel("Mean AUC (±1 std)")
plt.title("Decision Tree, Cross-Validation for depth determination")
plt.grid(True)
plt.savefig("../result/cross_val_DT_baseline.png", dpi=300) 
plt.show()
####################################################################################################
#RE-TRAINNING and TESTING MODEL WITH CHOSEN HYPERPARAMETER

test_mask = data['patient_id'].isin(test_ids) 

#training set labels and features, 80% patients
X_train, y_train = X[train_idx], y[train_idx] #train_idx defined before for patients in training data
#test set labels and features, 20% patients
X_test, y_test = X[test_mask], y[test_mask]

#for results data frame
patient_ids_test = groups[test_mask]
image_ids_test = data['img_id'][test_mask]

pipe = make_pipeline(
    StandardScaler(),  
    DecisionTreeClassifier(max_depth=4, random_state=42)
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test) #automatically applies a default threshold of 0.5, used for False negatives investingation
y_prob = pipe.predict_proba(X_test)[:, 1]

#metrics
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Metrics on test set:")
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1 Score:  {f1:.4f}")

#confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Benign", "Cancer"],
    cmap="Oranges",
    values_format="d"
)
plt.title("Confusion Matrix of decision tree (depth=4)")
plt.grid(False)
plt.savefig("../result/confusion_matrix_baseline.png", dpi=300) 
plt.show()

#results
results_df = pd.DataFrame({
    "image_id": image_ids_test.values,
    "patient_id": patient_ids_test.values,
    "label": y_test.values,
    "probability": y_prob
})

#check for balance, because we excluded Nan features
print(f'Total number of images: {len(results_df)}')
print(f'True cancerous images: {len(results_df[results_df['label']==1])}')
print(f'True non-cancerous images: {len(results_df[results_df['label']==0])}')

results_df.to_csv("../result/result_baseline.csv", index=False)