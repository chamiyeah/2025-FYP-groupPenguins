import pandas as pd
import sys
import os
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

sys.path.append(os.path.abspath('..'))
########### DATA LOADING ############################################################################################################

def load_and_splitbypatient(data, features):
    """ 
    - Splitting data into training and testing sets by patient ID (preventing leakage)
        
    Input:
    ----------
        data : pandas.DataFrame
            The full dataset with extracted features, labels, and patient/image ids
        features : list of str
            List of features for training 

    Returns:
    -------
        X : pandas.DataFrame
            Feature matrix based on selected feature
        y : pandas.Series
            Binary labels (0 for benign, 1 for cancer)
        groups : pandas.Series
            Patient IDs corresponding to each row in the data
        train_idx : pandas.Series (bool mask)
            Boolean mask to select training rows (80% of all patients)
        test_idx : pandas.Series (bool mask)
            Boolean mask to select testing rows (20% of all patients)
    """

    data.dropna(inplace=True) #drop rows with Nan in features dataframe
    data['cancer'] = data['cancer'].astype(int) #convert bool to binary

    X = data[features]#features - baseline
    y = data['cancer']#labels

    #grouping 
    groups = data['patient_id']
    unique_patients = data['patient_id'].unique() #number of unique patients

    #split into training and testing considering patient id, 80% patients in training and 20% in test (patients distribution tested before)
    train_ids, test_ids = train_test_split(unique_patients, test_size=0.2, random_state=42) 

    #using training set for cross validation
    train_idx = data['patient_id'].isin(train_ids)#selecting patients in that are in training set

    test_idx = data['patient_id'].isin(test_ids) 

    return X, y, groups, train_idx, test_idx

########## CROSS VALIDATION #################################################################################################################

def cross_validation(data, features, max_depth, visualize = True, save = True):
    """
    Cross-validation with varying max_depth values,
    using GroupKFold

    Also:
    - Checks cancer class balance in the training set
    - Optionally visualizes and/or saves cross-validation results

    Input:
    ----------
        data : pandas.DataFrame
            The full dataset with extracted features, labels, and patient/image ids
        features : list of str
            List of features for training 
        max_depth : int
            Maximum depth to try (from 1 to max_depth - 1) for decision tree
        visualize : bool(default=True)
            Show an error bar plot of AUC vs tree depth
        save : bool(default=True)
            Save the cross-validation results to CSV and the plot to PNG

    Returns:
    -------
        summary_df : pandas.DataFrame
            Data frame containing mean AUC and standard deviation for each tree depth
        cancer_balance : float
            Percentage of cancer cases in the training set, just a check
    """

    X, y, groups, train_idx, _ = load_and_splitbypatient(data, features)

    X_train, y_train = X[train_idx], y[train_idx] #getting training set patients features and labels
    groups_train = groups[train_idx] #saving patients ids for GroupKFold

    #checking balance
    cancer_balance = round(y_train.sum()/len(X_train)*100)

    max_depth_values = range(1, max_depth) 
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
        summary.append({'max_depth': depth, 'mean_auc': mean_auc, 'std_auc': std_auc})

    #result for cross validation
    summary_df = pd.DataFrame(summary)

    if save: 
        summary_df.to_csv("result/cross_val_DT_baseline.csv", index=False)

    if visualize:
        plt.errorbar(summary_df['max_depth'], summary_df['mean_auc'], yerr=summary_df['std_auc'], fmt='-o')
        plt.xlabel("Tree Depth")
        plt.ylabel("Mean AUC (±1 std)")
        plt.title("Decision Tree, Cross-Validation for depth determination")
        plt.grid(True)
        plt.savefig("result/cross_val_DT_baseline.png", dpi=300) 
        plt.show()

    return summary_df, cancer_balance

######### MODEL TRAINING #######################################################################################################################

def model_training(data, features):
    """
    Trains a Decision Tree classifier with chosen depth on the training data.
    Handling that no patient appears in both training and test sets.

    Input:
    ----------
        data : pandas.DataFrame
            The full dataset with extracted features, labels, and patient/image ids
        features : list of str
            List of features for training 

    Returns:
    -------
        pipe : sklearn.pipeline.Pipeline
            A fitted pipeline containing a StandardScaler and DecisionTreeClassifier with chosen depth
    """
    X, y, _, train_idx, _ = load_and_splitbypatient(data, features)

    #training set labels and features, 80% patients
    X_train, y_train = X[train_idx], y[train_idx] #train_idx defined before for patients in training data

    # Print class balance information
    print("\nTraining Set Balance:")
    print(f"Total samples in training set: {len(y_train)}")
    print(f"Non-cancerous cases (0): {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
    print(f"Cancerous cases (1): {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)\n")

    pipe = make_pipeline(
        StandardScaler(),  
        DecisionTreeClassifier(max_depth=4, random_state=42)
    )

    pipe.fit(X_train, y_train)

    return pipe

######### PREDICTION AND EVALUTATION #####################################################################################################################

def prediction_evaluation(data, features, pipe, result_dir, name1, name2):
    """
    Uses trained Decision Tree (actually retrains inside the function) and evaluates it on the test set.
    Outputs evaluation metrics, displays the confusion matrix, and saves the predictions if desired.

    Input:
    ----------
        data : pandas.DataFrame
            The full dataset with extracted features, labels, and patient/image ids
        features : list of str
            List of features for training 
        pipe : sklearn.pipeline.Pipeline
            A trained pipeline consisting of a StandardScaler and a DecisionTreeClassifier
        result_dir : str or pathlib.Path
            Path to directory where all the results are saved in
        name1: str
            File name for confusion matrix image
        name2: str
            File name for results CSV

    Returns:
    -------
        None
            Prints performance metrics and confusion matrix instead
    
    Side Effects:
    -------------
        - Saves confusion matrix image to '../result/confusion_matrix_baseline.png'
        - Saves results CSV to '../result/result_baseline.csv' (if save=True)

    Output CSV:
    -------------------
    - image_id: ID of the image
    - patient_id: ID of the patient
    - label: ground truth (0 or 1)
    - probability: model predicted probability of cancer
    """

    X, y, groups, _, test_mask = load_and_splitbypatient(data, features)

    #test set labels and features, 20% patients
    X_test, y_test = X[test_mask], y[test_mask]

    y_pred = pipe.predict(X_test) #automatically applies a default threshold of 0.5, used for False negatives investingation
    y_prob = pipe.predict_proba(X_test)[:, 1]

    #metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #for results data frame
    patient_ids_test = groups[test_mask]
    image_ids_test = data['img_id'][test_mask]

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
    plt.savefig(result_dir / name1, dpi=300) 
    plt.show()

    #results
    results_df = pd.DataFrame({
        "image_id": image_ids_test.values,
        "patient_id": patient_ids_test.values,
        "label": y_test.values,
        "probability": y_prob
    })

    #check for balance, because we excluded Nan features
    print(f'Total number of images in test set: {len(results_df)}')
    print(f'True cancerous images in test set: {len(results_df[results_df['label']==1])}')
    print(f'True non-cancerous images in test set: {len(results_df[results_df['label']==0])}')
    
    results_df.to_csv(result_dir / name2, index=False)