import sys
from os.path import join
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def compute_metrics(y_true, y_pred, y_prob=None, classes=None):
    """
    Compute classification metrics for binary classification (cancer vs non-cancer)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='cancer', zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label='cancer', zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label='cancer', zero_division=0)
    }
    
    # Calculate ROC AUC if probabilities are provided
    if y_prob is not None and classes is not None:
        # Convert labels to binary (1 for cancer, 0 for non-cancer)
        y_true_binary = (y_true == 'cancer').astype(int)
        # Get probability for cancer class
        cancer_probs = y_prob[:, list(classes).index('cancer')]
        metrics['roc_auc'] = roc_auc_score(y_true_binary, cancer_probs)
    
    # Add separate metrics for non-cancer class
    metrics['non_cancer_precision'] = precision_score(y_true, y_pred, pos_label='non-cancer', zero_division=0)
    metrics['non_cancer_recall'] = recall_score(y_true, y_pred, pos_label='non-cancer', zero_division=0)
    metrics['non_cancer_f1'] = f1_score(y_true, y_pred, pos_label='non-cancer', zero_division=0)
    
    return metrics

def main(features_path, label_path, save_path):
    """
    KNN Model training
    
    Arguments:
        features_path: Path to the extracted features CSV
        label_path: Path to the dataset CSV containing labels
        save_path: Path to save the results
    """
    # Load extracted features
    print("Loading data...")
    try:
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(label_path)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e.filename}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: One of the input files is empty")
        return
    
    # Print diagnostic distribution
    print("\nDiagnostic Distribution in Dataset:")
    print(labels_df['diagnostic'].value_counts())
    
    # Debug print
    print(f"\nInitial shapes:")
    print(f"Features DataFrame: {features_df.shape}")
    print(f"Labels DataFrame: {labels_df.shape}")
    
    # Merge features with labels
    print("\nProcessing loaded data...")
    features_df['image_name'] = features_df['image_name'].str.replace('_mask', '')
    merged_df = pd.merge(features_df, labels_df[['img_id', 'diagnostic']], 
                        left_on='image_name', right_on='img_id', how='inner')
    
    # Debug print
    print(f"\nAfter merge:")
    print(f"Merged DataFrame: {merged_df.shape}")
    
    if merged_df.empty:
        print("Error: No matching records found after merging features with labels")
        return
    
    # Features for training
    feature_columns = ['pigment_network', 'blue_veil', 'vascular', 'globules', 'streaks',
                      'irregular_pigmentation', 'regression', 'asymmetry', 'compactness', 'convexity']
    
    # Sanity check for missing values
    missing_values = merged_df[feature_columns].isnull().sum()
    if missing_values.any():
        print("\nWarning: Missing values detected in features:")
        print(missing_values[missing_values > 0])
        print("\nRemoving rows with missing values...")
        merged_df = merged_df.dropna(subset=feature_columns)
        if merged_df.empty:
            print("Error: No records exist!")
            return
    
    X = merged_df[feature_columns]
    
    # Convert multiclass labels to binary (cancer vs non-cancer)
    y = merged_df['diagnostic'].map(lambda x: 'cancer' if x in ['MEL', 'BCC', 'SCC'] else 'non-cancer')
    
    print("\nBinary class distribution (Cancer vs Non-cancer):")
    print(y.value_counts())
    print("\nClass distribution percentages:")
    print(y.value_counts(normalize=True) * 100)
    
    # Split dataset using stratification first
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)
    
    # Standardizing features only on training set
    print("\nStandardizing features (only fitting on training data)...") # cz we need to keep the test set untouched)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test set using training set's parameters
    X_test_scaled = scaler.transform(X_test)
    
    # Verify class distributions after split
    print("\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    print("\nTest set class distribution:")
    print(pd.Series(y_test).value_counts())
    
    # Handle class imbalance using SMOTE
    print("\nChecking class distributions before SMOTE...")
    class_counts = pd.Series(y_train).value_counts()
    print(class_counts)
    
    # Get minimum samples in any class
    min_samples = min(class_counts)
    
    if min_samples < 5: #Just for testing stages, Remove when we have enough data
        print(f"\nWarning: Minority class has only {min_samples} samples.")
        print("Using SMOTE with k_neighbors=min_samples-1 to handle small sample size...")
        k_neighbors = max(1, min_samples - 1)  # Ensure at least 1 neighbor
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    else:
        print("\nApplying standard SMOTE...")
        smote = SMOTE(random_state=42)
        
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Verify class distribution after SMOTE
    print("\nResampled training set class distribution:")
    print(pd.Series(y_train_resampled).value_counts())
    
    
    
    # Train classifier  KNN
    print("\nTraining KNN classifier...")
    # Use weights='distance' to give more weight to closer neighbors
    # Increase n_neighbors to reduce overfitting
    clf = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = compute_metrics(y_test, y_pred, y_prob, clf.classes_)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print("\nResults:")
    print("-" * 50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print("\nCancer Detection Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nNon-Cancer Detection Metrics:")
    print(f"Precision: {metrics['non_cancer_precision']:.4f}")
    print(f"Recall: {metrics['non_cancer_recall']:.4f}")
    print(f"F1 Score: {metrics['non_cancer_f1']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Calculate feature importance using variance
    feature_variances = np.var(X, axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_variances
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Variance (importance measure):")
    print(feature_importance)
    
    # Save results
    print("\nSaving results...")
    test_indices = merged_df.index[y_test.index]
    result_df = merged_df.loc[test_indices, ['image_name', 'diagnostic']].copy()
    result_df['predicted_label'] = y_pred
    
    # Save probabilities for each class
    classes = clf.classes_
    for i, class_name in enumerate(classes):
        result_df[f'probability_{class_name}'] = y_prob[:, i]
    
    result_df['correct_prediction'] = result_df['diagnostic'].map(
        lambda x: 'cancer' if x in ['MEL', 'BCC', 'SCC'] else 'non-cancer'
    ) == result_df['predicted_label']
    
    # Add all metrics
    for metric_name, metric_value in metrics.items():
        result_df[f'metric_{metric_name}'] = metric_value
    
    # Save predictions and metrics
    result_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
    
    # Save feature importance
    importance_path = save_path.replace('.csv', '_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature variances saved to: {importance_path}")

if __name__ == "__main__":
    features_path = "result/masked_out_images/extracted_features.csv"
    label_path = "dataset.csv"
    save_path = "result/result_main_baseline_knn.csv"
    
    main(features_path, label_path, save_path)


