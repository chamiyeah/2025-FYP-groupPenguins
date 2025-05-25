import sys
from os.path import join
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics for multiclass classification
    """    # Overall metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    #per-class metrics (specially MEL)
    """
    Using zero_division=0 to handle cases where a class might not be predicted (got errors with 0 div maybe cz of the 
    dataset being imbalanced (some classes have very few samples,model is biased towards majority classes)
    
    STILL HAVE SOME 0 DIVISION ISSUES !!!
    
    """ 

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    #map metrics to class names using unique classes in y_true
    #assumes y_true contains class labels as strings
    classes = np.unique(y_true)
    
    for i, class_name in enumerate(classes):
        metrics[f'{class_name}_precision'] = per_class_precision[i]
        metrics[f'{class_name}_recall'] = per_class_recall[i]
        metrics[f'{class_name}_f1'] = per_class_f1[i]
    
    return metrics

def main(features_path, label_path, save_path):
    """
     KNN model training using the extracted features
    
    Args:
        features_path: Path to the extracted features CSV
        label_path: Path to the dataset CSV containing labels
        save_path: Path to save the results
    """
    
    #load extracted features
    print("Loading data...")
    try:
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(label_path)
        
    #error handling - file reading
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e.filename}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: One of the input files is empty")
        return
    

      # Print diagnostic distribution
    print("\nDiagnostic Distribution in Dataset:")
    print(labels_df['diagnostic'].value_counts())
    print("\nClass meanings:")
    print("MEL: Melanoma (skin cancer)")
    print("BCC: Basal Cell Carcinoma (skin cancer)")
    print("SCC: Squamous Cell Carcinoma (skin cancer)")
    print("NEV: Nevus/mole (benign)")
    print("ACK: Actinic Keratosis (precancerous)")
    
    #merge features with labels
    print("\nProcessing loaded data...")
    features_df['image_name'] = features_df['image_name'].str.replace('_mask', '')  # Remove _mask if present
    merged_df = pd.merge(features_df, labels_df[['img_id', 'diagnostic']], 
                        left_on='image_name', right_on='img_id', how='inner')
    
    if merged_df.empty:
        print("Error: No matching records found after merging features with labels")
        return
    
    #features for training
    feature_columns = ['pigment_network', 'blue_veil', 'vascular', 'globules', 'streaks',
                      'irregular_pigmentation', 'regression', 'asymmetry', 'compactness', 'convexity']
    
    
    
    #Sanity check for missing values
    missing_values = merged_df[feature_columns].isnull().sum()
    if missing_values.any():
        print("\nWarning !! : Missing values detected in features:")
        print(missing_values[missing_values > 0])
        print("\nRemoving rows with missing values...")
        merged_df = merged_df.dropna(subset=feature_columns)
        if merged_df.empty:
            print("Error: No records exist!")
            return
    
    X = merged_df[feature_columns]
    y = merged_df['diagnostic']
      
    #standardizing the features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    #Print class distribution before splitting -  SANITY CHECK 02
    print("\nClass distribution in full dataset:")
    print(y.value_counts())
    print("\nClass distribution percentages:")
    print(y.value_counts(normalize=True) * 100)
    
    #splitting the dataset to train & test
    print("\nSplitting dataset...")
    # Use stratified split to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    #Veryfy after split, wether the split maintained class distributions - Sanity Check 03
    print("\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    print("\nTest set class distribution:")
    print(pd.Series(y_test).value_counts())
    
    #classifier training 
    print("Training KNN classifier...")
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')  # Using default parameters
    clf.fit(X_train, y_train)
      #make predictions
    print("\nMaking predictions...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)  # Probabilities for ALL classes
    
    #calculate metrics
    print("\nCalculating metrics...")
    metrics = compute_metrics(y_test, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\nResults:")
    print("-" * 50)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # For KNN,feature importance is calculated using the feature VARIANCE
    """
    Importanit !!! Calculate variances on original features, not scaled / standardized ones 
    like in logistical regression !!!
    """
    
    feature_variances = np.var(X, axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_variances
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Variance :")
    print(feature_importance)
    
    #Saving results
    print("\nSaving results...")
    test_indices = merged_df.index[y_test.index]
    result_df = merged_df.loc[test_indices, ['image_name', 'diagnostic']].copy()
    result_df['predicted_label'] = y_pred
    
    #save probabilities for each class
    classes = np.unique(y)
    for i, class_name in enumerate(classes):
        result_df[f'probability_{class_name}'] = y_prob[:, i]
    
    result_df['correct_prediction'] = result_df['diagnostic'] == result_df['predicted_label']
    
    #add all metrics
    for metric_name, metric_value in metrics.items():
        result_df[f'metric_{metric_name}'] = metric_value
    
    #save predictions and metrics
    result_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
    
    #save feature importance to a separate file
    importance_path = save_path.replace('.csv', '_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature variances results has been saved to: {importance_path}")

if __name__ == "__main__":
    features_path = "result/masked_out_images/extracted_features.csv"
    label_path = "dataset.csv"
    save_path = "result/result_main_baseline_knn.csv"
    
    main(features_path, label_path, save_path)
