import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline

def main():
    print("Starting enhanced KNN baseline model training...")
    
    # Load and prepare data
    data_path = 'feature_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    os.makedirs('result', exist_ok=True)
    
    # Load dataset
    features_df = pd.read_csv(data_path)
    print(f"Dataset loaded with {len(features_df)} samples")
    
    # Prepare features and target
    X = features_df[['border', 'asymmetry', 'color_score']]
    y = features_df['cancer']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(weights='distance'))
    ])
    
    # Parameter grid for GridSearchCV
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__metric': ['minkowski', 'manhattan'],
        'knn__p': [1, 2],  # p=1 for manhattan, p=2 for euclidean
        'knn__weights': ['distance']
    }
    
    # Perform grid search with cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    
    print("\nPerforming grid search...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Make predictions with best model
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\nTest Set Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Cancer', 'Cancer'])
    plt.yticks(tick_marks, ['Non-Cancer', 'Cancer'])
    
    # Add text annotations to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('result/confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': ['border', 'asymmetry', 'color_score'],
        'importance': np.abs(grid_search.best_estimator_.named_steps['scaler'].scale_)
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(8, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('result/feature_importance.png')
    plt.close()
    
    print("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save predictions and model details
    results = pd.DataFrame({
        'patient_id': features_df['patient_id'],
        'img_id': features_df['img_id'],
        'true_label': features_df['cancer'],
        'predicted_label': grid_search.predict(X),
        'probability': grid_search.predict_proba(X)[:, 1],
        'border': X['border'],
        'asymmetry': X['asymmetry'],
        'color_score': X['color_score']
    })
    
    results.to_csv('result/result_baseline_knn.csv', index=False)
    print("\nResults saved to result/result_baseline_knn.csv")
    
    # Save cross-validation results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv('result/cross_validation_results.csv', index=False)
    print("Cross-validation results saved to result/cross_validation_results.csv")

if __name__ == "__main__":
    main()