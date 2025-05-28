from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from model_utils import (
    load_and_preprocess_data,
    calculate_metrics,
    plot_confusion_matrix,
    plot_cross_validation_results
)

class DecisionTreeModel:
    def __init__(self, max_depth=4, random_state=42):
        """Initialize the DecisionTree classifier with given parameters"""
        self.max_depth = max_depth
        self.random_state = random_state
        self.pipeline = make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        )
     #load the data and create the test and train split    
    def train_test_split(self, data_path):
        self.data, self.X, self.y, self.groups = load_and_preprocess_data(data_path)
        unique_patients = self.data['patient_id'].unique()
        
        #slit by patient ID to avoid data leakage
        train_ids, test_ids = train_test_split(
            unique_patients, test_size=0.2, random_state=self.random_state
        )
        
        train_mask = self.data['patient_id'].isin(train_ids)
        test_mask = self.data['patient_id'].isin(test_ids)
        
        self.X_train = self.X[train_mask]
        self.y_train = self.y[train_mask]
        self.X_test = self.X[test_mask]
        self.y_test = self.y[test_mask]
        self.groups_train = self.groups[train_mask]
        
        return train_ids, test_ids

    def cross_validate(self, max_depth_range=range(1, 21)):
        """cross-validation with different depth values"""
        cv = GroupKFold(n_splits=5)
        summary = []

        for depth in max_depth_range:
            clf = make_pipeline(
                StandardScaler(),
                DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
            )
            auc_scores = cross_val_score(
                clf, self.X_train, self.y_train, 
                cv=cv, groups=self.groups_train, scoring='roc_auc'
            )
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            print(f"max_depth={depth}, AUC={mean_auc:.4f} ± {std_auc:.4f}")
            summary.append({
                'max_depth': depth, 
                'mean_auc': mean_auc, 
                'std_auc': std_auc
            })
        
        self.cv_results = pd.DataFrame(summary)
        return self.cv_results

    def train_and_evaluate(self):
        """train the model and evaluate on test set"""
        # Train the model
        self.pipeline.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(self.X_test)
        y_prob = self.pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(self.y_test, y_pred, y_prob)
        
        # Store results
        results_df = pd.DataFrame({
            "patient_id": self.groups[self.data['patient_id'].isin(self.y_test.index)],
            "img_id": self.data.loc[self.y_test.index, 'img_id'],
            "true_label": self.y_test,
            "predicted_label": y_pred,
            "probability": y_prob
        })
        
        return metrics, results_df, y_pred, y_prob

    def plot_results(self, y_pred, save_dir="result"):
        """plot and save results"""
        # Plot confusion matrix
        plot_confusion_matrix(
            self.y_test, y_pred,
            title=f"Confusion Matrix (max_depth={self.max_depth})",
            save_path=f"{save_dir}/confusion_matrix.png"
        )
        
        #plot cross-validation results
        if hasattr(self, 'cv_results'):
            plot_cross_validation_results(
                self.cv_results,
                save_path=f"{save_dir}/cross_validation_results.png"
            )

def main():
    # Example usage
    model = DecisionTreeModel(max_depth=4)
    
    #load data and create train-test split
    train_ids, test_ids = model.train_test_split('result/enhanced_feature_dataset.csv')
    
    #perform cross-validation
    cv_results = model.cross_validate()
    cv_results.to_csv("result/cross_val_DT_baseline.csv", index=False)
    
    #train and evaluate
    metrics, results_df, y_pred, y_prob = model.train_and_evaluate()
    
    #print results
    print("\nMetrics on test set:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    #olot results
    model.plot_results(y_pred)
    
    #save
    results_df.to_csv("result/result_baseline_fixed.csv", index=False)

if __name__ == "__main__":
    main()