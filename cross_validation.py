import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import pandas as pd
from pathlib import Path
from util.classifier_final import cross_validation

def main():

    base_dir = Path(__file__).parent.resolve()
    feature_path = base_dir / "result" / "feature_dataset.csv"
    
    try:

        data = pd.read_csv(feature_path)
        print(f"Loaded data from {feature_path}")
        print(f"Total samples: {len(data)}")
        
        features = [
            'border', 'asymmetry', 'mean_H', 'std_H', 
            'mean_S', 'std_S', 'mean_V', 'std_V', 
            'color_entropy', 'melanoma_colors'
        ]
        

        print("\nPerforming cross-validation...")
        max_depth = 21  
        summary_df, cancer_balance = cross_validation(
            data=data,
            features=features,
            max_depth=max_depth,
            visualize=True,  
            save=True     
        )
        

        print(f"\nCancer cases in training set: {cancer_balance}%")
        print("\nCross-validation results:")
        print(summary_df)
        

        best_result = summary_df.loc[summary_df['mean_auc'].idxmax()]
        print(f"\nBest performance:")
        print(f"Tree depth: {best_result['max_depth']}")
        print(f"Mean AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find feature dataset at {feature_path}")
    except Exception as e:
        print(f"Error during cross-validation: {str(e)}")

if __name__ == "__main__":
    main()