# %% [markdown]
# # Testing K nearest neigbours

# %% [markdown]
# Using the features file to separate into labels and features

# %%
import pandas as pd
import numpy as np

data = pd.read_csv('./util/feature_dataset.csv') 
data.fillna(0, inplace=True)
X = data[['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V']]

data['cancer'] = data['cancer'].astype(int)
y = data['cancer']
groups = data['patient_id']
unique_patients = data['patient_id'].unique()

# %% [markdown]
# Running cross-validation over different k values (get AUC)

# %%
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

#split data into training and test sets (by patient ID)
unique_patients = data['patient_id'].unique()
train_ids, test_ids = train_test_split(unique_patients, test_size=0.2, random_state=42)
train_idx = data['patient_id'].isin(train_ids)

X_train, y_train = X[train_idx], y[train_idx]
groups_train = groups[train_idx]

#standardScaler on training data (within pipeline below)
#GroupKFold CV with k from 3 to 10
cv = GroupKFold(n_splits=5)
k_values = range(1, 101, 2)
summary = []

for k in k_values:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, groups=groups_train, scoring='roc_auc')
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"k={k}, AUC={mean_auc:.4f} ± {std_auc:.4f}")
    summary.append({'k': k, 'mean_auc': mean_auc, 'std_auc': std_auc})

summary_df = pd.DataFrame(summary)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.errorbar(summary_df['k'], summary_df['mean_auc'], yerr=summary_df['std_auc'], 
             fmt='-o', capsize=5, color='blue')
plt.title('Mean AUC with Standard Deviation for Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('AUC Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Chosing k=25 to train the model

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Create test mask similar to train mask
test_mask = data['patient_id'].isin(test_ids)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_mask], y[test_mask]
groups_train = groups[train_idx]
patient_ids_test = groups[test_mask]

# Fit final model on training data (best k = 25)
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))
pipe.fit(X_train, y_train)

# Predict on test set
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]  # Probability of cancer

# Metrics
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Final Evaluation on Test Set:")
print(f"Accuracy: {acc:.4f}")
print(f"Recall: {rec:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Benign", "Cancer"],
    cmap="Greens",
    values_format="d"
)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

results_df = pd.DataFrame({
    "patient_id": patient_ids_test.values,
    "true_label": y_test.values,
    "cancer_probability": y_prob
})

# results_df.to_csv("knn_final_test_probs.csv", index=False)

# %%
results_df[results_df['cancer_probability']>0.70].head(30)

# %% [markdown]
# Checking features

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=data, 
    x='border', 
    y='mean_H', 
    hue='cancer', 
    palette='coolwarm', 
    alpha=0.7
)
plt.title("2D Feature Space: Asymmetry vs Border")
plt.xlabel("Boarder")
plt.ylabel("mean_h")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot
ax.scatter(
    data['asymmetry'], 
    data['border'], 
    data['mean_H'], 
    c=data['cancer'], 
    cmap='coolwarm', 
    alpha=0.7
)

ax.set_title("3D Feature Space: Asymmetry, Border, mean_H")
ax.set_xlabel("Asymmetry")
ax.set_ylabel("Border")
ax.set_zlabel("mean_H")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Patient Distribution

# %%
# Analyze patient distribution
patient_counts = data['patient_id'].value_counts()

print(f"Total number of unique patients: {len(patient_counts)}")
print(f"Average images per patient: {patient_counts.mean():.2f}")
print(f"Max images per patient: {patient_counts.max()}")
print(f"Min images per patient: {patient_counts.min()}")

# Show distribution of number of images per patient
plt.figure(figsize=(10, 5))
patient_counts.hist(bins=20)
plt.title('Distribution of Images per Patient')
plt.xlabel('Number of Images')
plt.ylabel('Number of Patients')
plt.show()

# %% [markdown]
# ## Train-Test Split
# function for keeping all images from a patient in one group

# %%
from sklearn.model_selection import train_test_split

# First, let's verify our split maintains patient separation
def verify_patient_split(train_ids, test_ids):
    intersection = set(train_ids) & set(test_ids)
    if len(intersection) > 0:
        print(f"Warning: {len(intersection)} patients appear in both sets!")
        return False
    return True

# Split patients into train and test sets
unique_patients = data['patient_id'].unique()
train_ids, test_ids = train_test_split(unique_patients, test_size=0.2, random_state=42)

# Verify the split
is_valid_split = verify_patient_split(train_ids, test_ids)
print(f"Split is valid: {is_valid_split}")

# Create train and test masks
train_mask = data['patient_id'].isin(train_ids)
test_mask = data['patient_id'].isin(test_ids)

# Print split statistics
print(f"\nTraining set:")
print(f"Number of patients: {len(train_ids)}")
print(f"Number of images: {train_mask.sum()}")

print(f"\nTest set:")
print(f"Number of patients: {len(test_ids)}")
print(f"Number of images: {test_mask.sum()}")

# Create the actual split datasets
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
groups_train = groups[train_mask]

# %% [markdown]
# # Decision tree

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


max_depth_values = range(1, 21) 
cv = GroupKFold(n_splits=5)

summary = []

for depth in max_depth_values:
    clf = make_pipeline(
        StandardScaler(),  # Optional for DecisionTree but included for consistency
        DecisionTreeClassifier(max_depth=depth, random_state=42)
    )

    auc_scores = cross_val_score(
        clf, X_train, y_train, cv=cv, groups=groups_train, scoring='roc_auc'
    )
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"max_depth={depth}, AUC={mean_auc:.4f} ± {std_auc:.4f}")
    summary.append({'max_depth': depth, 'mean_auc': mean_auc, 'std_auc': std_auc})

summary_df = pd.DataFrame(summary)


# %%
import matplotlib.pyplot as plt

plt.errorbar(summary_df['max_depth'], summary_df['mean_auc'], yerr=summary_df['std_auc'], fmt='-o')
plt.xlabel("Tree Depth")
plt.ylabel("Mean AUC (±1 std)")
plt.title("Decision Tree Cross-Validation AUC")
plt.grid(True)
plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, 
    precision_score, f1_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd

# Use same data split
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_mask], y[test_mask]
patient_ids_test = groups[test_mask]

# Create pipeline with decision tree (depth=3)
pipe = make_pipeline(
    StandardScaler(),  # Optional for tree, included for consistency
    DecisionTreeClassifier(max_depth=3, random_state=42)
)

# Fit model
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Final Evaluation on Test Set (Decision Tree):")
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix plot
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Benign", "Cancer"],
    cmap="Oranges",
    values_format="d"
)
plt.title("Decision Tree (depth=3) – Confusion Matrix")
plt.grid(False)
plt.show()

# Save results
results_df = pd.DataFrame({
    "patient_id": patient_ids_test.values,
    "true_label": y_test.values,
    "cancer_probability": y_prob
})
# results_df.to_csv("decision_tree_test_probs.csv", index=False)



