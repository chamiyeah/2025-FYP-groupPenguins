import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_data = 'fyp2022-imaging/data/example_ground_truth.csv'
file_features = 'fyp2022-imaging/features/features.csv'

df = pd.read_csv(file_data)
features = pd.read_csv(file_features)

# Combine variables we want in one place
df = df.drop(['image_id','seborrheic_keratosis'],axis=1)
df['area'] = features['area']
df['perimeter'] = features['perimeter']

# Please remember that area and perimeter alone are often not sufficient for classification.
# When doing your project, you could also try the other features here.

print(df.head())