import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.preprocessing import LabelEncoder

# BASE_DIR is wherever this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use correct path (NO duplicate folders)
csv_path = os.path.join(BASE_DIR, 'crop_disease_data.csv')
model_path = os.path.join(BASE_DIR, 'model.pkl')

# Load dataset
df = pd.read_csv(csv_path)

# Features and label
X = df[['symptoms', 'crop']]
y = df['disease']

# Encode features
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)

# Train model
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# Save model
with open(model_path, "wb") as f:
    pickle.dump(model, f)
