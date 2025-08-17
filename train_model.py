import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import joblib

# ======================
# Load dataset
# ======================
# Example dataset - replace with your own smartphone dataset CSV
# Dataset must contain: Brand, Model, RAM, Storage, 5G, Price
df = pd.read_csv("Smart_phn_data.csv", on_bad_lines='skip')

# Features and target
X = df[["Brand", "Model", "RAM", "Storage", "5G"]]
y = df["Price"]

# Preprocessing
categorical_features = ["Brand", "Model", "5G"]
numeric_features = ["RAM", "Storage"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

# Model pipeline (Ridge regression here)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "smartphone_price_model.pkl")

print("✅ Model trained and saved as smartphone_price_model.pkl")

