import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data, build_preprocessor
from sklearn.pipeline import Pipeline

# Load raw data

raw_data = pd.read_csv("traffic_accidents.csv")
prepped_data = preprocess_data(raw_data)

# Downsample dataset to 10%

DOWNSAMPLE_FRACTION = 0.10    
prepped_data = prepped_data.sample(frac=DOWNSAMPLE_FRACTION, random_state=42)
print(f"Dataset downsampled to {len(prepped_data)} rows.")

# 70/15/15 split

X = prepped_data.drop(columns=["injury_severity"], axis=1)
y = prepped_data["injury_severity"]

# First split: 15% test, 85% temp_train
x_temp, x_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    shuffle=True,
    random_state=42
)

# Second split: from the 85% chunk,
# create EXACT 70/15 split from full dataset.
# Validation should be 15% of N:
# val_fraction_within_temp = 0.15 / 0.85 ≈ 0.17647

VAL_FRACTION = 0.15 / 0.85

X_train, X_val, y_train, y_val = train_test_split(
    x_temp, y_temp,
    test_size=VAL_FRACTION,
    shuffle=True,
    random_state=42
)

# Pre-Processor Build

preprocessor = build_preprocessor(X_train)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
        # model stuff will go here
    ]
)

# Modeling pipeline

from models import train_and_evaluate_all_models

# Model training + eval

best_model, test_metrics = train_and_evaluate_all_models(
    X_train,
    X_val,
    x_test,
    y_train,
    y_val,
    y_test,
    preprocessor=preprocessor
)

# Summary output
print("\n================ FINAL RESULTS ================\n")

# Extract simple name
best_model_name = type(best_model.named_steps["clf"]).__name__

print(f"Best Model Selected: {best_model_name}\n")

print("Test Set Performance:")
print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall   : {test_metrics['recall']:.4f}")
print(f"  F1 Score : {test_metrics['f1']:.4f}")

print("\n===============================================\n")

