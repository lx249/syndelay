'''
Created Date: Thursday, August 28th 2025, 2:04:20 pm
Author: Liming Xu

Copyright (c) 2025 SCAIL, IfM, University of Cambridge
'''
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)

# -------------------
# Data preparation
# -------------------
df = pd.read_csv("data/syndelay_v1.csv")

X = df.drop("label", axis=1)
y = df["label"]


# Identify feature types
num_features = [
    "profit_per_order",
    "sales_per_customer",
    "category_id",
    "customer_id",
    "customer_zipcode",
    "department_id",
    "latitude",
    "longitude",
    "order_customer_id",
    "order_date",
    "order_id",
    "order_item_cardprod_id",
    "order_item_discount",
    "order_item_discount_rate",
    "order_item_id",
    "order_item_product_price",
    "order_item_profit_ratio",
    "order_item_quantity",
    "sales",
    "order_item_total_amount",
    "order_profit_per_order",
    "product_card_id",
    "product_category_id",
    "product_price",
    "shipping_date",
]
cat_features = [
    "payment_type",
    "customer_city",
    "customer_country",
    "customer_segment",
    "customer_state",
    "market",
    "order_city",
    "order_country",
    "order_region",
    "order_state",
    "order_status",
    "shipping_mode",
]


# Train the model and use it for prediction
def train_and_predict(X, y):
    # -------------------
    # Train/val/test split
    # -------------------
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp
    )

    # -------------------
    # Preprocessing
    # -------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    # -------------------
    # Random Forest model
    # -------------------
    rf_clf = RandomForestClassifier(
        n_estimators=500,  # more trees → better stability
        max_depth=None,  # let trees grow fully
        min_samples_split=5,  # avoid overfitting on tiny splits
        min_samples_leaf=2,  # smoother decision boundaries
        max_features="sqrt",  # common best practice
        class_weight="balanced",  # handle imbalanced labels
        n_jobs=-1,  # use all cores
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf_clf)])

    # -------------------
    # Train model
    # -------------------
    pipeline.fit(X_train, y_train)

    # -------------------
    # Predictions
    # -------------------
    y_pred = pipeline.predict(X_test)

    return y_test, y_pred


# %%
# -------------------
# One run
# -------------------
y_test, y_pred = train_and_predict(X, y)
print("Random Forest:")
print(classification_report(y_test, y_pred))


# %%
# -------------------
# Repeated 10 times
# -------------------
n_runs = 10

# Store results
acc_list = []
macro_f1_list = []
weighted_f1_list = []

f1_class2_list = []
precision_class2_list = []
recall_class2_list = []

# Loop with progress bar
for i in tqdm(range(n_runs), desc="[Random Forest] Running repetitions"):

    # Train and predict
    y_test, y_pred = train_and_predict(X, y)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    f1_class2 = f1_score(y_test, y_pred, labels=[2], average=None)[0]
    precision_class2 = precision_score(y_test, y_pred, labels=[2], average=None)[0]
    recall_class2 = recall_score(y_test, y_pred, labels=[2], average=None)[0]

    # Append to list
    acc_list.append(accuracy)
    macro_f1_list.append(macro_f1)
    weighted_f1_list.append(weighted_f1)

    f1_class2_list.append(f1_class2)
    precision_class2_list.append(precision_class2)
    recall_class2_list.append(recall_class2)


# Compute mean and std
def mean_std(lst):
    return np.mean(lst), np.std(lst)


acc_mean, acc_std = mean_std(acc_list)
macro_mean, macro_std = mean_std(macro_f1_list)
weighted_mean, weighted_std = mean_std(weighted_f1_list)

f1_class2_mean, f1_class2_std = mean_std(f1_class2_list)
precision_class2_mean, precision_class2_std = mean_std(precision_class2_list)
recall_class2_mean, recall_class2_std = mean_std(recall_class2_list)

print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Macro F1: {macro_mean:.4f} ± {macro_std:.4f}")
print(f"Weighted F1: {weighted_mean:.4f} ± {weighted_std:.4f}")

print(f"F1 (class 2): {f1_class2_mean:.4f} ± {f1_class2_std:.4f}")
print(f"Precision (class 2): {precision_class2_mean:.4f} ± {precision_class2_std:.4f}")
print(f"Recall (class 2): {recall_class2_mean:.4f} ± {recall_class2_std:.4f}")
