'''
Created Date: Thursday, August 28th 2025, 10:19:09 am
Author: Liming Xu

Copyright (c) 2025 SCAIL, IfM, University of Cambridge
'''
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from catboost import CatBoostClassifier, Pool


# Data
df = pd.read_csv("data/syndelay_v1.csv")

# Features + label
X = df.drop("label", axis=1)
y = df["label"]

# Identify categorical, text, and datetime features

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
text_features = [
    "category_name",
    # "customer_zipcode",
    "department_name",
    "product_name",
]  
datetime_features = ["shipping_date"]

# CatBoost requires indices of categorical/text/datetime features
cat_feature_indices = [X.columns.get_loc(c) for c in cat_features]
text_feature_indices = [X.columns.get_loc(c) for c in text_features]
datetime_feature_indices = [X.columns.get_loc(c) for c in datetime_features]


# Train the model and use it for prediction
def train_and_predict(X, y):
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp
    )

    # Pool objects (CatBoost’s way of handling mixed features)
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_feature_indices,
        text_features=text_feature_indices,
        # datetime_features=datetime_feature_indices,
    )
    val_pool = Pool(
        data=X_val,
        label=y_val,
        cat_features=cat_feature_indices,
        text_features=text_feature_indices,
        # datetime_features=datetime_feature_indices,
    )

    # Define model
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        eval_metric="TotalF1:average=Weighted",  # CatBoost's built-in weighted F1
        verbose=100,
        early_stopping_rounds=50,
    )

    # Train
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Evaluate on test set
    test_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=cat_feature_indices,
        text_features=text_feature_indices,
        # datetime_features=datetime_feature_indices,
    )
    y_pred = model.predict(X_test)

    return y_test, y_pred


# %%
# -------------------
# One run
# -------------------
y_test, y_pred = train_and_predict(X, y)
print("CatBoost:")
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
for i in tqdm(range(n_runs), desc="[CatBoost] Running repetitions"):

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
