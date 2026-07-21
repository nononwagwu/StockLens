import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


print("Starting StockLens model training...")

# --------------------------------------------------
# 1. LOAD TRAINING DATA (2014-2017)
# --------------------------------------------------

train_files = [
    "data/2014_Financial_Data.csv",
    "data/2015_Financial_Data.csv",
    "data/2016_Financial_Data.csv",
    "data/2017_Financial_Data.csv",
]

train_dfs = []

for file in train_files:
    print(f"Loading {file}...")

    df = pd.read_csv(file)

    # Standardize target column name
    price_var_col = [
        col for col in df.columns
        if "PRICE VAR" in col.upper()
    ]

    if price_var_col:
        df = df.rename(
            columns={price_var_col[0]: "PRICE VAR [%]"}
        )

    train_dfs.append(df)


# Combine all training years
train_df = pd.concat(
    train_dfs,
    ignore_index=True
)

print(
    f"Loaded {len(train_df)} total training companies."
)


# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------

# Replace infinity values with NaN
train_df = train_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# Target variable
y_train = train_df["PRICE VAR [%]"].fillna(0)


# Remove extreme returns / penny-stock outliers
valid_rows = (
    (y_train > -90)
    & (y_train < 200)
)

train_df = train_df.loc[valid_rows].copy()
y_train = y_train.loc[valid_rows]


print(
    f"Training on {len(train_df)} companies "
    "after outlier filtering."
)


# --------------------------------------------------
# 3. PREPARE FEATURES
# --------------------------------------------------

columns_to_drop = [
    "PRICE VAR [%]",
    "class",
    "Class",
    "Sector",
    "Unnamed: 0"
]


X_train = train_df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# Keep only numeric financial features
X_train = X_train.select_dtypes(
    include=[np.number]
)


# Save feature names BEFORE converting with imputer
feature_names = X_train.columns.tolist()


print(
    f"Model uses {len(feature_names)} financial features."
)


# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)


X_train_imputed = imputer.fit_transform(
    X_train
)


# --------------------------------------------------
# 5. TRAIN RANDOM FOREST
# --------------------------------------------------

print("Training Random Forest model...")


rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)


rf_model.fit(
    X_train_imputed,
    y_train
)


print("Model training complete.")


# --------------------------------------------------
# 6. SAVE MODEL FILES
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)


joblib.dump(
    imputer,
    "models/imputer.pkl"
)


joblib.dump(
    feature_names,
    "models/features.pkl"
)


print("\nSaved:")

print(
    "models/random_forest.pkl"
)

print(
    "models/imputer.pkl"
)

print(
    "models/features.pkl"
)


print(
    "\nStockLens model is ready for Streamlit."
)