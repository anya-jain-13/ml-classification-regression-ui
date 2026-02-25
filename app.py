import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ------------------- Helpers -------------------
def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def guess_target_type(y: pd.Series) -> str:
    """
    Heuristic:
    - If non-numeric -> classification
    - If numeric:
        - if few unique values relative to length -> classification
        - else -> regression
    """
    y = y.dropna()
    if y.empty:
        return "unknown"

    if not is_numeric_series(y):
        return "classification"

    n = len(y)
    uniq = y.nunique()

    # Numeric target but very few unique values => likely classes (discrete)
    if uniq <= 20 and uniq / max(n, 1) <= 0.2:
        return "classification"
    return "regression"


def looks_like_id_column(col_name: str, y: pd.Series) -> bool:
    name = col_name.lower()
    id_keywords = ["id", "uuid", "guid", "index", "serial"]

    # Name-based check
    if any(k in name for k in id_keywords):
        return True

    # Value-based check: mostly unique values (like IDs)
    y_non_null = y.dropna()
    if len(y_non_null) > 0:
        uniqueness_ratio = y_non_null.nunique() / len(y_non_null)
        if uniqueness_ratio > 0.9:
            return True

    return False


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].isna().any():
            if pd.api.types.is_numeric_dtype(df2[col]):
                df2[col] = df2[col].fillna(df2[col].mean())
            else:
                mode_val = df2[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else ""
                df2[col] = df2[col].fillna(fill_val)
    return df2


def one_hot_encode(X: pd.DataFrame) -> pd.DataFrame:
    X_enc = pd.get_dummies(X, drop_first=False)
    X_enc = X_enc.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X_enc


def validate_nb_model(model_name: str, X: pd.DataFrame):
    X_np = X.to_numpy()

    if model_name == "MultinomialNB":
        if np.any(X_np < 0):
            st.error("❌ MultinomialNB requires **non-negative** features (counts). Your dataset has negative values.")
            st.stop()

    if model_name == "BernoulliNB":
        unique_vals = np.unique(X_np)
        if not set(unique_vals).issubset({0, 1}):
            st.error("❌ BernoulliNB requires **binary** features (0/1). Your features are not strictly 0/1.")
            st.stop()


# ------------------- UI -------------------
st.set_page_config(page_title="ML Lab App (Naive Bayes + Regression)", layout="wide")
st.title("ML Lab App: Naive Bayes (Classification) + Regression")
st.caption("Upload CSV → select target → choose task/model → get metrics + confusion matrix (classification).")

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file from the sidebar to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.shape[0] < 10 or df.shape[1] < 2:
    st.error("Dataset too small. Need at least ~10 rows and 2 columns.")
    st.stop()

st.subheader("Dataset Preview")
st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.dataframe(df.head(20), use_container_width=True)

target_col = st.sidebar.selectbox("Select Target (Label) Column", df.columns.tolist())

missing_mode = st.sidebar.selectbox(
    "Missing values handling",
    ["Drop rows with missing", "Fill numeric mean + categorical mode"],
    index=0,
)

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 10000, 42, 1)

# Clean missing values
if missing_mode == "Drop rows with missing":
    df_clean = df.dropna().copy()
else:
    df_clean = fill_missing(df)

if df_clean.shape[0] < 10:
    st.error("After missing value handling, dataset became too small.")
    st.stop()

y = df_clean[target_col]

# Block ID-like target columns
if looks_like_id_column(target_col, y):
    st.error("❌ The selected target column looks like an ID/identifier. Please choose a real label/target column (not an ID).")
    st.stop()

X_raw = df_clean.drop(columns=[target_col])

# Auto detect task based on target
auto_task = guess_target_type(y)

st.sidebar.subheader("Task Selection")
task = st.sidebar.selectbox(
    "Task Type",
    ["classification", "regression"],
    index=0 if auto_task != "regression" else 1,
)

st.write("### Target analysis")
st.write(f"- Target column: **{target_col}**")
st.write(f"- Auto-detected target type: **{auto_task}**")
st.write(f"- You selected task type: **{task}**")

if auto_task == "unknown":
    st.error("Target column has no usable values after cleaning.")
    st.stop()

if task == "regression" and auto_task == "classification":
    st.error("❌ You selected **Regression**, but your target looks **discrete/categorical**. Choose **Classification**.")
    st.stop()

if task == "classification" and auto_task == "regression":
    st.error("❌ You selected **Classification**, but your target looks **continuous numeric**. Choose **Regression**.")
    st.stop()

# Encode features
X = one_hot_encode(X_raw)

# Split data
stratify_arg = None
if task == "classification" and y.nunique() > 1:
    if y.value_counts().min() >= 2:
        stratify_arg = y

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=stratify_arg
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )


# ------------------- Classification -------------------
if task == "classification":
    st.sidebar.subheader("Model (Classification)")
    clf_name = st.sidebar.selectbox("Choose model", ["GaussianNB", "MultinomialNB", "BernoulliNB"], index=0)

    validate_nb_model(clf_name, X_train)

    if clf_name == "GaussianNB":
        model = GaussianNB()
    elif clf_name == "MultinomialNB":
        model = MultinomialNB()
    else:
        model = BernoulliNB()

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    st.subheader("Results (Classification)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Accuracy", f"{train_acc:.4f}")
    c2.metric("Testing Accuracy", f"{test_acc:.4f}")
    c3.metric("Classes", str(y.nunique()))

    st.subheader("Confusion Matrix (Test Set)")
    labels = np.unique(y)
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report (Test Set)")
    rep = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)


# ------------------- Regression -------------------
else:
    st.sidebar.subheader("Model (Regression)")
    reg_name = st.sidebar.selectbox("Choose model", ["LinearRegression", "RandomForestRegressor"], index=0)

    if not is_numeric_series(y):
        st.error("❌ Regression requires numeric target, but your target column is not numeric.")
        st.stop()

    if reg_name == "LinearRegression":
        model = LinearRegression()
    else:
        n_estimators = st.sidebar.slider("RF n_estimators", 50, 500, 200, 50)
        model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state))

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    st.subheader("Results (Regression)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train RMSE", f"{train_rmse:.4f}")
    c2.metric("Test RMSE", f"{test_rmse:.4f}")
    c3.metric("Test R²", f"{test_r2:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Train MAE", f"{train_mae:.4f}")
    c5.metric("Test MAE", f"{test_mae:.4f}")
    c6.metric("Train R²", f"{train_r2:.4f}")

    st.subheader("Actual vs Predicted (first 20 rows of test)")
    out_df = pd.DataFrame({"Actual": y_test.values[:20], "Predicted": test_pred[:20]})
    st.dataframe(out_df, use_container_width=True)