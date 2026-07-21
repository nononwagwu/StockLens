import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="StockLens",
    page_icon="📈",
    layout="wide"
)

st.title("StockLens")
st.write(
    "Explore machine learning predictions and financial insights "
    "for companies in our dataset."
)


# --------------------------------------------------
# LOAD MODEL COMPONENTS
# --------------------------------------------------


def load_model():
    model = joblib.load("models/random_forest.pkl")
    imputer = joblib.load("models/imputer.pkl")
    features = joblib.load("models/features.pkl")

    return model, imputer, features


model, imputer, features = load_model()
@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)



# --------------------------------------------------
# LOAD COMPANY DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/2018_Financial_Data.csv")

    # The first column contains the company ticker
    df = df.rename(columns={"Unnamed: 0": "Ticker"})
    company_names = pd.read_csv("data/company_names.csv")

    df = df.merge(
        company_names,
        on="Ticker",
        how="left"
    )

    # Standardize target column name
    price_columns = [
        col for col in df.columns
        if "PRICE VAR" in col.upper()
    ]

    if price_columns:
        df = df.rename(
            columns={price_columns[0]: "PRICE VAR [%]"}
        )

    return df


df = load_data()

# --------------------------------------------------
# COMPANY SELECTION
# --------------------------------------------------

st.header("Company Analysis")

# Create readable labels
df["Display Name"] = df.apply(
    lambda row: (
        f"{row['Company Name']} ({row['Ticker']})"
        if pd.notna(row["Company Name"])
        and row["Company Name"] != row["Ticker"]
        else row["Ticker"]
    ),
    axis=1
)

company_options = (
    df[["Ticker", "Display Name"]]
    .drop_duplicates()
    .sort_values("Display Name")
)

selected_display = st.selectbox(
    "Search or select a company:",
    company_options["Display Name"],
    index=None,
    placeholder="Search by company name or ticker..."
)

selected_ticker = None

if selected_display:
    selected_ticker = company_options.loc[
        company_options["Display Name"] == selected_display,
        "Ticker"
    ].iloc[0]


# --------------------------------------------------
# RUN COMPANY ANALYSIS
# --------------------------------------------------

if selected_ticker:

    company = df[
        df["Ticker"].astype(str) == selected_ticker
    ].iloc[0]

    company_name = company.get(
        "Company Name",
        selected_ticker
    )

    st.subheader(
        f"{company_name} ({selected_ticker})"
    )

    # Get the exact features expected by the model
    company_features = pd.DataFrame(
        [company.reindex(features)]
    )

    # Replace infinity values
    company_features = company_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Apply the same preprocessing used during training
    company_features_imputed = imputer.transform(
        company_features
    )

    # Generate prediction
    prediction = model.predict(
        company_features_imputed
    )[0]


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    actual_return = company.get(
        "PRICE VAR [%]",
        np.nan
    )

    if pd.notna(actual_return):
        prediction_error = abs(
            actual_return - prediction
        )
    else:
        prediction_error = np.nan

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Predicted 12-Month Return",
            f"{prediction:.2f}%"
        )

    with col2:
        if pd.notna(actual_return):
            st.metric(
                "Actual 12-Month Return",
                f"{actual_return:.2f}%"
            )
        else:
            st.metric(
                "Actual 12-Month Return",
                "N/A"
            )

    with col3:
        if pd.notna(prediction_error):
            st.metric(
                "Prediction Error",
                f"{prediction_error:.2f}%"
            )
        else:
            st.metric(
                "Prediction Error",
                "N/A"
            )

    with col4:
        sector = company.get(
            "Sector",
            "Unknown"
        )

        st.metric(
            "Sector",
            str(sector)
        )

    
        # --------------------------------------------------
    # COMPANY FINANCIAL SNAPSHOT
    # --------------------------------------------------

    st.subheader("Company Financial Snapshot")

    snapshot_features = [
        "Revenue",
        "Net Income",
        "EPS",
        "ROE",
        "ROA",
        "Debt to Equity"
    ]

    
    def format_financial_value(metric, value):
        if metric in ["Revenue", "Net Income"]:
            if abs(value) >= 1_000_000_000:
                return f"${value / 1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000:
                return f"${value / 1_000_000:.2f}M"
            else:
                return f"${value:,.2f}"

        elif metric == "EPS":
            return f"${value:.2f}"

        elif metric in ["ROE", "ROA"]:
            return f"{value * 100:.2f}%"

        elif metric == "Debt to Equity":
            return f"{value:.2f}"

        return f"{value:,.2f}"

    available_snapshot = {
        feature: format_financial_value(
            feature,
            company[feature]
        )
        for feature in snapshot_features
        if feature in company.index
        and pd.notna(company[feature])
    }

    if available_snapshot:
        snapshot_df = pd.DataFrame(
            available_snapshot.items(),
            columns=["Financial Metric", "Value"]
        )

        st.dataframe(
            snapshot_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info(
            "Financial snapshot metrics are unavailable "
            "for this company."
        )

        # --------------------------------------------------
    # COMPANY-SPECIFIC MODEL EXPLANATION
    # --------------------------------------------------

    st.subheader("Why Did the Model Make This Prediction?")

    # Create a SHAP explainer for the Random Forest
    explainer = load_shap_explainer(model)

    # Calculate SHAP values for the selected company
    shap_values = explainer.shap_values(
        company_features_imputed
    )

    # Create a DataFrame containing each feature's impact
    shap_df = pd.DataFrame({
        "Financial Feature": features,
        "Impact": shap_values[0]
    })

    # Find the features with the largest impact
    shap_df["Absolute Impact"] = shap_df[
        "Impact"
    ].abs()

    top_shap = (
        shap_df
        .sort_values(
            "Absolute Impact",
            ascending=False
        )
        .head(10)
        .sort_values("Impact")
    )

    st.write(
        "These financial features had the strongest influence "
        f"on the model's prediction for {company_name}."
    )

    st.bar_chart(
        top_shap.set_index(
            "Financial Feature"
        )["Impact"]
    )

    st.caption(
        "Positive values pushed the predicted return higher. "
        "Negative values pushed it lower."
    )

    # --------------------------------------------------
    # MODEL INSIGHTS
    # --------------------------------------------------

    st.subheader("Model Insights")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Financial Feature": features,
        "Importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    st.write(
        "Top 10 financial features influencing "
        "the Random Forest model:"
    )

    st.bar_chart(
        importance_df.set_index(
            "Financial Feature"
        )
    )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "About StockLens"
)

st.sidebar.write(
    """
    StockLens uses a Random Forest regression model
    trained on historical financial data from
    2014–2017 to estimate 12-month stock returns.

    The application uses 2018 company financial
    data for model analysis and demonstration.
    """
)