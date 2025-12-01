import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 California Housing Price Predictor")
st.markdown("""
This app predicts housing prices in California based on various features.
The model is trained on the California Housing Dataset (OpenML version).
""")

@st.cache_data
def load_data():
    df = pd.read_csv("california_housing_train.csv")
    df["target"] = df["median_house_value"]           # main target
    df["target_log"] = np.log1p(df["median_house_value"])  # log target
    return df, df.columns.tolist()

@st.cache_resource
def train_model(X_train, y_train):
    """Train a Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    housing_data, feature_names = load_data()

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode",
                                   ["Data Exploration", "Model Training", "Price Prediction"])

    if app_mode == "Data Exploration":
        show_data_exploration(housing_data, feature_names)

    elif app_mode == "Model Training":
        show_model_training(housing_data, feature_names)

    elif app_mode == "Price Prediction":
        show_price_prediction(housing_data, feature_names)


def show_data_exploration(housing_data, feature_names):
    st.header("📊 Data Exploration")

    if st.checkbox("Show Dataset Info"):
        st.subheader("Dataset Information")
        st.write(f"Shape: {housing_data.shape}")
        st.dataframe(housing_data.head())

        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Types:")
            st.write(housing_data.dtypes)
        with col2:
            st.write("Missing Values:")
            st.write(housing_data.isna().sum())

    if st.checkbox("Show Statistics"):
        st.subheader("Descriptive Statistics")
        st.dataframe(housing_data.describe())

    if st.checkbox("Show Target Variable Distribution"):
        st.subheader("Target Variable Distribution")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Original target
        ax1.hist(housing_data["target"], bins=50, alpha=0.7)
        ax1.set_xlabel("Median House Value ($100k)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Original Target Distribution")

        # Log target
        ax2.hist(housing_data["target_log"], bins=50, alpha=0.7)
        ax2.set_xlabel("Log(Median House Value)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Log-Transformed Target Distribution")

        st.pyplot(fig)


def show_model_training(housing_data, feature_names):
    st.header("🤖 Model Training")

    st.subheader("Feature Selection")
    selected_features = st.multiselect(
        "Choose features for the model:",
        feature_names,
        default=feature_names
    )

    if not selected_features:
        st.warning("Please select at least one feature!")
        return

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        use_log_target = st.checkbox("Use Log-Transformed Target", value=True)

    X = housing_data[selected_features]
    y = housing_data["target_log"] if use_log_target else housing_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model = train_model(X_train, y_train)

            y_pred = model.predict(X_test)

            if use_log_target:
                y_test_actual = np.expm1(y_test)
                y_pred_actual = np.expm1(y_pred)
            else:
                y_test_actual = y_test
                y_pred_actual = y_pred

            mse = mean_squared_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_actual, y_pred_actual)

            st.success("Model trained successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("MSE", f"{mse:.4f}")
            col3.metric("R² Score", f"{r2:.4f}")

            # Feature importance
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                "feature": selected_features,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance, x="importance", y="feature", ax=ax)
            st.pyplot(fig)

            # Prediction vs Actual
            st.subheader("Prediction vs Actual")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test_actual, y_pred_actual, alpha=0.5)
            ax.plot([y_test_actual.min(), y_test_actual.max()],
                    [y_test_actual.min(), y_test_actual.max()], "r--")
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            st.pyplot(fig)

            # Save model
            st.session_state["model"] = model
            st.session_state["selected_features"] = selected_features
            st.session_state["use_log_target"] = use_log_target


def show_price_prediction(housing_data, feature_names):
    st.header("🔮 Price Prediction")

    if "model" not in st.session_state:
        st.warning("Please train a model first!")
        return

    model = st.session_state["model"]
    selected_features = st.session_state["selected_features"]
    use_log_target = st.session_state["use_log_target"]

    st.subheader("Enter House Features")

    input_data = {}
    stats = housing_data[selected_features].describe()

    cols = st.columns(2)
    for i, feature in enumerate(selected_features):
        with cols[i % 2]:
            min_val = float(stats.loc["min", feature])
            max_val = float(stats.loc["max", feature])
            mean_val = float(stats.loc["mean", feature])

            input_data[feature] = st.slider(
                feature, min_val, max_val, mean_val, step=(max_val - min_val) / 100
            )

    if st.button("Predict House Price"):
        df_input = pd.DataFrame([input_data])

        pred = model.predict(df_input)[0]
        if use_log_target:
            pred = np.expm1(pred)

        price = pred * 100000

        st.success(f"### Predicted House Price: ${price:,.2f}")
        st.info(f"💰 Approx: ${price:,.0f}")

        avg_price = housing_data["target"].mean() * 100000
        diff = price - avg_price

        col1, col2 = st.columns(2)
        col1.metric("Predicted Price", f"${price:,.0f}")
        col2.metric("Average CA Price", f"${avg_price:,.0f}", f"{diff:+,.0f}")


if __name__ == "__main__":
    main()
