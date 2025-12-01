import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
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
The model is trained on the California Housing Dataset from scikit-learn.
""")

@st.cache_data
def load_data():
    """Load and preprocess the California housing dataset from OpenML"""
    from sklearn.datasets import fetch_openml
    df = fetch_openml(name="CaliforniaHousing", as_frame=True)

    housing_data = df.frame.copy()
    housing_data.rename(columns={"MedHouseVal": "target"}, inplace=True)

    # Log transform target
    housing_data["target_log"] = np.log1p(housing_data["target"])

    feature_names = [col for col in housing_data.columns if col not in ["target", "target_log"]]

    return housing_data, feature_names

@st.cache_resource
def train_model(X_train, y_train):
    """Train a Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # Load data
    housing_data, feature_names = load_data()
    
    # Sidebar for navigation
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
    
    # Show dataset info
    if st.checkbox("Show Dataset Info"):
        st.subheader("Dataset Information")
        st.write(f"Shape: {housing_data.shape}")
        st.write("First 5 rows:")
        st.dataframe(housing_data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Types:")
            st.write(housing_data.dtypes)
        with col2:
            st.write("Missing Values:")
            st.write(housing_data.isna().sum())
    
    # Show statistics
    if st.checkbox("Show Statistics"):
        st.subheader("Descriptive Statistics")
        st.dataframe(housing_data.describe())
    
    # Show distribution of target variable
    if st.checkbox("Show Target Variable Distribution"):
        st.subheader("Target Variable Distribution")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original target
        ax1.hist(housing_data["target"], bins=50, alpha=0.7, color='skyblue')
        ax1.set_xlabel("Median House Value (in $100,000)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Original Target Distribution")
        
        # Log-transformed target
        ax2.hist(housing_data["target_log"], bins=50, alpha=0.7, color='lightcoral')
        ax2.set_xlabel("Log(Median House Value)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Log-Transformed Target Distribution")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write(f"Original target skewness: {housing_data['target'].skew():.3f}")
        st.write(f"Log-transformed target skewness: {housing_data['target_log'].skew():.3f}")

def show_model_training(housing_data, feature_names):
    st.header("🤖 Model Training")
    
    # Feature selection
    st.subheader("Feature Selection")
    selected_features = st.multiselect(
        "Choose features for the model:",
        feature_names,
        default=feature_names
    )
    
    if not selected_features:
        st.warning("Please select at least one feature!")
        return
    
    # Train-test split parameters
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        use_log_target = st.checkbox("Use Log-Transformed Target", value=True)
    
    # Prepare data
    X = housing_data[selected_features]
    y = housing_data["target_log"] if use_log_target else housing_data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model = train_model(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Convert back from log if needed
            if use_log_target:
                y_test_original = np.expm1(y_test)
                y_pred_original = np.expm1(y_pred)
            else:
                y_test_original = y_test
                y_pred_original = y_pred
            
            # Calculate metrics
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_original, y_pred_original)
            
            # Display results
            st.success("Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("MSE", f"{mse:.4f}")
            col3.metric("R² Score", f"{r2:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)
            
            # Prediction vs Actual plot
            st.subheader("Prediction vs Actual")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test_original, y_pred_original, alpha=0.5)
            ax.plot([y_test_original.min(), y_test_original.max()], 
                   [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Prices")
            ax.set_ylabel("Predicted Prices")
            ax.set_title("Actual vs Predicted Prices")
            st.pyplot(fig)
            
            # Store the model and feature names in session state for prediction
            st.session_state['model'] = model
            st.session_state['selected_features'] = selected_features
            st.session_state['use_log_target'] = use_log_target

def show_price_prediction(housing_data, feature_names):
    st.header("🔮 Price Prediction")
    
    # Check if model is trained
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' section!")
        return
    
    model = st.session_state['model']
    selected_features = st.session_state['selected_features']
    use_log_target = st.session_state['use_log_target']
    
    st.subheader("Enter House Features")
    
    # Create input fields based on selected features
    input_data = {}
    
    # Get min and max values for sliders
    feature_stats = housing_data[selected_features].describe()
    
    # Arrange inputs in columns
    cols = st.columns(2)
    for i, feature in enumerate(selected_features):
        with cols[i % 2]:
            min_val = float(feature_stats.loc['min', feature])
            max_val = float(feature_stats.loc['max', feature])
            mean_val = float(feature_stats.loc['mean', feature])
            
            input_data[feature] = st.slider(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100
            )
    
    # Predict button
    if st.button("Predict House Price"):
        # Prepare input for prediction
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Convert back from log if needed
        if use_log_target:
            prediction = np.expm1(prediction)
        
        # Display result
        st.success(f"### Predicted House Price: ${prediction * 100000:,.2f}")
        st.info(f"💰 This is approximately ${prediction * 100000:,.0f}")
        
        # Show comparison with dataset statistics
        st.subheader("Price Comparison")
        avg_price = housing_data["target"].mean() * 100000
        price_diff = (prediction * 100000) - avg_price
        
        col1, col2 = st.columns(2)
        col1.metric("Predicted Price", f"${prediction * 100000:,.0f}")
        col2.metric("Average CA Price", f"${avg_price:,.0f}", 
                   f"{price_diff:+,.0f}")

if __name__ == "__main__":
    main()