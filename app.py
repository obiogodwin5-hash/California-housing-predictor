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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stNumberInput > div > div > input {
        background-color: #ffffff;
    }
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #f44336;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the California housing dataset"""
    try:
        df = fetch_california_housing()
        housing_data = pd.DataFrame(data=df['data'], columns=df['feature_names'])
        housing_data["target"] = df["target"]
        
        # Apply log transformation to target variable
        housing_data["target_log"] = np.log1p(housing_data["target"])
        
        return housing_data, df['feature_names']
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def train_model(X_train, y_train, n_estimators=100):
    """Train a Random Forest model"""
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

def main():
    # Load data
    housing_data, feature_names = load_data()
    
    if housing_data is None:
        st.error("Failed to load data. Please try again.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("🏠 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode", 
        ["📊 Data Exploration", "🤖 Model Training", "🔮 Price Prediction"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app predicts California housing prices using machine learning. "
        "Data source: scikit-learn California Housing dataset."
    )
    
    if app_mode == "📊 Data Exploration":
        show_data_exploration(housing_data, feature_names)
    
    elif app_mode == "🤖 Model Training":
        show_model_training(housing_data, feature_names)
    
    elif app_mode == "🔮 Price Prediction":
        show_price_prediction(housing_data, feature_names)

def show_data_exploration(housing_data, feature_names):
    st.markdown('<div class="main-header">📊 California Housing Data Exploration</div>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(housing_data):,}")
    with col2:
        st.metric("Number of Features", len(feature_names))
    with col3:
        st.metric("Average Price", f"${housing_data['target'].mean() * 100000:,.0f}")
    
    # Show dataset
    if st.checkbox("Show Raw Data"):
        st.subheader("Dataset Preview")
        st.dataframe(housing_data.head(100), use_container_width=True)
    
    # Show statistics
    if st.checkbox("Show Detailed Statistics"):
        st.subheader("Descriptive Statistics")
        st.dataframe(housing_data.describe(), use_container_width=True)
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(housing_data["target"] * 100000, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel("House Price ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of House Prices")
        ax.ticklabel_format(style='plain', axis='x')
        st.pyplot(fig)
    
    with col2:
        # Correlation heatmap
        st.write("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = housing_data[feature_names + ['target']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

def show_model_training(housing_data, feature_names):
    st.markdown('<div class="main-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Selection")
        selected_features = st.multiselect(
            "Choose features for the model:",
            feature_names,
            default=feature_names
        )
        
        # CHANGED: Number of Trees from slider to number input
        n_estimators = st.number_input(
            "Number of Trees",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Enter the number of trees in the Random Forest (50 to 200)"
        )
    
    with col2:
        st.subheader("Training Parameters")
        
        # Input box for test size
        test_size = st.number_input(
            "Test Set Size (0.1 to 0.5)", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.2,
            step=0.05,
            help="Enter a value between 0.1 and 0.5 for the proportion of data to use as test set"
        )
        
        use_log_target = st.checkbox("Use Log-Transformed Target", value=True)
        random_state = st.number_input("Random State", value=42)
    
    if not selected_features:
        st.warning("Please select at least one feature!")
        return
    
    # Prepare data
    X = housing_data[selected_features]
    y = housing_data["target_log"] if use_log_target else housing_data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model... This may take a few seconds."):
            model = train_model(X_train, y_train, n_estimators)
            
            if model is None:
                st.error("Model training failed!")
                return
            
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
            mae = np.mean(np.abs(y_test_original - y_pred_original))
            
            # Display results
            st.success("✅ Model trained successfully!")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"${rmse * 100000:,.0f}")
            col2.metric("MSE", f"${mse * 1000000:,.0f}K")
            col3.metric("MAE", f"${mae * 100000:,.0f}")
            col4.metric("R² Score", f"{r2:.4f}")
            
            # Feature importance
            st.subheader("📈 Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color='lightcoral')
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance in Random Forest Model")
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center')
            
            st.pyplot(fig)
            
            # Store the model and feature names in session state for prediction
            st.session_state['model'] = model
            st.session_state['selected_features'] = selected_features
            st.session_state['use_log_target'] = use_log_target
            
            st.balloons()

def show_price_prediction(housing_data, feature_names):
    st.markdown('<div class="main-header">🔮 Price Prediction</div>', unsafe_allow_html=True)
    
    # Check if model is trained
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Model Training' section!")
        st.info("Go to the 'Model Training' tab, select features, and click 'Train Model'.")
        return
    
    model = st.session_state['model']
    selected_features = st.session_state['selected_features']
    use_log_target = st.session_state['use_log_target']
    
    st.subheader("Enter House Features")
    st.write("Fill in the values for each feature below:")
    
    # Get min and max values for validation
    feature_stats = housing_data[selected_features].describe()
    
    # Feature descriptions
    feature_descriptions = {
        'MedInc': 'Median income in block group (in tens of thousands of dollars)',
        'HouseAge': 'Median house age in block group (in years)',
        'AveRooms': 'Average number of rooms per household',
        'AveBedrms': 'Average number of bedrooms per household',
        'Population': 'Block group population',
        'AveOccup': 'Average number of household members',
        'Latitude': 'Block group latitude (degrees)',
        'Longitude': 'Block group longitude (degrees)'
    }
    
    # Create input boxes based on selected features
    input_data = {}
    validation_errors = []
    
    # Arrange inputs in columns
    cols = st.columns(2)
    for i, feature in enumerate(selected_features):
        with cols[i % 2]:
            min_val = float(feature_stats.loc['min', feature])
            max_val = float(feature_stats.loc['max', feature])
            mean_val = float(feature_stats.loc['mean', feature])
            
            # Format help text with statistics
            help_text = f"""
            {feature_descriptions.get(feature, feature)}
            
            Range: {min_val:.2f} to {max_val:.2f}
            Mean: {mean_val:.2f}
            """
            
            # Create number input box
            input_value = st.number_input(
                label=f"{feature}",
                value=float(mean_val),
                min_value=float(min_val),
                max_value=float(max_val),
                step=float((max_val - min_val) / 100),
                format="%.2f",
                help=help_text,
                key=f"input_{feature}"
            )
            
            # Store the value
            input_data[feature] = input_value
            
            # Validate the input
            if input_value < min_val or input_value > max_val:
                validation_errors.append(
                    f"{feature}: Value {input_value:.2f} is outside the valid range [{min_val:.2f}, {max_val:.2f}]"
                )
    
    # Add some spacing
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🎯 Predict House Price", type="primary", use_container_width=True)
    
    if predict_btn:
        # Check for validation errors
        if validation_errors:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("❌ **Validation Errors:**")
            for error in validation_errors:
                st.write(f"• {error}")
            st.write("Please adjust the values to be within the specified ranges.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Validate inputs again (in case user manually typed invalid values)
        all_valid = True
        for feature in selected_features:
            min_val = float(feature_stats.loc['min', feature])
            max_val = float(feature_stats.loc['max', feature])
            if input_data[feature] < min_val or input_data[feature] > max_val:
                all_valid = False
                break
        
        if not all_valid:
            st.error("❌ Some values are outside the valid range. Please adjust the inputs.")
            return
        
        # Prepare input for prediction
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Convert back from log if needed
        if use_log_target:
            prediction = np.expm1(prediction)
        
        predicted_price = prediction * 100000
        
        # Display result in a nice box
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.success(f"## Predicted House Price: ${predicted_price:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show comparison with dataset statistics
        st.subheader("📊 Price Analysis")
        avg_price = housing_data["target"].mean() * 100000
        price_diff = predicted_price - avg_price
        percent_diff = (price_diff / avg_price) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Price", f"${predicted_price:,.0f}")
        col2.metric("Average CA Price", f"${avg_price:,.0f}")
        col3.metric("Difference", f"${price_diff:+,.0f}", f"{percent_diff:+.1f}%")
        
        # Price interpretation
        if price_diff > 0:
            st.info(f"🏡 This house is **{percent_diff:.1f}% above** the California average")
            if percent_diff > 20:
                st.warning("⚠️ This is significantly above average. Consider if the features justify the premium.")
            elif percent_diff > 10:
                st.info("💰 This house is moderately above average price.")
        else:
            st.info(f"💰 This house is **{abs(percent_diff):.1f}% below** the California average")
            if abs(percent_diff) > 20:
                st.success("🎉 This could be a great value opportunity!")
            elif abs(percent_diff) > 10:
                st.info("📉 This house is moderately below average price.")
        
        # Show the input values for reference
        with st.expander("📋 Review Input Values"):
            st.write("Here are the values you entered:")
            review_data = []
            for feature in selected_features:
                min_val = float(feature_stats.loc['min', feature])
                max_val = float(feature_stats.loc['max', feature])
                status = "✅ Within range" if min_val <= input_data[feature] <= max_val else "❌ Out of range"
                
                review_data.append({
                    "Feature": feature,
                    "Value": input_data[feature],
                    "Min": min_val,
                    "Max": max_val,
                    "Status": status
                })
            st.table(pd.DataFrame(review_data))

if __name__ == "__main__":
    main()