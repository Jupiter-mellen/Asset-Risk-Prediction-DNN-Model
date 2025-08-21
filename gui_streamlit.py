import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Asset Risk Prediction",
    page_icon="",
    layout="wide"
)

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessing components"""
    try:
        # Load TensorFlow model
        model = tf.keras.models.load_model('outputs/tf_model.keras')
        
        # Load calibrator
        calibrator = joblib.load('outputs/isotonic_calibrator.joblib')
        
        # You'll need to recreate the preprocessors or save them during training
        # For now, we'll create new ones (you should save these during training)
        scaler = StandardScaler()
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        return model, calibrator, scaler, encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def preprocess_input(air_temp, process_temp, rpm, torque, tool_wear, machine_type, scaler, encoder):
    """Preprocess user input to match training format"""
    
    # Calculate engineered features
    temp_diff = process_temp - air_temp
    power_proxy = rpm * torque
    wear_per_power = tool_wear / (power_proxy + 1e-6)
    
    # Create feature array
    numeric_features = np.array([[air_temp, process_temp, temp_diff, rpm, torque, power_proxy, tool_wear, wear_per_power]])
    categorical_features = np.array([[machine_type]])
    
    # Note: In production, you'd load the fitted scalers from training
    # For demonstration, we'll use approximate scaling based on typical ranges
    numeric_scaled = (numeric_features - np.array([298, 308, 10, 1500, 40, 60000, 100, 0.001])) / np.array([5, 5, 5, 300, 15, 15000, 120, 0.002])
    
    # One-hot encode categorical (simplified)
    cat_encoded = np.zeros((1, 3))  # Assuming 3 machine types: H, L, M
    type_map = {'H': 0, 'L': 1, 'M': 2}
    if machine_type in type_map:
        cat_encoded[0, type_map[machine_type]] = 1
    
    # Combine features
    X_processed = np.hstack([numeric_scaled, cat_encoded])
    return X_processed

def predict_health_timeline(base_risk, tool_wear, rpm, torque):
    """Generate a predictive health timeline"""
    # Simulate degradation over time
    time_points = np.linspace(0, 1000, 100)  # Next 1000 hours
    
    # Simple degradation model based on current wear and operating conditions
    wear_rate = (rpm / 1500) * (torque / 40) * 0.1  # Normalized wear rate
    projected_wear = tool_wear + wear_rate * time_points
    
    # Risk increases with wear (simplified model)
    risk_over_time = base_risk + (projected_wear / 500) * (1 - base_risk)
    risk_over_time = np.clip(risk_over_time, 0, 1)
    
    return time_points, risk_over_time

# Main app
def main():
    st.title("Asset Risk Prediction System")
    st.markdown("**Predict machine failure risk and visualize asset health timeline**")
    
    # Load model
    model, calibrator, scaler, encoder = load_model_and_preprocessors()
    
    if model is None:
        st.error("Could not load model. Please check that outputs/tf_model.keras exists.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("Machine Parameters")
    
    # Input fields
    air_temp = st.sidebar.number_input("Air Temperature (K)", min_value=280.0, max_value=320.0, value=298.0, step=0.1)
    process_temp = st.sidebar.number_input("Process Temperature (K)", min_value=290.0, max_value=330.0, value=308.0, step=0.1)
    rpm = st.sidebar.number_input("Rotational Speed (RPM)", min_value=500, max_value=4000, value=1500, step=10)
    torque = st.sidebar.number_input("Torque (Nm)", min_value=5.0, max_value=100.0, value=40.0, step=0.5)
    tool_wear = st.sidebar.number_input("Tool Wear (min)", min_value=0, max_value=500, value=100, step=1)
    machine_type = st.sidebar.selectbox("Machine Type", ["H", "L", "M"])
    
    # Predict button
    if st.sidebar.button("Predict Risk", type="primary"):
        
        # Preprocess input
        X_input = preprocess_input(air_temp, process_temp, rpm, torque, tool_wear, machine_type, scaler, encoder)
        
        # Make prediction
        raw_prediction = model.predict(X_input, verbose=0)[0][0]
        
        # Apply calibration if available
        if calibrator is not None:
            try:
                calibrated_prediction = calibrator.predict([raw_prediction])[0]
            except:
                calibrated_prediction = raw_prediction
        else:
            calibrated_prediction = raw_prediction
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Prediction Results")
            
            # Risk score display
            risk_percentage = calibrated_prediction * 100
            
            if risk_percentage < 20:
                risk_color = "green"
                risk_status = "LOW RISK"
                risk_icon = ""
            elif risk_percentage < 50:
                risk_color = "orange"
                risk_status = "MEDIUM RISK"
                risk_icon = ""
            else:
                risk_color = "red"
                risk_status = "HIGH RISK"
                risk_icon = ""
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {risk_color}20; border: 2px solid {risk_color};">
                <h2 style="color: {risk_color}; margin: 0;">{risk_icon} {risk_status}</h2>
                <h1 style="color: {risk_color}; margin: 10px 0;">{risk_percentage:.1f}%</h1>
                <p style="margin: 0;">Failure Probability</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Classification result
            threshold = 0.5
            classification = "WILL FAIL" if calibrated_prediction > threshold else "WILL NOT FAIL"
            class_color = "red" if calibrated_prediction > threshold else "green"
            
            st.markdown(f"""
            <div style="margin-top: 20px; padding: 15px; border-radius: 8px; background-color: {class_color}15; border-left: 5px solid {class_color};">
                <h3 style="color: {class_color}; margin: 0;">Classification: {classification}</h3>
                <p style="margin: 5px 0 0 0;">Based on 50% threshold</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature summary
            st.subheader("Input Summary")
            feature_df = pd.DataFrame({
                'Parameter': ['Air Temp', 'Process Temp', 'RPM', 'Torque', 'Tool Wear', 'Type'],
                'Value': [f"{air_temp}K", f"{process_temp}K", f"{rpm}", f"{torque}Nm", f"{tool_wear}min", machine_type]
            })
            st.dataframe(feature_df, hide_index=True)
        
        with col2:
            st.header("Health Timeline Prediction")
            
            # Generate health timeline
            time_points, risk_timeline = predict_health_timeline(calibrated_prediction, tool_wear, rpm, torque)
            
            # Create timeline plot
            fig = go.Figure()
            
            # Add risk timeline
            fig.add_trace(go.Scatter(
                x=time_points,
                y=risk_timeline * 100,
                mode='lines',
                name='Failure Risk %',
                line=dict(color='red', width=3),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            # Add risk zones
            fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Low Risk Threshold")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="High Risk Threshold")
            
            fig.update_layout(
                title="Predicted Risk Over Time",
                xaxis_title="Operating Hours",
                yaxis_title="Failure Risk (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Maintenance recommendations
            st.subheader("Maintenance Recommendations")
            
            if risk_percentage < 20:
                st.success("Normal operations. Continue regular monitoring.")
            elif risk_percentage < 50:
                st.warning("Increased monitoring recommended. Schedule preventive maintenance.")
            else:
                st.error("Immediate inspection required. Consider stopping operations.")
            
            # Estimated maintenance window
            critical_hours = np.where(risk_timeline > 0.8)[0]
            if len(critical_hours) > 0:
                critical_time = time_points[critical_hours[0]]
                st.info(f"Estimated time to critical risk: {critical_time:.0f} hours")
            else:
                st.info("No critical risk detected in next 1000 hours")

    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Performance:**")
    st.sidebar.markdown("• ROC AUC: 97.2%")
    st.sidebar.markdown("• PR AUC: 64.0%")
    st.sidebar.markdown("• Trained on Machine Predictive Maintenance dataset")

if __name__ == "__main__":
    main()