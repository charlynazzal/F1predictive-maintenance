"""
Simple web dashboard for the Predictive Maintenance System using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import time

# Import the PredictiveMaintenanceSystem class
from predictive_maintenance import PredictiveMaintenanceSystem

# Set page configuration
st.set_page_config(
    page_title="F1 Predictive Maintenance Dashboard",
    page_icon="🏎️",
    layout="wide"
)

# --- Session State Initialization ---
# Initialize pm_system only once
if 'pm_system' not in st.session_state:
    st.session_state.pm_system = PredictiveMaintenanceSystem()

# Initialize other state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None # Holds the raw uploaded/generated data
if 'live_data_on' not in st.session_state:
    st.session_state.live_data_on = False
if 'trained_feature_cols' not in st.session_state:
    st.session_state.trained_feature_cols = None # Store feature columns used for training
if 'trained_sequence_length' not in st.session_state:
    st.session_state.trained_sequence_length = None # Store sequence length used for RUL
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False # Track if models were loaded
if 'scaler_fitted' not in st.session_state:
    # Check both loaded status and internal flag
    st.session_state.scaler_fitted = st.session_state.pm_system.fitted_scaler

# --- Helper Functions ---
def generate_synthetic_data(n_samples=500, freq='1T'):
    """Generate improved synthetic bearing sensor data"""
    timestamps = pd.date_range(start=datetime.now() - timedelta(minutes=n_samples-1), periods=n_samples, freq=freq)
    data = pd.DataFrame({'timestamp': timestamps})

    # Base sensor readings
    data['bearing1_horz'] = np.random.normal(0, 0.1, n_samples)
    data['bearing1_vert'] = np.random.normal(0, 0.1, n_samples)
    data['bearing2_horz'] = np.random.normal(0, 0.1, n_samples)
    data['bearing2_vert'] = np.random.normal(0, 0.1, n_samples)
    data['temperature'] = np.random.normal(75, 5, n_samples) + np.linspace(0, 15, n_samples) # Temp increases

    # Add degradation trend (non-linear increase in vibration)
    degradation = (np.linspace(0, 1.0, n_samples) ** 2.5) * 2.0 # Steeper increase towards end
    data['bearing1_horz'] += degradation * 0.4
    data['bearing1_vert'] += degradation * 0.3

    # Add anomalies (random spikes, more likely towards end)
    anomaly_prob = np.linspace(0.01, 0.1, n_samples) # Probability increases over time
    is_anomaly = np.random.rand(n_samples) < anomaly_prob
    data.loc[is_anomaly, 'bearing1_horz'] += np.random.uniform(1.0, 3.0, size=is_anomaly.sum())

    # Create failure labels (fail in last 10%)
    failure_point = int(0.9 * n_samples)
    data['failure'] = 0
    data.loc[failure_point:, 'failure'] = 1

    # Create RUL (Remaining Useful Life in days)
    samples_per_day = pd.Timedelta('1 day') / pd.Timedelta(freq)
    data['rul'] = (n_samples - 1 - np.arange(n_samples)) / samples_per_day
    # Make RUL drop faster near failure
    data.loc[failure_point:, 'rul'] *= np.linspace(1, 0, n_samples - failure_point)**1.5

    return data.set_index('timestamp')

# --- Sidebar ---
st.sidebar.header("Controls")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload sensor data (CSV)", type=["csv"])
if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        # Basic validation and timestamp parsing
        if 'timestamp' not in raw_data.columns:
            st.sidebar.warning("CSV missing 'timestamp' column. Generating default timestamps.")
            # Try to infer frequency or default to 1 minute
            freq = pd.infer_freq(raw_data.index) if isinstance(raw_data.index, pd.DatetimeIndex) else '1T'
            raw_data['timestamp'] = pd.date_range(
                end=datetime.now(), periods=len(raw_data), freq=freq
            )
        else:
            try:
                raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
            except Exception:
                 st.sidebar.error("Could not parse 'timestamp' column. Please ensure it's in a recognizable format.")
                 raw_data = None # Prevent using invalid data

        if raw_data is not None:
             st.session_state.data = raw_data.set_index('timestamp').sort_index()
             st.sidebar.success(f"Loaded {len(st.session_state.data)} records.")
             # Reset trained state if new data is loaded
             st.session_state.trained_feature_cols = None
             st.session_state.trained_sequence_length = None
             st.session_state.scaler_fitted = False
             st.session_state.models_loaded = False
             # Re-initialize system to clear old models/scaler if new data loaded
             st.session_state.pm_system = PredictiveMaintenanceSystem()

    except Exception as e:
        st.sidebar.error(f"Error processing CSV: {e}")
        st.session_state.data = None

elif st.session_state.data is None:
    if st.sidebar.button("Load Demo Data"):
        with st.spinner("Generating demo data..."):
            st.session_state.data = generate_synthetic_data(n_samples=1000, freq='1T')
        st.sidebar.success("Demo data loaded!")
        # Reset state for demo data
        st.session_state.trained_feature_cols = None
        st.session_state.trained_sequence_length = None
        st.session_state.scaler_fitted = False
        st.session_state.models_loaded = False
        st.session_state.pm_system = PredictiveMaintenanceSystem()


# Model loading/saving section
st.sidebar.subheader("Model Management")
models_dir = "./trained_models" # Changed directory name
os.makedirs(models_dir, exist_ok=True)

if st.sidebar.button("Load Models"):
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
         st.sidebar.warning(f"Model directory '{models_dir}' is empty or doesn't exist.")
    else:
        try:
            with st.spinner("Loading models and scaler..."):
                # Instantiate new system before loading to ensure clean state
                st.session_state.pm_system = PredictiveMaintenanceSystem()
                st.session_state.pm_system.load_models(models_dir)
                # Check if scaler was loaded successfully
                st.session_state.scaler_fitted = st.session_state.pm_system.fitted_scaler
                st.session_state.models_loaded = True
                # Attempt to load feature columns and sequence length if saved previously
                # (This requires modification to save_models/load_models or separate file handling)
                # For now, assume they need to be re-selected or inferred
                st.session_state.trained_feature_cols = None # Reset, needs manual selection or loading mechanism
                st.session_state.trained_sequence_length = None # Reset
            st.sidebar.success("Models and scaler loaded!")
            if not st.session_state.scaler_fitted:
                 st.sidebar.warning("Scaler file not found or failed to load.")
        except Exception as e:
            st.sidebar.error(f"Error loading models: {e}")
            st.session_state.scaler_fitted = False
            st.session_state.models_loaded = False

# Save button enabled only if scaler is fitted
col_save, col_status = st.sidebar.columns([1, 3])
can_save = st.session_state.scaler_fitted or st.session_state.models_loaded # Allow saving if models loaded or scaler fitted
save_disabled = not can_save
if col_save.button("Save Models", disabled=save_disabled):
    try:
        with st.spinner("Saving models and scaler..."):
            # Pass feature cols and seq length to save_models (requires modifying pm_system.save_models)
            # Alternatively, save them separately here
            st.session_state.pm_system.save_models(models_dir)
            # Save feature cols and seq length to a separate config file (e.g., JSON)
            config_path = os.path.join(models_dir, "training_config.json")
            config_data = {
                "feature_cols": st.session_state.trained_feature_cols,
                "sequence_length": st.session_state.trained_sequence_length
            }
            import json
            with open(config_path, 'w') as f:
                 json.dump(config_data, f)

        st.sidebar.success(f"Models saved to {models_dir}")
    except Exception as e:
        st.sidebar.error(f"Error saving models: {e}")

if save_disabled:
     col_status.caption("Train or load models to enable saving.")

# Status Indicator
st.sidebar.subheader("System Status")
if st.session_state.scaler_fitted:
    st.sidebar.success("Scaler is Fitted")
else:
    st.sidebar.warning("Scaler not Fitted")

if st.session_state.pm_system.anomaly_detector:
    st.sidebar.success("Anomaly Detector Ready")
else:
    st.sidebar.warning("Anomaly Detector Not Ready")

if st.session_state.pm_system.failure_classifier:
    st.sidebar.success("Failure Classifier Ready")
else:
    st.sidebar.warning("Failure Classifier Not Ready")

if st.session_state.pm_system.rul_predictor:
    st.sidebar.success("RUL Predictor Ready")
else:
    st.sidebar.warning("RUL Predictor Not Ready")


# Live data simulation toggle (Removed for simplicity in refactoring)
# st.sidebar.subheader("Simulation")
# live_data = st.sidebar.checkbox("Simulate live data", value=st.session_state.live_data_on)
# st.session_state.live_data_on = live_data

# --- Main Layout: Tabs ---
st.title("🏎️ Formula One Predictive Maintenance Dashboard")
st.markdown("Monitor component health, train models, and analyze sensor data.")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Training", "Analysis"])

# --- Dashboard Tab --- #
with tab1:
    st.header("Live Health Monitoring")

    if st.session_state.data is None:
        st.info("Load data or demo data using the sidebar to view the dashboard.")
    else:
        # Check if models are ready for prediction
        scaler_ready = st.session_state.scaler_fitted
        features_defined = st.session_state.trained_feature_cols is not None
        seq_len_defined = st.session_state.trained_sequence_length is not None

        # Prediction requires scaler and defined features
        can_predict = scaler_ready and features_defined

        if not can_predict:
            st.warning("Models need to be trained or loaded, and scaler must be fitted before predictions can be shown. Please visit the Training tab or load models.")
        else:
            # Get latest data point for metrics
            latest_data_point = st.session_state.data.iloc[-1:]

            # Prepare features for prediction
            try:
                features_raw = latest_data_point[st.session_state.trained_feature_cols]
                features_scaled = st.session_state.pm_system.preprocess_data(features_raw, st.session_state.trained_feature_cols)

                # Get predictions
                anomaly_score = None
                failure_prob = None
                rul = None

                if st.session_state.pm_system.anomaly_detector:
                    anomaly_score = st.session_state.pm_system.anomaly_detector.decision_function(features_scaled)[0]

                if st.session_state.pm_system.failure_classifier:
                    failure_prob = st.session_state.pm_system.predict_failure_probability(features_scaled)[0]

                if st.session_state.pm_system.rul_predictor and seq_len_defined:
                    # Get the last sequence of data needed
                    seq_len = st.session_state.trained_sequence_length
                    if len(st.session_state.data) >= seq_len:
                        recent_data_raw = st.session_state.data.iloc[-seq_len:]
                        recent_features_raw = recent_data_raw[st.session_state.trained_feature_cols]
                        recent_features_scaled = st.session_state.pm_system.preprocess_data(recent_features_raw, st.session_state.trained_feature_cols)
                        # Reshape for LSTM: [1, time_steps, features]
                        X_seq = recent_features_scaled.values.reshape(1, seq_len, len(st.session_state.trained_feature_cols))
                        rul = st.session_state.pm_system.predict_rul(X_seq)[0]
                    else:
                         st.caption(f"Not enough data ({len(st.session_state.data)} points) for RUL sequence length ({seq_len}).")

                # Display Metrics
                st.subheader("Latest Component Status")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    temp_col = 'temperature' # Assume 'temperature' exists for demo metric
                    if temp_col in latest_data_point.columns:
                         st.metric(
                             label="Temperature",
                             value=f"{latest_data_point[temp_col].iloc[-1]:.1f}°C"
                         )
                    else:
                         st.metric(label="Temperature", value="N/A")
                with col2:
                     st.metric(label="Anomaly Score", value=f"{anomaly_score:.3f}" if anomaly_score is not None else "N/A")
                with col3:
                     st.metric(label="Failure Probability", value=f"{(failure_prob * 100):.1f}%" if failure_prob is not None else "N/A")
                with col4:
                     st.metric(label="Predicted RUL", value=f"{(rul * 24):.1f} hours" if rul is not None else "N/A") # Display in hours

                # Health Status Visualization (using last N points)
                st.subheader("Recent Health Trend")
                num_points_plot = min(150, len(st.session_state.data)) # Plot last 150 points or fewer
                chart_data_raw = st.session_state.data.iloc[-num_points_plot:]

                try:
                    chart_features_raw = chart_data_raw[st.session_state.trained_feature_cols]
                    chart_features_scaled = st.session_state.pm_system.preprocess_data(chart_features_raw, st.session_state.trained_feature_cols)

                    # Get predictions for the chart period
                    chart_anomaly_flags = None
                    chart_failure_probs = None
                    chart_rul_values = None

                    if st.session_state.pm_system.anomaly_detector:
                         chart_anom_results = st.session_state.pm_system.detect_anomalies(chart_features_scaled)
                         chart_anomaly_flags = chart_anom_results['is_anomaly'].values

                    if st.session_state.pm_system.failure_classifier:
                         chart_failure_probs = st.session_state.pm_system.predict_failure_probability(chart_features_scaled)

                    if st.session_state.pm_system.rul_predictor and seq_len_defined:
                        if len(chart_features_scaled) >= seq_len:
                            # Use _create_sequences to get predictions aligned with the end of each sequence
                            # We need the target col temporarily for the function call signature
                            chart_data_temp_rul = chart_features_scaled.copy()
                            chart_data_temp_rul['_temp_target'] = 0 # Dummy target
                            X_chart_seq, _ = st.session_state.pm_system._create_sequences(chart_data_temp_rul, st.session_state.trained_feature_cols, '_temp_target', seq_len)
                            if X_chart_seq.shape[0] > 0:
                                rul_preds = st.session_state.pm_system.predict_rul(X_chart_seq)
                                # Pad NaNs at the beginning to align with original chart data
                                chart_rul_values = np.full(len(chart_data_raw), np.nan)
                                chart_rul_values[seq_len:] = rul_preds
                        else:
                            chart_rul_values = np.full(len(chart_data_raw), np.nan)

                    # Use the pm_system's visualize method
                    # Need to pass the *raw* data for plotting original values
                    st.session_state.pm_system.visualize_health_status(
                         component_data=chart_data_raw, # Pass raw data for plotting
                         feature_cols=st.session_state.trained_feature_cols,
                         failure_probs=chart_failure_probs,
                         rul_values=chart_rul_values,
                         anomaly_flags=chart_anomaly_flags,
                         timestamp_col=None # Use index
                    )
                    # Display the plot in Streamlit
                    st.pyplot(plt.gcf()) # Get the current figure generated by visualize_health_status
                    plt.clf() # Clear the figure after displaying

                except Exception as e:
                    st.error(f"Error generating health trend plot: {e}")
                    st.exception(e) # Show full traceback for debugging

            except KeyError as e:
                 st.error(f"Missing expected feature column: {e}. Ensure the loaded/trained feature columns match the current data.")
            except ValueError as e:
                 st.error(f"Data processing error: {e}. This might happen if the scaler wasn't fitted correctly or data format is wrong.")
            except Exception as e:
                 st.error(f"An unexpected error occurred on the dashboard: {e}")
                 st.exception(e)

# --- Training Tab --- #
with tab2:
    st.header("Model Training")

    if st.session_state.data is None:
        st.warning("Load data or demo data using the sidebar before training.")
    else:
        st.info(f"Available data: {len(st.session_state.data)} records from {st.session_state.data.index.min()} to {st.session_state.data.index.max()}")

        # --- Training Configuration ---
        st.subheader("Configure Training")

        # Select Features
        available_cols = st.session_state.data.select_dtypes(include=np.number).columns.tolist()
        # Exclude known target columns if they exist
        potential_features = [col for col in available_cols if col not in ['rul', 'failure']]

        # Default features: use stored if available, else use potential features
        default_features = st.session_state.trained_feature_cols if st.session_state.trained_feature_cols else potential_features

        feature_cols = st.multiselect(
            "Select features for training",
            options=available_cols,
            default=default_features,
            key="feature_select"
        )

        # Training Parameters
        col1, col2 = st.columns(2)
        with col1:
            train_size_perc = st.slider("Training data split (%) T", 50, 95, 75, key="train_split")
            anomaly_contamination = st.slider("Expected Anomaly Contamination (%) T", 0.1, 20.0, 3.0, step=0.1, format="%.1f%%", key="anom_cont")
            rf_estimators = st.number_input("Random Forest Estimators T", min_value=50, max_value=500, value=150, step=10, key="rf_n")
            rf_max_depth = st.number_input("Random Forest Max Depth T", min_value=5, max_value=50, value=12, step=1, key="rf_depth")

        with col2:
            # Only show RUL options if 'rul' column exists
            has_rul_col = 'rul' in st.session_state.data.columns
            sequence_length = st.slider("LSTM Sequence Length T", 10, 100, st.session_state.trained_sequence_length or 30, key="seq_len", disabled=not has_rul_col)
            epochs = st.slider("LSTM Training Epochs T", 10, 200, 25, key="epochs", disabled=not has_rul_col)
            lstm_units_1 = st.number_input("LSTM Layer 1 Units T", min_value=16, max_value=256, value=70, step=8, key="lstm1", disabled=not has_rul_col)
            lstm_units_2 = st.number_input("LSTM Layer 2 Units T", min_value=16, max_value=128, value=35, step=8, key="lstm2", disabled=not has_rul_col)

        # Check if required columns exist for specific models
        has_failure_col = 'failure' in st.session_state.data.columns

        st.markdown("--- T") # Visual separator

        # Train Button
        if not feature_cols:
             st.warning("Please select at least one feature column.")
        elif st.button("Start Training", key="train_button", disabled=not feature_cols):
            if not feature_cols:
                 st.error("No feature columns selected!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    # 1. Prepare Data
                    status_text.text("Splitting data...")
                    progress_bar.progress(5)
                    train_idx = int(len(st.session_state.data) * train_size_perc / 100)
                    train_data_raw = st.session_state.data.iloc[:train_idx].copy()
                    test_data_raw = st.session_state.data.iloc[train_idx:].copy()
                    st.write(f"Training on {len(train_data_raw)} samples, Testing on {len(test_data_raw)} samples.")
                    st.write(f"Using features: {feature_cols}")

                    # Instantiate a new system for training to ensure clean state
                    st.session_state.pm_system = PredictiveMaintenanceSystem()

                    # 2. Fit Scaler
                    status_text.text("Fitting data scaler...")
                    progress_bar.progress(10)
                    st.session_state.pm_system.fit_scaler(train_data_raw, feature_cols)
                    st.session_state.scaler_fitted = True

                    # 3. Preprocess Data (Scale)
                    status_text.text("Preprocessing (scaling) data...")
                    progress_bar.progress(15)
                    train_data_scaled = st.session_state.pm_system.preprocess_data(train_data_raw, feature_cols)
                    test_data_scaled = st.session_state.pm_system.preprocess_data(test_data_raw, feature_cols)

                    # --- Train Models --- #

                    # 4. Train Anomaly Detector
                    status_text.text("Training Anomaly Detector (Isolation Forest)...")
                    progress_bar.progress(25)
                    # Use only normal data for training IF failure column exists, else use all training data
                    if has_failure_col:
                        normal_data_scaled = train_data_scaled[train_data_raw['failure'] == 0][feature_cols]
                        if len(normal_data_scaled) == 0:
                             st.warning("No 'normal' samples (failure=0) found in training split. Training anomaly detector on all training data.")
                             normal_data_scaled = train_data_scaled[feature_cols]
                    else:
                         normal_data_scaled = train_data_scaled[feature_cols]

                    if len(normal_data_scaled) > 0:
                         st.session_state.pm_system.train_anomaly_detector(
                             normal_data_scaled,
                             contamination=anomaly_contamination / 100.0,
                             n_estimators=rf_estimators # Reuse RF estimators for consistency?
                         )
                    else:
                         st.warning("Skipping anomaly detector training due to lack of suitable training data.")

                    # 5. Train Failure Classifier
                    if has_failure_col:
                        status_text.text("Training Failure Classifier (Random Forest)...")
                        progress_bar.progress(45)
                        X_train_class = train_data_scaled[feature_cols]
                        y_train_class = train_data_raw['failure']
                        # Check if both classes are present
                        if len(y_train_class.unique()) < 2:
                             st.warning(f"Only one class ({y_train_class.unique()[0]}) found in failure labels for training. Skipping classifier training.")
                        else:
                             st.session_state.pm_system.train_failure_classifier(
                                 X_train_class, y_train_class,
                                 n_estimators=rf_estimators,
                                 max_depth=rf_max_depth
                             )
                    else:
                         status_text.text("Skipping Failure Classifier training ('failure' column not found).")
                         progress_bar.progress(45)

                    # 6. Train RUL Predictor
                    if has_rul_col:
                        status_text.text("Preparing sequences for RUL (LSTM)...")
                        progress_bar.progress(65)
                        # Create sequences from SCALED data, but use RAW RUL as target
                        rul_train_data_for_seq = train_data_scaled.copy()
                        rul_train_data_for_seq['rul'] = train_data_raw['rul']
                        X_train_seq, y_train_rul = st.session_state.pm_system._create_sequences(
                            rul_train_data_for_seq, feature_cols, 'rul', sequence_length
                        )

                        if X_train_seq.shape[0] > 0:
                            status_text.text(f"Building RUL Predictor (LSTM) with {X_train_seq.shape[0]} sequences...")
                            progress_bar.progress(75)
                            st.session_state.pm_system.build_rul_predictor(
                                sequence_length, X_train_seq.shape[2],
                                lstm_units_1=lstm_units_1,
                                lstm_units_2=lstm_units_2
                            )

                            status_text.text("Training RUL Predictor (LSTM). This may take a while...")
                            # Capture training plot
                            with st.expander("Show RUL Training Progress Plot", expanded=False):
                                fig_rul, ax_rul = plt.subplots()
                                st.session_state.pm_system.train_rul_predictor(
                                    X_train_seq, y_train_rul,
                                    epochs=epochs,
                                    validation_split=0.15,
                                    plot_history=False # Disable default plot
                                )
                                # Manually plot history if needed
                                history = st.session_state.pm_system.rul_predictor.history.history
                                ax_rul.plot(history['loss'], label='Training Loss')
                                ax_rul.plot(history['val_loss'], label='Validation Loss')
                                ax_rul.set_title('RUL Predictor Training History')
                                ax_rul.set_xlabel('Epochs')
                                ax_rul.set_ylabel('Loss (MSE)')
                                ax_rul.legend()
                                ax_rul.grid(True)
                                st.pyplot(fig_rul)
                                plt.close(fig_rul) # Close figure
                        else:
                             st.warning("Not enough data to create sequences for RUL training.")
                    else:
                         status_text.text("Skipping RUL Predictor training ('rul' column not found).")
                         progress_bar.progress(75)

                    # 7. Evaluation (Optional - display results)
                    status_text.text("Evaluating models on test set...")
                    progress_bar.progress(90)
                    X_test_scaled = test_data_scaled[feature_cols]
                    y_test_failure = test_data_raw['failure'] if has_failure_col else None

                    X_test_seq = None
                    y_test_rul = None
                    if has_rul_col:
                         rul_test_data_for_seq = test_data_scaled.copy()
                         rul_test_data_for_seq['rul'] = test_data_raw['rul']
                         X_test_seq, y_test_rul = st.session_state.pm_system._create_sequences(
                             rul_test_data_for_seq, feature_cols, 'rul', sequence_length
                         )
                         if X_test_seq.shape[0] == 0:
                             X_test_seq, y_test_rul = None, None # Ensure they are None if empty

                    # Align evaluation data if sequences were created
                    eval_start_index = sequence_length if X_test_seq is not None else 0
                    if eval_start_index > len(X_test_scaled):
                        st.warning("Test set too small for evaluation after sequence creation.")
                    else:
                         eval_results = st.session_state.pm_system.evaluate_models(
                             X_test_scaled[eval_start_index:],
                             y_test_failure.iloc[eval_start_index:] if y_test_failure is not None else None,
                             X_test_seq,
                             y_test_rul
                         )
                         st.subheader("Model Evaluation Results (on Test Set)")
                         if 'failure_accuracy' in eval_results:
                              st.metric("Failure Classification Accuracy", f"{eval_results['failure_accuracy']:.3f}")
                              st.text("Classification Report:")
                              st.text(eval_results['failure_report'])
                              # Display confusion matrix plot
                              st.pyplot(plt.gcf()) # Get the figure generated by evaluate_models
                              plt.clf()

                         if 'rul_mae' in eval_results:
                              st.metric("RUL Prediction MAE (days)", f"{eval_results['rul_mae']:.3f}")
                              st.metric("RUL Prediction MSE (days^2)", f"{eval_results['rul_mse']:.3f}")
                              # Display RUL scatter plot
                              st.pyplot(plt.gcf()) # Get the figure generated by evaluate_models
                              plt.clf()

                    # 8. Update session state
                    status_text.text("Finalizing training...")
                    progress_bar.progress(100)
                    st.session_state.trained_feature_cols = feature_cols
                    st.session_state.trained_sequence_length = sequence_length if has_rul_col else None
                    st.session_state.models_loaded = False # Mark as trained, not loaded

                    status_text.empty()
                    progress_bar.empty()
                    st.success("Training and evaluation complete!")

                except FileNotFoundError as e: # Catch potential issues with model saving/loading paths
                     st.error(f"File path error: {e}")
                except ValueError as e:
                     st.error(f"Data or parameter error during training: {e}")
                     st.exception(e) # Show traceback for debugging
                except Exception as e:
                     st.error(f"An unexpected error occurred during training: {e}")
                     st.exception(e) # Show traceback for debugging
                finally:
                     # Clear status elements even if error occurs
                     status_text.empty()
                     progress_bar.empty()

# --- Analysis Tab --- #
with tab3:
    st.header("Data & Anomaly Analysis")

    if st.session_state.data is None:
        st.warning("Load data or demo data using the sidebar to perform analysis.")
    else:
        # Analysis requires feature columns to be defined (either from training/loading or selection)
        if not st.session_state.trained_feature_cols:
            st.info("Select features used for models if trained/loaded, or relevant columns for general analysis.")
            analysis_features = st.multiselect(
                "Select features for analysis",
                options=st.session_state.data.select_dtypes(include=np.number).columns.tolist(),
                default=st.session_state.data.select_dtypes(include=np.number).columns.tolist()[:5] # Default to first 5 numeric
            )
        else:
            analysis_features = st.session_state.trained_feature_cols
            st.info(f"Using features from last training/loading: {analysis_features}")

        if not analysis_features:
            st.warning("Please select features to analyze.")
        else:
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                "Data Overview", "Feature Analysis", "Anomaly Insights"
            ])

            # Data Overview Tab
            with analysis_tab1:
                st.subheader("Raw Data Sample")
                st.dataframe(st.session_state.data.head()) # Show raw data head

                st.subheader("Summary Statistics (Selected Features)")
                st.dataframe(st.session_state.data[analysis_features].describe())

                st.subheader("Selected Features Over Time")
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for col in analysis_features:
                        ax.plot(st.session_state.data.index, st.session_state.data[col], label=col)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig) # Close figure
                except Exception as e:
                    st.error(f"Error plotting features: {e}")

            # Feature Analysis Tab
            with analysis_tab2:
                st.subheader("Feature Correlation Heatmap")
                try:
                    corr_matrix = st.session_state.data[analysis_features].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax)
                    plt.title("Correlation Matrix of Selected Features")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                     st.error(f"Error calculating/plotting correlation: {e}")

                st.subheader("Feature Distribution Plots")
                try:
                    num_features = len(analysis_features)
                    cols = 3
                    rows = (num_features + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                    axes = axes.flatten() # Flatten to easily iterate
                    for i, col in enumerate(analysis_features):
                         sns.histplot(st.session_state.data[col], kde=True, ax=axes[i])
                         axes[i].set_title(f"Distribution of {col}")
                         axes[i].grid(True)
                    # Hide unused subplots
                    for j in range(i + 1, len(axes)):
                         fig.delaxes(axes[j])
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                     st.error(f"Error plotting distributions: {e}")

            # Anomaly Insights Tab
            with analysis_tab3:
                st.subheader("Anomaly Detection Analysis")

                # Requires scaler and anomaly detector
                if not st.session_state.scaler_fitted:
                    st.warning("Scaler is not fitted. Please train models first.")
                elif st.session_state.pm_system.anomaly_detector is None:
                    st.warning("Anomaly detector model is not trained or loaded.")
                elif not st.session_state.trained_feature_cols:
                     st.warning("Feature columns used for the trained anomaly detector are unknown. Please retrain or load model configuration.")
                else:
                    try:
                        # Use features defined during training for consistency
                        analysis_features_trained = st.session_state.trained_feature_cols
                        st.info(f"Running anomaly detection using features: {analysis_features_trained}")

                        # Scale the data using the fitted scaler
                        data_raw_analysis = st.session_state.data[analysis_features_trained]
                        data_scaled_analysis = st.session_state.pm_system.preprocess_data(data_raw_analysis, analysis_features_trained)

                        # Detect anomalies
                        anomaly_results = st.session_state.pm_system.detect_anomalies(data_scaled_analysis)
                        anomaly_count = anomaly_results['is_anomaly'].sum()
                        anomaly_perc = (anomaly_count / len(anomaly_results)) * 100

                        st.metric(label="Total Detected Anomalies", value=f"{anomaly_count} ({anomaly_perc:.1f}%) T")

                        # Anomaly Score Distribution
                        st.subheader("Anomaly Score Distribution")
                        fig_score, ax_score = plt.subplots(figsize=(10, 5))
                        sns.histplot(anomaly_results['anomaly_score'], kde=True, ax=ax_score, bins=50)
                        # Get threshold if available (may not be directly accessible depending on sklearn version/model)
                        threshold = getattr(st.session_state.pm_system.anomaly_detector, 'offset_', None)
                        if threshold is not None:
                             # Adjust threshold if decision_function is used (negative offset means score < threshold = anomaly)
                             decision_threshold = -threshold
                             ax_score.axvline(decision_threshold, color='r', linestyle='--', label=f'Decision Threshold ({decision_threshold:.3f})')
                        ax_score.set_title('Distribution of Anomaly Scores')
                        ax_score.set_xlabel('Score (Lower means more anomalous)')
                        ax_score.legend()
                        st.pyplot(fig_score)
                        plt.close(fig_score)

                        # Anomalies Over Time Plot
                        st.subheader("Anomalies Highlighted Over Time")
                        feature_to_plot = st.selectbox(
                            "Select feature to visualize anomalies on",
                            analysis_features_trained,
                            key="anomaly_feature_select"
                        )
                        if feature_to_plot:
                            plot_data_anom = st.session_state.data[[feature_to_plot]].copy()
                            # Ensure index alignment before assigning
                            anomaly_results.index = plot_data_anom.index[:len(anomaly_results)]
                            plot_data_anom['is_anomaly'] = anomaly_results['is_anomaly']

                            fig_anom, ax_anom = plt.subplots(figsize=(12, 6))
                            normal_pts = plot_data_anom[plot_data_anom['is_anomaly'] == 0]
                            anom_pts = plot_data_anom[plot_data_anom['is_anomaly'] == 1]

                            ax_anom.plot(normal_pts.index, normal_pts[feature_to_plot], label='Normal', color='cornflowerblue', linestyle='-', marker='.', markersize=3, alpha=0.7)
                            if not anom_pts.empty:
                                 ax_anom.scatter(anom_pts.index, anom_pts[feature_to_plot], label='Anomaly', color='red', s=50, marker='x')

                            ax_anom.set_title(f'{feature_to_plot} with Anomalies Highlighted')
                            ax_anom.set_xlabel('Time')
                            ax_anom.set_ylabel(feature_to_plot)
                            ax_anom.legend()
                            ax_anom.grid(True)
                            plt.xticks(rotation=30)
                            plt.tight_layout()
                            st.pyplot(fig_anom)
                            plt.close(fig_anom)

                        # Top Anomalous Data Points
                        st.subheader("Data Points with Lowest Anomaly Scores")
                        top_anomalies = anomaly_results.sort_values('anomaly_score').head(15)
                        # Merge with original data for context
                        top_anomalies_display = st.session_state.data.loc[top_anomalies.index].join(top_anomalies['anomaly_score'])
                        st.dataframe(top_anomalies_display)

                    except KeyError as e:
                         st.error(f"Missing feature column required for anomaly analysis: {e}. Ensure the selected features match the trained model.")
                    except ValueError as e:
                         st.error(f"Data processing error during anomaly analysis: {e}. Check scaler and data.")
                    except Exception as e:
                         st.error(f"An unexpected error occurred during anomaly analysis: {e}")
                         st.exception(e)