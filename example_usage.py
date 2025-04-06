"""
Example of using the Predictive Maintenance System with real-world sensor data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import the PredictiveMaintenanceSystem class
from predictive_maintenance import PredictiveMaintenanceSystem

# Function to load bearing dataset from NASA
def load_bearing_dataset(data_dir):
    """
    Load the NASA IMS bearing dataset
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    
    This is a simplified loader - you may need to adapt it for the actual data structure
    """
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        sys.exit(1)
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} data files")
    
    # Initialize an empty dataframe to store all data
    all_data = pd.DataFrame()
    
    # Process each file
    for file in csv_files:
        # Construct full file path
        file_path = os.path.join(data_dir, file)
        
        # Extract timestamp from filename (assuming format like 2003.10.22.12.06.24)
        # This is specific to the NASA dataset format
        try:
            timestamp_str = os.path.splitext(file)[0]
            timestamp = pd.to_datetime(timestamp_str, format='%Y.%m.%d.%H.%M.%S')
        except ValueError:
            print(f"Warning: Could not parse timestamp from filename {file}. Using file modification time.")
            timestamp = pd.to_datetime(os.path.getmtime(file_path), unit='s')
        
        # Read data file
        # Assuming the file has columns for bearing readings but no headers
        try:
            df = pd.read_csv(file_path, header=None)
            # Rename columns based on NASA dataset structure (assuming 4 bearings)
            if df.shape[1] == 4:
                df.columns = ['bearing1_horz', 'bearing1_vert', 'bearing2_horz', 'bearing2_vert']
            elif df.shape[1] == 8: # Example for 2 sets of 4 sensors
                df.columns = ['b1_1', 'b1_2', 'b2_1', 'b2_2', 'b3_1', 'b3_2', 'b4_1', 'b4_2']
            else:
                print(f"Warning: Unexpected number of columns ({df.shape[1]}) in {file}. Skipping.")
                continue

            # Add timestamp
            df['timestamp'] = timestamp
            # Append to the full dataset
            all_data = pd.concat([all_data, df])
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    if all_data.empty:
        print("Error: No data loaded.")
        sys.exit(1)
    
    # Sort by timestamp
    all_data = all_data.sort_values('timestamp')
    all_data.reset_index(drop=True, inplace=True)
    
    # Add a sequential time index if needed
    # all_data['time_idx'] = range(len(all_data))
    
    # For this example, let's create a simple failure flag and RUL
    # In a real scenario, you would have actual labels or determine this based on domain knowledge
    # Assume the bearing fails in the last 10% of the data
    failure_start = int(0.9 * len(all_data))
    all_data['failure'] = 0
    all_data.loc[failure_start:, 'failure'] = 1
    
    # Create a RUL column (days until failure)
    # This is simplified - assumes constant sampling rate
    time_diff = all_data['timestamp'].diff().median() if len(all_data) > 1 else pd.Timedelta('1 second') # Estimate sampling interval
    if pd.isna(time_diff):
        time_diff = pd.Timedelta('1 second') # Default if median fails

    samples_to_failure = len(all_data) - all_data.index
    all_data['rul'] = samples_to_failure * time_diff / pd.Timedelta('1 day')
    all_data = all_data.set_index('timestamp') # Set timestamp index
    
    print(f"Loaded NASA bearing data: {len(all_data)} samples")
    return all_data


# Function to generate synthetic F1 component data
def generate_synthetic_f1_data(n_samples=1000):
    """
    Generate synthetic data that simulates F1 component telemetry
    """
    # Create timestamps
    start_date = pd.Timestamp('2023-01-01')
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='10min')
    
    # Base values for healthy state
    bearing1_horz_base = 0.05
    bearing1_vert_base = 0.04
    bearing2_horz_base = 0.06
    bearing2_vert_base = 0.05
    temperature_base = 85  # Higher temp for F1 components
    
    # Create arrays for sensor readings
    bearing1_horz = np.random.normal(bearing1_horz_base, 0.01, n_samples)
    bearing1_vert = np.random.normal(bearing1_vert_base, 0.01, n_samples)
    bearing2_horz = np.random.normal(bearing2_horz_base, 0.01, n_samples)
    bearing2_vert = np.random.normal(bearing2_vert_base, 0.01, n_samples)
    temperature = np.random.normal(temperature_base, 2, n_samples)
    
    # Add degradation trend
    degradation = (np.linspace(0, 1, n_samples) ** 2.5) * 0.3 # Non-linear degradation
    bearing1_horz += degradation * 0.5
    bearing1_vert += degradation * 0.4
    temperature += degradation * 15
    
    # Add some random spikes to simulate anomalies
    anomaly_indices = np.random.choice(range(n_samples), size=int(n_samples*0.05), replace=False)
    bearing1_horz[anomaly_indices] += np.random.uniform(0.1, 0.4, size=len(anomaly_indices))
    bearing1_vert[anomaly_indices] += np.random.uniform(0.1, 0.3, size=len(anomaly_indices))
    
    # Create the dataframe
    data = pd.DataFrame({
        'timestamp': timestamps,
        'bearing1_horz': bearing1_horz,
        'bearing1_vert': bearing1_vert,
        'bearing2_horz': bearing2_horz,
        'bearing2_vert': bearing2_vert,
        'temperature': temperature
    }).set_index('timestamp')
    
    # Add failure flag (1 for last 10% of data)
    failure_threshold = int(0.9 * n_samples)
    data['failure'] = 0
    data.iloc[failure_threshold:, data.columns.get_loc('failure')] = 1 # Use iloc for setting
    
    # Add RUL (days until end of data)
    samples_per_day = 24 * 6 # 6 samples per hour (10 min freq)
    data['rul'] = (n_samples - 1 - np.arange(n_samples)) / samples_per_day
    
    print(f"Generated synthetic F1 component data with {n_samples} samples")
    return data


# Main execution function (Refactored)
def main():
    print("=" * 80)
    print("Predictive Maintenance System for Formula One - Usage Example (Refactored)")
    print("=" * 80)
    
    # --- Configuration ---
    DATA_SOURCE = 'synthetic' # 'nasa', 'synthetic', 'csv'
    NASA_DATA_DIR = "./data/nasa_bearing" # Example path, change if needed
    CUSTOM_CSV_PATH = "./data/my_data.csv" # Example path
    SYNTHETIC_SAMPLES = 1500
    TRAIN_SPLIT_RATIO = 0.75
    MODEL_SAVE_DIR = "./trained_models_example"
    SEQUENCE_LENGTH = 30 # RUL sequence length

    # Define feature columns (ensure these exist in your chosen dataset)
    # Common features for example datasets
    feature_cols = ['bearing1_horz', 'bearing1_vert', 'bearing2_horz', 'bearing2_vert', 'temperature']
    failure_col = 'failure'
    rul_col = 'rul'

    # --- Load Data ---
    print(f"\nLoading data (Source: {DATA_SOURCE})...")
    data_raw = None
    if DATA_SOURCE == 'nasa':
        data_raw = load_bearing_dataset(NASA_DATA_DIR)
        # Ensure standard feature columns exist for NASA data if different
        # feature_cols = [col for col in data_raw.columns if col not in [failure_col, rul_col]] # Dynamically get features
    elif DATA_SOURCE == 'synthetic':
        data_raw = generate_synthetic_f1_data(SYNTHETIC_SAMPLES)
    elif DATA_SOURCE == 'csv':
        try:
            data_raw = pd.read_csv(CUSTOM_CSV_PATH)
            if 'timestamp' in data_raw.columns:
                data_raw['timestamp'] = pd.to_datetime(data_raw['timestamp'])
                data_raw = data_raw.set_index('timestamp').sort_index()
            else:
                print("Warning: CSV has no 'timestamp' column. Using default index.")

            # Check for required columns
            required = feature_cols + [failure_col, rul_col]
            missing = [col for col in required if col not in data_raw.columns]
            if missing:
                print(f"Error: Custom CSV missing required columns: {missing}")
                sys.exit(1)

        except FileNotFoundError:
            print(f"Error: Custom CSV file not found at {CUSTOM_CSV_PATH}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading custom CSV: {e}")
            sys.exit(1)
    else:
        print(f"Invalid DATA_SOURCE: {DATA_SOURCE}")
        sys.exit(1)

    if data_raw is None or data_raw.empty:
        print("Failed to load data.")
        sys.exit(1)

    # Verify feature columns exist in loaded data
    missing_features = [col for col in feature_cols if col not in data_raw.columns]
    if missing_features:
        print(f"Error: Defined feature columns {missing_features} not found in the loaded data.")
        print(f"Available columns: {data_raw.columns.tolist()}")
        sys.exit(1)

    # --- Data Splitting ---
    print("\nSplitting data into training and testing sets...")
    train_size = int(len(data_raw) * TRAIN_SPLIT_RATIO)
    train_data_raw = data_raw.iloc[:train_size].copy()
    test_data_raw = data_raw.iloc[train_size:].copy()
    print(f"Training samples: {len(train_data_raw)}, Testing samples: {len(test_data_raw)}")

    # --- Initialize System ---
    pm_system = PredictiveMaintenanceSystem()

    # --- Step 1: Fit Scaler ---
    print("\n[Step 1] Fitting scaler on training data...")
    pm_system.fit_scaler(train_data_raw, feature_cols)

    # --- Step 2: Preprocess (Scale) Data ---
    print("\n[Step 2] Preprocessing (scaling) data...")
    train_data_scaled = pm_system.preprocess_data(train_data_raw, feature_cols)
    test_data_scaled = pm_system.preprocess_data(test_data_raw, feature_cols)

    # --- Step 3: Train Anomaly Detector ---
    print("\n[Step 3] Training Anomaly Detector...")
    # Use normal data from the scaled training set
    normal_train_data_scaled = train_data_scaled[train_data_raw[failure_col] == 0]
    if normal_train_data_scaled.empty:
        print("Warning: No normal samples found in training data for anomaly detection. Training on all data.")
        normal_train_data_scaled = train_data_scaled

    pm_system.train_anomaly_detector(normal_train_data_scaled[feature_cols], contamination=0.03)

    # --- Step 4: Detect Anomalies ---
    print("\n[Step 4] Detecting anomalies in test data...")
    # Use scaled test features
    anomaly_results = pm_system.detect_anomalies(test_data_scaled[feature_cols])
    anomaly_count = anomaly_results['is_anomaly'].sum()
    print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(test_data_scaled)*100:.2f}% of test data)")

    # --- Step 5: Train Failure Classifier ---
    print("\n[Step 5] Training Failure Classifier...")
    X_train_class = train_data_scaled[feature_cols]
    y_train_class = train_data_raw[failure_col]
    if len(y_train_class.unique()) < 2:
        print("Warning: Only one class found for failure classification. Skipping training.")
    else:
        pm_system.train_failure_classifier(X_train_class, y_train_class)

    # --- Step 6: Predict Failure Probability ---
    print("\n[Step 6] Predicting failure probability...")
    X_test_class = test_data_scaled[feature_cols]
    failure_probs = pm_system.predict_failure_probability(X_test_class)

    # --- Step 7: Prepare Sequences for RUL ---
    print(f"\n[Step 7] Preparing sequences for RUL (length={SEQUENCE_LENGTH})...")
    # Add raw RUL target back to scaled data for sequence creation
    train_data_for_seq = train_data_scaled.copy()
    train_data_for_seq[rul_col] = train_data_raw[rul_col]
    X_train_seq, y_train_rul = pm_system._create_sequences(train_data_for_seq, feature_cols, rul_col, SEQUENCE_LENGTH)

    test_data_for_seq = test_data_scaled.copy()
    test_data_for_seq[rul_col] = test_data_raw[rul_col]
    X_test_seq, y_test_rul = pm_system._create_sequences(test_data_for_seq, feature_cols, rul_col, SEQUENCE_LENGTH)
    print(f"Created {X_train_seq.shape[0]} training sequences, {X_test_seq.shape[0]} test sequences.")

    # --- Step 8: Build and Train RUL Predictor ---
    if X_train_seq.shape[0] > 0:
        print("\n[Step 8] Building and training RUL predictor...")
        pm_system.build_rul_predictor(SEQUENCE_LENGTH, X_train_seq.shape[2]) # n_features from sequence shape
        pm_system.train_rul_predictor(X_train_seq, y_train_rul, epochs=30, plot_history=True) # Show plot
    else:
        print("Warning: No sequences created for RUL training. Skipping.")

    # --- Step 9: Predict RUL ---
    rul_predictions = None
    if pm_system.rul_predictor is not None and X_test_seq.shape[0] > 0:
        print("\n[Step 9] Predicting RUL for test data...")
        rul_predictions = pm_system.predict_rul(X_test_seq)
    else:
        print("Skipping RUL prediction (model not trained or no test sequences).")

    # --- Step 10: Evaluate Models ---
    print("\n[Step 10] Evaluating model performance...")
    # Align test data for evaluation (remove first `sequence_length` rows for non-sequence models)
    eval_start_index = SEQUENCE_LENGTH if X_test_seq is not None and X_test_seq.shape[0] > 0 else 0
    if eval_start_index >= len(test_data_scaled):
        print("Warning: Test set too small for evaluation after sequence alignment.")
    else:
        evaluation = pm_system.evaluate_models(
            X_test_scaled[eval_start_index:], # Use scaled features, aligned
            test_data_raw[failure_col].iloc[eval_start_index:], # Use raw labels, aligned
            X_test_seq, # Sequences already aligned
            y_test_rul # RUL labels already aligned
        )
        # Plots from evaluation will be shown automatically

    # --- Step 11: Visualize Health Status ---
    print("\n[Step 11] Visualizing component health status for test data...")
    # Use raw test data for visualization, align with predictions
    viz_data_raw = test_data_raw.iloc[eval_start_index:].copy()
    # Align predictions
    viz_failure_probs = failure_probs[eval_start_index:]
    viz_anomaly_flags = anomaly_results['is_anomaly'].iloc[eval_start_index:].values
    # RUL predictions are already aligned with the end of sequences
    viz_rul_preds = rul_predictions

    if len(viz_data_raw) > 0:
        pm_system.visualize_health_status(
            component_data=viz_data_raw,
            feature_cols=feature_cols,
            failure_probs=viz_failure_probs,
            rul_values=viz_rul_preds,
            anomaly_flags=viz_anomaly_flags,
            timestamp_col=None # Use index
        )
        plt.show() # Ensure plot is displayed
    else:
        print("Skipping visualization due to insufficient aligned test data.")

    # --- Step 12: Save Models ---
    print(f"\n[Step 12] Saving trained models to {MODEL_SAVE_DIR}...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    pm_system.save_models(MODEL_SAVE_DIR)
    # Optionally save feature columns and sequence length
    config_path = os.path.join(MODEL_SAVE_DIR, "training_config.json")
    config_data = {"feature_cols": feature_cols, "sequence_length": SEQUENCE_LENGTH}
    import json
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

    print("\nExample completed successfully!")
    print("\nNext steps:")
    print(f"1. Explore saved models in '{MODEL_SAVE_DIR}'")
    print("2. Use these models in the dashboard (load from the sidebar)")
    print("3. Adapt this script or the system for your specific data and requirements")

if __name__ == "__main__":
    main()