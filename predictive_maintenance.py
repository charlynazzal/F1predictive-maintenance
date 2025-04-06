import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

class PredictiveMaintenanceSystem:
    """
    A predictive maintenance system for automotive/F1 components that:
    1. Processes sensor data to detect anomalies
    2. Predicts component failure probability
    3. Estimates remaining useful life (RUL)
    """

    def __init__(self):
        self.anomaly_detector = None
        self.failure_classifier = None
        self.rul_predictor = None
        self.scaler = StandardScaler()
        self.fitted_scaler = False

    def fit_scaler(self, data, numerical_cols):
        """
        Fit the StandardScaler on the training data.

        Parameters:
        - data: DataFrame containing training data.
        - numerical_cols: List of column names to scale.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if not all(col in data.columns for col in numerical_cols):
            raise ValueError("One or more specified numerical_cols not found in data.")

        self.scaler.fit(data[numerical_cols])
        self.fitted_scaler = True
        print("Scaler fitted successfully.")

    def preprocess_data(self, data, numerical_cols, vibration_col=None, add_fft_feature=False):
        """
        Preprocess sensor data for analysis: fill missing values and scale features.
        Assumes scaler has already been fitted using fit_scaler.

        Parameters:
        - data: DataFrame with sensor readings.
        - numerical_cols: List of numerical columns to scale.
        - vibration_col: Name of the vibration sensor column for FFT (optional).
        - add_fft_feature: Boolean flag to enable FFT feature calculation.

        Returns:
        - Processed DataFrame.
        """
        if not self.fitted_scaler:
            raise ValueError("Scaler has not been fitted yet. Call fit_scaler() first.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        processed_data = data.copy()

        # Fill missing values
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')

        # Scale numerical features using the fitted scaler
        if not all(col in processed_data.columns for col in numerical_cols):
             raise ValueError("One or more specified numerical_cols not found in data for scaling.")
        processed_data[numerical_cols] = self.scaler.transform(processed_data[numerical_cols])

        # Feature engineering: FFT
        if add_fft_feature:
            if vibration_col is None or vibration_col not in processed_data.columns:
                raise ValueError("vibration_col must be specified and exist in data if add_fft_feature is True.")
            processed_data[f'{vibration_col}_fft_mag'] = self._extract_frequency_features(processed_data[vibration_col])

        return processed_data

    def _extract_frequency_features(self, vibration_data, n_features=1):
        """Extract dominant frequency magnitude feature from vibration sensor data using FFT."""
        # Simple implementation: Magnitude of the dominant frequency component
        # Can be expanded for more complex features
        try:
            fft_result = np.fft.fft(vibration_data.values)
            magnitudes = np.abs(fft_result)
            # Exclude the DC component (index 0)
            dominant_freq_magnitude = np.max(magnitudes[1:len(magnitudes)//2]) if len(magnitudes) > 1 else 0
            # Return a series with the same index as input
            return pd.Series([dominant_freq_magnitude] * len(vibration_data), index=vibration_data.index)
        except Exception as e:
            print(f"Error during FFT calculation: {e}")
            return pd.Series([0] * len(vibration_data), index=vibration_data.index) # Return zeros or handle appropriately

    def _create_sequences(self, data, feature_cols, target_col, sequence_length):
        """
        Create sequences for LSTM input.

        Parameters:
        - data: DataFrame with processed data (including target_col).
        - feature_cols: List of columns to use as features.
        - target_col: Name of the column to use as the target (e.g., 'rul').
        - sequence_length: Number of time steps in each sequence.

        Returns:
        - X_sequences (np.array): 3D array [samples, time steps, features]
        - y_sequences (np.array): 1D array of target values
        """
        X_sequences = []
        y_sequences = []

        if not all(col in data.columns for col in feature_cols):
            raise ValueError("One or more feature_cols not found in data.")
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")

        for i in range(len(data) - sequence_length):
            X_sequences.append(data.iloc[i:i+sequence_length][feature_cols].values)
            y_sequences.append(data.iloc[i+sequence_length][target_col])

        return np.array(X_sequences), np.array(y_sequences)

    def train_anomaly_detector(self, normal_data, contamination=0.05, n_estimators=100, random_state=42):
        """
        Train an anomaly detection model on normal operation data.
        Assumes normal_data is already preprocessed (scaled).

        Parameters:
        - normal_data: DataFrame with scaled sensor readings during normal operation.
        - contamination: Expected proportion of anomalies in the data.
        - n_estimators: Number of trees in the forest.
        - random_state: Random seed for reproducibility.
        """
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.anomaly_detector.fit(normal_data)
        print("Anomaly detector trained successfully")

    def detect_anomalies(self, test_data):
        """
        Detect anomalies in new sensor data.
        Assumes test_data is already preprocessed (scaled).

        Parameters:
        - test_data: Scaled sensor data to analyze.

        Returns:
        - DataFrame with original data, anomaly scores, and binary flags.
        """
        if self.anomaly_detector is None:
            raise ValueError("Anomaly detector has not been trained yet")

        anomaly_scores = self.anomaly_detector.decision_function(test_data)
        anomaly_predictions = self.anomaly_detector.predict(test_data)

        # Convert predictions to binary (1 for anomaly, 0 for normal)
        anomaly_flags = np.where(anomaly_predictions == -1, 1, 0)

        # Create result DataFrame
        result_data = test_data.copy() # Start with scaled data
        result_data['anomaly_score'] = anomaly_scores
        result_data['is_anomaly'] = anomaly_flags

        return result_data

    def train_failure_classifier(self, X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
        """
        Train a model to predict component failure probability.
        Assumes X_train is already preprocessed (scaled).

        Parameters:
        - X_train: Scaled feature matrix of sensor data.
        - y_train: Binary labels (0=healthy, 1=failed).
        - n_estimators: Number of trees in the forest.
        - max_depth: Maximum depth of the trees.
        - random_state: Random seed.
        """
        self.failure_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.failure_classifier.fit(X_train, y_train)
        print("Failure classifier trained successfully")

    def predict_failure_probability(self, X_test):
        """
        Predict probability of component failure.
        Assumes X_test is already preprocessed (scaled).

        Parameters:
        - X_test: Scaled feature matrix of new sensor data.

        Returns:
        - Array of failure probabilities.
        """
        if self.failure_classifier is None:
            raise ValueError("Failure classifier has not been trained yet")

        failure_probs = self.failure_classifier.predict_proba(X_test)[:, 1]
        return failure_probs

    def build_rul_predictor(self, sequence_length, n_features, lstm_units_1=64, lstm_units_2=32, dense_units=16, dropout_rate=0.2):
        """
        Build an LSTM model to predict Remaining Useful Life.

        Parameters:
        - sequence_length: Number of time steps in each input sequence.
        - n_features: Number of features in each time step.
        - lstm_units_1: Units in the first LSTM layer.
        - lstm_units_2: Units in the second LSTM layer.
        - dense_units: Units in the Dense layer.
        - dropout_rate: Dropout rate.
        """
        self.rul_predictor = Sequential([
            LSTM(lstm_units_1, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units_2, activation='relu'),
            Dropout(dropout_rate),
            Dense(dense_units, activation='relu'),
            Dense(1) # Output is RUL value
        ])
        self.rul_predictor.compile(optimizer='adam', loss='mean_squared_error')
        print("RUL predictor built successfully")

    def train_rul_predictor(self, X_train_seq, y_train_rul, epochs=50, batch_size=32, validation_split=0.2, plot_history=True):
        """
        Train the RUL predictor model.

        Parameters:
        - X_train_seq: 3D array of sequences [samples, time steps, features].
        - y_train_rul: Array of RUL values.
        - epochs: Number of training epochs.
        - batch_size: Batch size.
        - validation_split: Proportion for validation.
        - plot_history: Whether to plot training/validation loss.
        """
        if self.rul_predictor is None:
            raise ValueError("RUL predictor has not been built yet. Call build_rul_predictor() first.")

        history = self.rul_predictor.fit(
            X_train_seq, y_train_rul,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        if plot_history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('RUL Predictor Training History')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.show()

        print("RUL predictor trained successfully")

    def predict_rul(self, X_test_seq):
        """
        Predict Remaining Useful Life for components.

        Parameters:
        - X_test_seq: 3D array of sequences of sensor data.

        Returns:
        - Array of predicted RUL values.
        """
        if self.rul_predictor is None:
            raise ValueError("RUL predictor has not been trained yet")

        rul_predictions = self.rul_predictor.predict(X_test_seq)
        return rul_predictions.flatten()

    def visualize_health_status(self, component_data, feature_cols, failure_probs=None, rul_values=None, anomaly_flags=None, timestamp_col='timestamp'):
        """
        Visualize component health status using original (unscaled) data.

        Parameters:
        - component_data: DataFrame with original sensor data and timestamps.
        - feature_cols: List of feature columns to plot.
        - failure_probs: Array of failure probabilities.
        - rul_values: Array of RUL predictions.
        - anomaly_flags: Array of anomaly detection results (0 or 1).
        - timestamp_col: Name of the timestamp column.
        """
        if timestamp_col not in component_data.columns:
             # Use index if timestamp column not found
             time_axis = component_data.index
             x_label = 'Sample Index'
        else:
             time_axis = component_data[timestamp_col]
             x_label = 'Timestamp'

        plt.figure(figsize=(15, 12)) # Increased height

        # Plot 1: Raw sensor data (select subset)
        plt.subplot(3, 1, 1)
        plot_cols = [col for col in feature_cols if col in component_data.columns][:5] # Plot up to 5 features
        for col in plot_cols:
            plt.plot(time_axis, component_data[col], label=col)
        plt.title('Raw Sensor Data (Subset)')
        plt.xlabel(x_label)
        plt.ylabel('Sensor Value')
        plt.legend()
        plt.grid(True)

        # Plot 2: Failure probability and anomalies
        plt.subplot(3, 1, 2)
        ax2 = plt.gca()
        if failure_probs is not None:
            # Ensure indices align if lengths differ (e.g., due to sequence creation)
            fp_time_axis = time_axis[-len(failure_probs):] if len(failure_probs) < len(time_axis) else time_axis[:len(failure_probs)]
            ax2.plot(fp_time_axis, failure_probs, 'r-', label='Failure Probability')

        if anomaly_flags is not None:
            # Ensure indices align
            anom_time_axis = time_axis[-len(anomaly_flags):] if len(anomaly_flags) < len(time_axis) else time_axis[:len(anomaly_flags)]
            anomaly_indices = np.where(anomaly_flags == 1)[0]
            if len(anomaly_indices) > 0:
                 # Map anomaly_indices back to the time axis
                 anom_timestamps = anom_time_axis[anomaly_indices]
                 ax2.scatter(anom_timestamps, failure_probs[anomaly_indices] if failure_probs is not None and len(failure_probs) == len(anomaly_flags) else [0.95] * len(anomaly_indices),
                             color='orange', marker='o', s=50, label='Anomalies', zorder=5) # zorder to draw on top

        ax2.set_title('Failure Probability and Anomalies')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Probability / Anomaly')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Remaining Useful Life
        plt.subplot(3, 1, 3)
        if rul_values is not None:
            # Ensure indices align
            rul_time_axis = time_axis[-len(rul_values):] if len(rul_values) < len(time_axis) else time_axis[:len(rul_values)]
            plt.plot(rul_time_axis, rul_values, 'g-', label='Predicted RUL')

            # Example threshold
            plt.axhline(y=20, color='purple', linestyle='--', label='Maintenance Threshold (e.g., 20 days)')

        plt.title('Remaining Useful Life Prediction')
        plt.xlabel(x_label)
        plt.ylabel('RUL (Units)') # Make unit clear if possible (e.g., days)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate_models(self, X_test_scaled, y_test_failure, X_test_seq=None, y_test_rul=None):
        """
        Evaluate the performance of trained models.
        Assumes X_test_scaled is already preprocessed (scaled).

        Parameters:
        - X_test_scaled: Scaled feature matrix for testing failure classifier.
        - y_test_failure: True failure labels.
        - X_test_seq: Sequence data for testing RUL predictor (already scaled within sequences).
        - y_test_rul: True RUL values.
        """
        results = {}

        # Evaluate failure classifier
        if self.failure_classifier is not None and y_test_failure is not None:
            y_pred = self.failure_classifier.predict(X_test_scaled)
            failure_accuracy = accuracy_score(y_test_failure, y_pred)
            report = classification_report(y_test_failure, y_pred, output_dict=True)

            results['failure_accuracy'] = failure_accuracy
            results['failure_report'] = classification_report(y_test_failure, y_pred) # Store string report too

            print(f"--- Failure Classifier Evaluation ---")
            print(f"Accuracy: {failure_accuracy:.4f}")
            print("Classification Report:")
            print(results['failure_report'])

            # Plot confusion matrix
            cm = confusion_matrix(y_test_failure, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.failure_classifier.classes_,
                        yticklabels=self.failure_classifier.classes_)
            plt.title('Confusion Matrix - Failure Prediction')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

        # Evaluate RUL predictor
        if self.rul_predictor is not None and X_test_seq is not None and y_test_rul is not None:
            y_pred_rul = self.predict_rul(X_test_seq)
            mse = mean_squared_error(y_test_rul, y_pred_rul)
            mae = mean_absolute_error(y_test_rul, y_pred_rul)

            results['rul_mse'] = mse
            results['rul_mae'] = mae

            print(f"--- RUL Predictor Evaluation ---")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")

            # Plot predicted vs actual RUL
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test_rul, y_pred_rul, alpha=0.6, edgecolors='k', s=50)
            # Add identity line
            lims = [min(min(y_test_rul), min(y_pred_rul)), max(max(y_test_rul), max(y_pred_rul))]
            plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal Prediction')
            plt.title('Predicted vs Actual RUL')
            plt.xlabel('Actual RUL')
            plt.ylabel('Predicted RUL')
            plt.legend()
            plt.grid(True)
            plt.axis('equal') # Ensure x and y axes have the same scale
            plt.xlim(lims)
            plt.ylim(lims)
            plt.show()

        return results

    def save_models(self, path):
        """Save trained models and the scaler to disk."""
        os.makedirs(path, exist_ok=True) # Ensure directory exists

        if self.anomaly_detector is not None:
            joblib.dump(self.anomaly_detector, os.path.join(path, "anomaly_detector.pkl"))

        if self.failure_classifier is not None:
            joblib.dump(self.failure_classifier, os.path.join(path, "failure_classifier.pkl"))

        if self.rul_predictor is not None:
            self.rul_predictor.save(os.path.join(path, "rul_predictor.h5"))

        # Save scaler
        if self.fitted_scaler:
             joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        else:
             print("Warning: Scaler was not fitted, not saving scaler.")

        print(f"Models and scaler saved to {path}")

    def load_models(self, path):
        """Load trained models and the scaler from disk."""
        try:
            anomaly_path = os.path.join(path, "anomaly_detector.pkl")
            if os.path.exists(anomaly_path):
                self.anomaly_detector = joblib.load(anomaly_path)

            classifier_path = os.path.join(path, "failure_classifier.pkl")
            if os.path.exists(classifier_path):
                self.failure_classifier = joblib.load(classifier_path)

            rul_path = os.path.join(path, "rul_predictor.h5")
            if os.path.exists(rul_path):
                self.rul_predictor = tf.keras.models.load_model(rul_path)

            scaler_path = os.path.join(path, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.fitted_scaler = True # Mark scaler as loaded/fitted
            else:
                 self.fitted_scaler = False # Ensure flag is correct if file missing

            print(f"Models and scaler loaded from {path}")
        except Exception as e:
            print(f"Error loading models from {path}: {e}")
            # Reset states if loading fails partially
            self.__init__() # Re-initialize to default state


# Example usage with NASA Bearing Dataset (Modified for Refactoring)
def demo_with_nasa_bearing_dataset():
    """
    Demonstrate the Predictive Maintenance System using synthetic bearing data.
    """
    # --- 1. Generate Synthetic Data ---
    def generate_synthetic_bearing_data(n_samples=288, freq='10T'): # 2 days default
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq=freq)
        data = pd.DataFrame({'timestamp': timestamps})

        # Base sensor readings
        data['bearing1_horz'] = np.random.normal(0, 0.1, n_samples)
        data['bearing1_vert'] = np.random.normal(0, 0.1, n_samples)
        data['bearing2_horz'] = np.random.normal(0, 0.1, n_samples)
        data['bearing2_vert'] = np.random.normal(0, 0.1, n_samples)
        data['temperature'] = np.random.normal(50, 5, n_samples) + np.linspace(0, 5, n_samples) # Add slight temp increase

        # Add degradation trend (linear increase in vibration)
        degradation = np.linspace(0, 1.5, n_samples)
        data['bearing1_horz'] += degradation * 0.3
        data['bearing1_vert'] += degradation * 0.2

        # Add anomalies (random spikes)
        anomaly_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.03), replace=False)
        data.loc[anomaly_indices, 'bearing1_horz'] += np.random.uniform(1.0, 2.5, size=len(anomaly_indices))

        # Create failure labels (fail in last 10% of lifecycle)
        failure_point = int(0.9 * n_samples)
        data['failure'] = 0
        data.loc[failure_point:, 'failure'] = 1

        # Create RUL (Remaining Useful Life in days)
        # RUL decreases linearly towards the end of the known lifecycle (n_samples)
        samples_per_day = pd.Timedelta('1 day') / pd.Timedelta(freq)
        data['rul'] = (n_samples - 1 - data.index) / samples_per_day

        return data.set_index('timestamp') # Set timestamp as index

    print("Generating synthetic NASA bearing dataset...")
    bearing_data = generate_synthetic_bearing_data(n_samples=1000, freq='1T') # More data, 1 min freq
    print(f"Dataset shape: {bearing_data.shape}")
    print(bearing_data.head())

    # --- 2. Initialize System & Define Features ---
    pm_system = PredictiveMaintenanceSystem()
    # Define columns used for modeling
    feature_cols = ['bearing1_horz', 'bearing1_vert', 'bearing2_horz', 'bearing2_vert', 'temperature']
    failure_col = 'failure'
    rul_col = 'rul'
    vibration_col_fft = 'bearing1_horz' # Example for FFT feature

    # --- 3. Split Data ---
    train_size = int(0.7 * len(bearing_data))
    training_data_raw = bearing_data.iloc[:train_size].copy()
    testing_data_raw = bearing_data.iloc[train_size:].copy()
    print(f"Training data shape: {training_data_raw.shape}")
    print(f"Testing data shape: {testing_data_raw.shape}")

    # --- 4. Fit Scaler ---
    print("Fitting scaler on training data...")
    pm_system.fit_scaler(training_data_raw, feature_cols)

    # --- 5. Preprocess Data (Transform using fitted scaler) ---
    print("Preprocessing training and testing data...")
    training_data_scaled = pm_system.preprocess_data(training_data_raw, feature_cols, vibration_col=vibration_col_fft, add_fft_feature=False) # No FFT for this demo
    testing_data_scaled = pm_system.preprocess_data(testing_data_raw, feature_cols, vibration_col=vibration_col_fft, add_fft_feature=False) # No FFT for this demo

    # --- 6. Train Anomaly Detector ---
    # Use only 'normal' data from training set (where failure == 0)
    normal_data_scaled = training_data_scaled[training_data_raw[failure_col] == 0][feature_cols]
    print(f"Training anomaly detector on {len(normal_data_scaled)} normal samples...")
    pm_system.train_anomaly_detector(normal_data_scaled, contamination=0.03, n_estimators=150) # Adjusted params

    # --- 7. Detect Anomalies in Test Data ---
    print("Detecting anomalies in test data...")
    # Pass only feature columns for detection
    anomaly_results = pm_system.detect_anomalies(testing_data_scaled[feature_cols])
    print(f"Detected {anomaly_results['is_anomaly'].sum()} anomalies in test set.")

    # --- 8. Train Failure Classifier ---
    print("Training failure classifier...")
    X_train_class = training_data_scaled[feature_cols]
    y_train_class = training_data_raw[failure_col] # Use original failure labels
    pm_system.train_failure_classifier(X_train_class, y_train_class, n_estimators=150, max_depth=12) # Adjusted params

    # --- 9. Predict Failure Probability ---
    print("Predicting failure probability on test data...")
    X_test_class = testing_data_scaled[feature_cols]
    failure_probs = pm_system.predict_failure_probability(X_test_class)

    # --- 10. Prepare Sequences for RUL ---
    sequence_length = 30 # e.g., 30 minutes
    print(f"Creating sequences for RUL prediction (length={sequence_length})...")
    # Important: Create sequences from the *scaled* data, including the RUL target for y
    training_data_for_seq = training_data_scaled.copy()
    training_data_for_seq[rul_col] = training_data_raw[rul_col] # Add original RUL back for target
    X_train_seq, y_train_rul = pm_system._create_sequences(training_data_for_seq, feature_cols, rul_col, sequence_length)

    testing_data_for_seq = testing_data_scaled.copy()
    testing_data_for_seq[rul_col] = testing_data_raw[rul_col] # Add original RUL back for target
    X_test_seq, y_test_rul = pm_system._create_sequences(testing_data_for_seq, feature_cols, rul_col, sequence_length)
    print(f"Train sequences: X shape={X_train_seq.shape}, y shape={y_train_rul.shape}")
    print(f"Test sequences: X shape={X_test_seq.shape}, y shape={y_test_rul.shape}")

    # --- 11. Build and Train RUL Predictor ---
    n_features = X_train_seq.shape[2]
    print("Building RUL predictor...")
    pm_system.build_rul_predictor(sequence_length, n_features, lstm_units_1=70, lstm_units_2=35) # Adjusted params

    print("Training RUL predictor...")
    # Reduced epochs for faster demo, increase for real use
    pm_system.train_rul_predictor(X_train_seq, y_train_rul, epochs=20, batch_size=64, validation_split=0.15, plot_history=True)

    # --- 12. Predict RUL ---
    print("Predicting Remaining Useful Life on test sequences...")
    rul_predictions = pm_system.predict_rul(X_test_seq)

    # --- 13. Evaluate Models ---
    print("Evaluating models on test data...")
    # Note: Pass scaled features for classifier evaluation
    eval_results = pm_system.evaluate_models(
        X_test_scaled[sequence_length:][feature_cols], # Align classifier test set with sequence output
        testing_data_raw[failure_col].iloc[sequence_length:], # Align labels
        X_test_seq,
        y_test_rul
    )

    # --- 14. Visualize Results ---
    print("Visualizing health status for test data...")
    # Use the raw test data for visualization, align with predictions
    viz_data = testing_data_raw.iloc[sequence_length:].copy()
    # Align predictions/anomaly flags with the visualization data timeframe
    viz_failure_probs = failure_probs[sequence_length:]
    viz_anomaly_flags = anomaly_results['is_anomaly'].values[sequence_length:]
    viz_rul_values = rul_predictions # Already aligned from _create_sequences output

    pm_system.visualize_health_status(
        viz_data,
        feature_cols=feature_cols,
        failure_probs=viz_failure_probs,
        rul_values=viz_rul_values,
        anomaly_flags=viz_anomaly_flags,
        timestamp_col=None # Use index since we set it earlier
    )

    # --- 15. Save/Load Models (Optional Demo) ---
    model_save_path = "./trained_models"
    print(f"Saving models to {model_save_path}...")
    pm_system.save_models(model_save_path)

    print("Creating new system instance and loading models...")
    new_pm_system = PredictiveMaintenanceSystem()
    new_pm_system.load_models(model_save_path)
    print("Models loaded into new instance.")

    # Example: Make a prediction with the loaded model
    if new_pm_system.rul_predictor is not None and X_test_seq.shape[0] > 0:
         loaded_rul_pred = new_pm_system.predict_rul(X_test_seq[:5]) # Predict on first 5 test sequences
         print("Example RUL prediction using loaded model (first 5 test sequences):")
         print(loaded_rul_pred)

    print("Demo completed successfully!")


# Run the demo if executed directly
if __name__ == "__main__":
    demo_with_nasa_bearing_dataset()