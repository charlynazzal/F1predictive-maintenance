# Predictive Maintenance AI System for Formula One

This project implements an AI-powered predictive maintenance system specifically designed for Formula One and automotive applications. It aims to solve the critical problem of component failure prediction, optimizing maintenance schedules, and improving vehicle reliability.

## Overview

The system integrates multiple AI technologies:

1. **Anomaly Detection** - Identifies unusual patterns in sensor data that might indicate early-stage component degradation.
2. **Failure Classification** - Predicts the probability of component failure based on current sensor readings.
3. **Remaining Useful Life (RUL) Prediction** - Estimates how much operational time remains before a component needs replacement.

## Components

### Core Predictive Maintenance System

The `PredictiveMaintenanceSystem` class provides the core functionality:

- Data preprocessing and feature engineering
- Anomaly detection using Isolation Forest
- Failure classification using Random Forest
- RUL prediction using LSTM neural networks
- Visualization and model evaluation utilities

### Interactive Dashboard

A web-based dashboard built with Streamlit provides:

- Real-time monitoring of component health
- Visualization of sensor data and predictions
- Model training interface
- Data analysis tools
- Anomaly detection results

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/charlynazzal/F1predictive-maintenance.git
   cd F1predictive-maintenance
   ```

2. **Recommended:** Create and activate a virtual environment (e.g., using Python 3.10 or 3.11):
   ```bash
   # Example using Python 3.10
   py -3.10 -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   # source .venv/bin/activate
   ```

3. Install dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. Start the dashboard:
   ```
   streamlit run dashboard.py
   ```

2. Use the provided demo data or upload your own CSV file with sensor readings

3. Train the models using the Training tab

4. Monitor component health in real-time on the Dashboard tab

## Data Format

The system expects data with the following columns:

- `timestamp`: Date and time of the reading
- Sensor columns (e.g., `bearing1_horz`, `bearing1_vert`, `temperature`)
- `failure`: Binary flag indicating if the component failed (1) or not (0)
- `rul`: Remaining useful life in days

## Use Cases

### Formula One Applications

- **Race Weekend Reliability**: Monitor critical components during practice, qualifying, and race
- **Between-Race Maintenance**: Optimize component replacement between races
- **Development Testing**: Identify potential failure modes during development testing

### Consumer Automotive Applications

- **Fleet Management**: Predict maintenance needs across vehicle fleets
- **Warranty Optimization**: Reduce warranty costs through timely maintenance
- **Quality Control**: Identify patterns that indicate manufacturing issues

## Future Enhancements

- Integration with digital twin technology
- Real-time data streaming from vehicle telemetry
- Multi-component health assessment
- Explainable AI features to help engineers understand predictions
- Transfer learning capabilities to adapt to new component types

## Acknowledgments

- NASA Prognostics Data Repository for inspiration and benchmark datasets
- Formula One technical regulations and maintenance practices

- Charly
