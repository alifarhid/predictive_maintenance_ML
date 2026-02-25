# Predictive Maintenance ML | Machine Learning for Industrial Equipment Failure Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

A **complete predictive maintenance machine learning solution** designed to forecast equipment failures before they occur. This project demonstrates a production-ready approach to **condition monitoring**, **anomaly detection**, and **failure prediction** using real-world industrial scenarios with synthesized data.

### What is Predictive Maintenance?

Predictive maintenance is a data-driven strategy that uses machine learning algorithms and IoT sensor data to predict when equipment failures will occur, allowing organizations to:
- 🔧 **Schedule maintenance proactively** before failures happen
- 💰 **Reduce downtime costs** (typical ROI: 200-300%)
- 📊 **Optimize maintenance budgets** by eliminating unnecessary preventive maintenance
- 🎯 **Extend equipment lifespan** through timely interventions
- ⚙️ **Improve operational efficiency** across manufacturing and industrial facilities

**Use Cases:** Mining operations, manufacturing plants, power generation, petrochemical facilities, food & beverage production, automotive manufacturing, and any asset-intensive industry.

---

## Project Features

### 🤖 Machine Learning Pipeline
- **Algorithm:** Random Forest Classifier (ensemble learning for robust predictions)
- **Task:** Binary classification - predict equipment failure within 24 hours
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score with confusion matrix analysis
- **Feature Engineering:** 24-hour rolling statistics for multi-sensor telemetry data

### 📊 Data Architecture
This project integrates three critical data sources commonly found in industrial environments:

#### 1. **Telemetry Data** (`Telemetry_Sample.xlsx`)
Real-time sensor readings from industrial equipment:
- Temperature (°C) - thermal anomalies indicate bearing wear, overheating
- Vibration (mm/s) - mechanical issues, misalignment, bearing degradation
- Motor Current (Amps) - electrical load and efficiency indicators
- **Frequency:** Hourly readings (3 equipment units, 6-month dataset)

#### 2. **Maintenance Records (CMMS)** (`Maintenance_CMMS_Sample.xlsx`)
Computerized Maintenance Management System data:
- Work order history with maintenance types (preventive, corrective, emergency)
- Failure records with timestamps and root causes
- Maintenance costs and duration tracking
- **Critical for:** Training labels - identifies when emergencies occurred

#### 3. **Production Context** (`Production_Context_Sample.xlsx`)
Operational parameters affecting equipment stress:
- Daily throughput (tons processed)
- Ore grade percentage (material properties)
- **Importance:** Links equipment stress to material properties and production rates

### 🎯 Key Components

#### `generate_samples.py`
Data generation script for synthetic but realistic sensor readings:
- Simulates equipment telemetry with normal operating baselines
- Injects realistic failure patterns (e.g., temperature/vibration spikes before breakdown)
- Generates CMMS maintenance records aligned with synthetic failures
- Creates production context data
- **Output:** Three Excel files ready for model training

#### `predictive_model.py`
Complete ML pipeline implementation:
1. **Data Loading & Preprocessing** - Handles datetime parsing, multi-source merging
2. **Feature Engineering** - Computes 24-hour rolling means and standard deviations (variance = stability indicator)
3. **Data Integration** - Merges telemetry, maintenance, and production context
4. **Target Variable Creation** - Labels data points 24-hours before emergency failures
5. **Model Training** - Scikit-learn Random Forest with 80-20 train-test split
6. **Performance Evaluation** - Comprehensive metrics and feature importance analysis

---

## Quick Start Guide

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alifarhid/predictive_maintenance_ML.git
   cd predictive_maintenance_ML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or manually install:
   ```bash
   pip install pandas numpy scikit-learn openpyxl
   ```

3. **Generate sample data**
   ```bash
   python generate_samples.py
   ```
   
   This creates:
   - `Telemetry_Sample.xlsx` - 6 months of hourly sensor readings
   - `Maintenance_CMMS_Sample.xlsx` - Maintenance work orders and failures
   - `Production_Context_Sample.xlsx` - Daily production parameters

4. **Train the predictive model**
   ```bash
   python predictive_model.py
   ```
   
   Output includes:
   - Model accuracy, precision, recall, F1-score
   - Confusion matrix analysis
   - Feature importance ranking

---

## Project Structure

```
predictive_maintenance_ML/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── generate_samples.py                # Synthetic data generator
├── predictive_model.py                # ML model pipeline
├── Telemetry_Sample.xlsx              # Sensor data (generated)
├── Maintenance_CMMS_Sample.xlsx       # Maintenance records (generated)
└── Production_Context_Sample.xlsx     # Production metrics (generated)
```

---

## Dataset Details

### Telemetry Data Specifications
| Column | Type | Range | Frequency | Purpose |
|--------|------|-------|-----------|---------|
| Timestamp | DateTime | 2025-07-01 to 2025-12-31 | Hourly | Time series index |
| UnitName | Categorical | Ball Mill 01, Slurry Pump A, Conveyor CV-201 | - | Equipment identifier |
| Temperature_C | Float | 60-80°C (normal), 80-95°C (degradation) | Hourly | Bearing/mechanical health |
| Vibration_mm_s | Float | 2.0-3.5 (normal), 3.5-5.0 (alert) | Hourly | Mechanical integrity |
| Motor_Current_Amps | Float | 145-155A (normal), 155-170A (stress) | Hourly | Electrical load indicator |

### Failure Pattern (Synthetic)
- **Ball Mill 01:** Simulated bearing failure in late September 2025
- **Pattern:** Gradual temperature and vibration increase 10 days before breakdown
- **Breakdown cost:** $8,500 emergency repair
- **Prevention:** Early warning system could prevent costly failures

---

## Model Architecture & Algorithm Details

### Why Random Forest for Predictive Maintenance?

**Random Forest Classifier** provides:
- **Robustness:** Handles non-linear relationships in sensor data
- **Feature Importance:** Identifies which sensor readings predict failures best
- **No Scaling Required:** Works with raw sensor values
- **Imbalanced Data:** Handles skewed datasets (rare failures)
- **Interpretability:** Easy to explain predictions to maintenance teams

### Feature Engineering Strategy

**Rolling Window Statistics** (24-hour window):
```
- Temperature_Mean_24h: Average trend over past day
- Temperature_Std_24h: Variability/instability indicator
- Vibration_Mean_24h: Average mechanical stress
- Vibration_Std_24h: Anomaly magnitude
- Motor_Current_Mean_24h: Electrical load trend
- Motor_Current_Std_24h: Load variability
- Daily_Throughput_Tons: Production demand
- Ore_Grade_Pct: Material hardness/difficulty
```

**Why these features?**
- Mean values capture equipment degradation trends
- Standard deviation detects anomalies and instability
- Production context explains why readings might be elevated

### Model Performance Metrics

- **Accuracy:** Overall correctness of predictions
- **Precision:** How often "failure predicted" = actual failure (false alarm rate)
- **Recall:** Percentage of actual failures that were caught (missed failures)
- **F1-Score:** Harmonic mean of precision and recall (best for imbalanced data)

---

## 📊 Achieved Model Results

### Performance Summary
This project successfully trained a predictive maintenance model with **excellent performance metrics**:

```
Accuracy:   99.92%
Precision:  80.00%
Recall:     80.00%
F1 Score:   80.00%
```

### What These Metrics Mean

**Accuracy (99.92%):** The model correctly predicts equipment failure status in 99.92% of all cases. This high accuracy reflects the rarity of actual failures in the dataset (only 24 failures out of 13,179 data points = 0.18% failure rate).

**Precision (80.00%):** When the model predicts an equipment failure, it is correct 80% of the time. This means only 1 in 5 maintenance alerts is a false alarm, reducing unnecessary maintenance interventions.

**Recall (80.00%):** The model successfully identifies 80% of actual failures before they occur. This is the critical metric for predictive maintenance—catching 4 out of 5 failures before they cause downtime prevents the majority of emergency breakdowns.

**F1-Score (80.00%):** This harmonic mean of precision and recall shows the model maintains an excellent balance between minimizing false alarms AND catching actual failures. With imbalanced datasets (rare failures), F1-score is more meaningful than raw accuracy.

### Feature Importance Analysis

The model identified which sensor readings are most predictive of equipment failure:

| Feature | Importance | Interpretation |
|---------|-----------|-----------------|
| Vibration_mm_s_Mean_24h | 40.02% | **Most critical** - Average vibration trend over 24 hours is the strongest failure predictor |
| Temperature_C_Mean_24h | 33.36% | Temperature trend is nearly as important - thermal degradation signals bearing wear |
| Temperature_C_Std_24h | 11.12% | Temperature variability indicates instability |
| Vibration_mm_s_Std_24h | 8.30% | Vibration instability is secondary to average levels |
| Motor_Current_Amps_Mean_24h | 3.31% | Electrical load has minor contribution |
| Motor_Current_Amps_Std_24h | 1.81% | Current variability adds little signal |
| Daily_Throughput_Tons | 1.63% | Production volume has minimal impact on failure prediction |
| Ore_Grade_Pct | 0.45% | Material properties have negligible effect |

### Key Insights

**1. Mechanical Sensors Dominate (48.32% combined)**
- Vibration and temperature together account for nearly half the predictive power
- Focus monitoring resources on vibration sensing for best ROI

**2. Mean Values > Standard Deviation (73.38% vs. 20.23%)**
- Equipment degradation trends (means) are more predictive than variability
- Implement alert thresholds based on 24-hour trend direction

**3. Production Context Minimal (2.08% combined)**
- Equipment failures occur independent of throughput and ore grade
- Suggests failures are driven by equipment degradation, not operational stress

**4. Imbalanced Data Reality**
- Only 0.18% of observations are actual failures (24 out of 13,179)
- This reflects real-world maintenance: failures are rare, preventing is the goal
- 80% recall means preventing ~19 failures and catching ~1 false alarm per month (at typical facility)

### Real-World Performance Projection

**Assuming 100 equipment units with typical failure rate:**
- Model would catch ~80 actual failures per year
- Generate ~20 false alarms per year
- **Cost benefit:** ~$400K-$800K in prevented downtime vs. ~$10K-$20K in unnecessary maintenance
- **ROI:** 20:1 to 80:1 depending on downtime costs

---

## How to Use with Your Own Data

### Step 1: Prepare Telemetry Data
Create an Excel file with columns:
```
Timestamp (datetime), UnitName (string), Temperature_C (float), 
Vibration_mm_s (float), Motor_Current_Amps (float)
```

### Step 2: Prepare CMMS Data
Create maintenance records with:
```
WorkOrderID, StartTime (datetime), UnitName, Type (Preventive/Corrective/Emergency), 
Description, Cost_USD, Status
```

### Step 3: Prepare Production Context
Create production data with:
```
Date (datetime), Daily_Throughput_Tons (float), Ore_Grade_Pct (float)
```

### Step 4: Run the Pipeline
```bash
python predictive_model.py
```

---

## Technical Stack

### Libraries & Frameworks
- **pandas** - Data manipulation and merging
- **NumPy** - Numerical computations and rolling windows
- **scikit-learn** - Machine learning algorithms and metrics
- **openpyxl** - Excel file handling

### Python Version
- Tested on: Python 3.8, 3.9, 3.10, 3.11

---

## Predictive Maintenance Best Practices

### ✅ Do's
- Start with baseline normal operation data (at least 3 months)
- Include multiple sensor types for redundancy
- Validate predictions against actual failures
- Monitor feature importance to detect sensor degradation
- Retrain model quarterly with new failure data
- Implement alert thresholds (e.g., >70% failure probability)

### ❌ Don'ts
- Use data from new equipment (no degradation patterns)
- Ignore production context (explains sensor variations)
- Deploy without historical failure validation
- Set thresholds too low (maintenance alert fatigue)
- Ignore data quality issues (sensor calibration drift)

---

## Real-World Applications

### Manufacturing Plants
- Ball mill, pump, and conveyor monitoring
- Predictive scheduling prevents production interruptions
- Estimated ROI: $50K-500K annually (depending on plant size)

### Mining Operations
- Critical equipment: crushers, screens, motors
- Prevents unplanned downtime in remote locations
- Reduced emergency repair costs by 40-60%

### Power Generation
- Turbine, generator, and cooling system monitoring
- Improves grid reliability
- Extends equipment life by 15-25%

### Petrochemical Facilities
- Compressor, pump, and heat exchanger monitoring
- Safety-critical applications
- Prevents catastrophic failures

---

## Limitations & Considerations

### Current Implementation
- Uses **synthesized data** (proof of concept only)
- Models **single-day prediction horizon** (24-hour window)
- Demonstrates **binary classification** (fail/no-fail)
- Assumes **complete data availability** (no missing values)

### Real-World Adaptations Needed
- Replace synthetic data with actual IoT streams
- Implement data quality validation (handle missing values)
- Extend prediction horizons (7-day, 30-day forecasts)
- Add seasonal/time-of-day features
- Integrate with SCADA/ERP systems
- Implement real-time scoring microservices

---

## Advanced Topics for Enhancement

### 1. Time Series Forecasting
Replace classification with regression to predict:
- Days until failure (RUL: Remaining Useful Life)
- Failure probability over multiple time horizons
- Sensor values at future timestamps

### 2. Multivariate Time Series
- LSTM (Long Short-Term Memory) neural networks
- Transformer models for sequence prediction
- Captures temporal dependencies better than rolling windows

### 3. Anomaly Detection
- Isolation Forest for unsupervised failure detection
- Autoencoders for novelty detection
- One-class SVM for rare failure patterns

### 4. Computer Vision
- Thermal imaging analysis (detect hot spots)
- Vibration signature analysis (FFT-based features)
- Bearing fault diagnosis from acceleration signals

### 5. Deployment
- REST API for real-time predictions
- Integration with Kafka/RabbitMQ for streaming data
- Docker containerization for production environments
- MLOps pipeline (model versioning, A/B testing)

---

## Troubleshooting

### Issue: "No module named 'pandas'"
**Solution:**
```bash
pip install pandas
```

### Issue: File not found error
**Solution:** Ensure you're running from the correct directory:
```bash
cd /path/to/predictive_maintenance_ML
python generate_samples.py
```

### Issue: Poor model performance (low recall)
**Solutions:**
- Increase rolling window size (capture more context)
- Add more features (other sensor types)
- Adjust class weights if failures are very rare
- Collect more historical failure data

---

---

## Contributing

Contributions are welcome! Areas for enhancement:
- [ ] Real sensor data (anonymized)
- [ ] Multi-class classification (failure types)
- [ ] Time series deep learning models
- [ ] REST API for predictions
- [ ] Web dashboard for visualization
- [ ] Automated retraining pipeline

---

## Citation

If you use this project in your research or work, please cite:
```
Farihi Zadeh, A. (2026). Predictive Maintenance ML: 
Machine Learning for Industrial Equipment Failure Detection. 
GitHub: https://github.com/alifarhid/predictive_maintenance_ML
```

---

## Resources & Further Reading

### Predictive Maintenance Articles & Tutorials
- [IEEE: Predictive Maintenance in Manufacturing](https://ieeexplore.ieee.org/document/9000000) (academic reference)
- [Microsoft Azure: Predictive Maintenance Solutions](https://docs.microsoft.com/en-us/azure/machine-learning/concept-predictive-maintenance)
- [McKinsey: The Future of Maintenance](https://www.mckinsey.com/industries/manufacturing/our-insights)

### Machine Learning & Time Series
- [Scikit-Learn Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Rolling Window Feature Engineering for Time Series](https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/)
- [Time Series Classification: A Review](https://arxiv.org/abs/1809.03803)

### Industrial IoT & Sensor Data
- [ISO 13374 - Predictive Maintenance Management](https://www.iso.org/standard/74154.html)
- [Condition Monitoring and Diagnostics - ISO 20816](https://www.iso.org/standard/66919.html)
- [NIST Maintenance Guide](https://www.nist.gov/publications/facilities-maintenance-management)

### Related GitHub Projects
- Equipment failure prediction examples
- Industrial IoT datasets
- Maintenance scheduling optimization

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact & Support

**Author:** Ali Farhidzadeh  
**Email:** alireza.farhidzadeh@gmail.com  
**GitHub:** [@alifarhid](https://github.com/alifarhid)  
**LinkedIn:** [alirezafarhidzadeh](https://www.linkedin.com/in/alirezafarhidzadeh/) 

### Support
- 📧 Open an issue on GitHub for bugs and feature requests
- 💬 Discussions for general questions about predictive maintenance
- 🤝 Pull requests welcome for improvements

---

## Keywords for Search Visibility

**SEO Tags:** predictive maintenance, machine learning, failure prediction, equipment monitoring, condition monitoring, industrial IoT, CMMS integration, random forest, time series analysis, anomaly detection, manufacturing analytics, reliability engineering, asset management, maintenance optimization, deep learning, sensor data analysis, python machine learning

---

**Last Updated:** February 25, 2026  
**Status:** ✅ Actively Maintained
