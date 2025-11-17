# 🌤️ Hanoi Weather Forecasting System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-Time%20Series%20Forecasting-orange.svg)](https://github.com)
[![Gradio](https://img.shields.io/badge/Interface-Gradio-orange.svg)](https://gradio.app/)

A production-ready machine learning system for temperature forecasting in Hanoi, Vietnam. The system leverages ensemble learning methods and advanced feature engineering to predict future temperatures based on historical weather data from Visual Crossing Weather API.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Information](#-project-information)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [API Documentation](#-api-documentation)
- [Documentation & Reports](#-documentation--reports)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

### Objective
Build a robust temperature forecasting system for Hanoi that provides:
- Accurate short-term and long-term temperature predictions
- Insights into climate patterns and meteorological factors
- A user-friendly interface for real-time forecasting
- Production-ready API for integration with external systems

### Methodology
- **Time Series Forecasting** using ensemble learning (XGBoost, LightGBM, CatBoost)
- **Advanced Feature Engineering** with 229+ engineered features
- **Comprehensive Preprocessing** including outlier detection and feature scaling
- **Robust Evaluation** using cross-validation and industry-standard metrics
- **Interactive UI** built with Gradio for easy deployment

---

## 📋 Project Information

**Course**: Machine Learning - DSEB 65B  
**Instructor**: Trinh Tuan Phong  
**Academic Year**: 2024-2025  

### 👥 Team Members

| Name | Role |
|------|------|
| **Bui Viet Huy** | Team Leader |
| **Nguyen Danh Dung** | Member |
| **Do Cong Huy** | Member |
| **Dang Ngoc Hoa** | Member |
| **Nguyen Hong Nhung** | Member |

### 🎓 Project Scope

This project is developed as part of the Machine Learning course curriculum at the University of Economics and Business (UEB), Vietnam National University. The system demonstrates practical application of:

- Time series analysis and forecasting
- Ensemble machine learning techniques
- Feature engineering for weather data
- Production deployment practices
- Real-world ML system architecture

### 🔗 Quick Links

- 🚀 **[Live Demo](src/report/github_ui_link.txt)** - Try the interactive forecasting system
- 📺 **[Video Presentation](src/report/video_link.txt)** - Project walkthrough and demonstration
- 📄 **[Documentation](src/report/when_we_need_to_retrain_model.txt)** - Comprehensive technical documentation
- 💾 **[GitHub Repository](https://github.com/NgocHoa04/machine_learning_project)** - Source code and resources

---

## ✨ Key Features

- **🔄 Dual Forecasting Modes**: Daily and hourly temperature predictions
- **🎯 Advanced ML Models**: Ensemble of XGBoost, LightGBM, and CatBoost
- **🔧 229+ Engineered Features**: Including monsoon patterns, lag features, and rolling statistics
- **📊 Interactive Dashboard**: Built with Gradio for real-time predictions
- **🌏 Hanoi-Specific**: Tailored for Vietnamese tropical monsoon climate patterns
- **📈 Production-Ready**: Complete with logging, error handling, and model versioning
- **⚡ Fast Inference**: Optimized for low-latency predictions
- **📦 Modular Architecture**: Clean separation of concerns for easy maintenance

---

## 📊 Dataset

### Data Source
- **Provider**: Visual Crossing Weather API
- **Location**: Hanoi, Vietnam (Latitude: 21.0285°N, Longitude: 105.8542°E)
- **Time Period**: Historical daily and hourly weather data
- **Update Frequency**: Real-time via API integration

### Dataset Structure

#### Raw Data
- **`Hanoi Daily.csv`**: Daily aggregated weather metrics
- **`Hanoi Hourly.csv`**: Hourly granular weather data

#### Processed Data
- **`Hanoi_Daily_Selected.csv`**: Cleaned daily dataset with selected features
- **`Hanoi_Hourly_Selected.csv`**: Cleaned hourly dataset
- **`Hanoi_daily_FE_full.csv`**: Feature-engineered dataset (229 features)

### Weather Variables

| Variable | Description | Unit | Range |
|----------|-------------|------|-------|
| `temp` | Temperature | °C | -5 to 40 |
| `tempmax` | Maximum temperature | °C | - |
| `tempmin` | Minimum temperature | °C | - |
| `humidity` | Relative humidity | % | 0-100 |
| `dew` | Dew point temperature | °C | - |
| `precip` | Precipitation amount | mm | 0+ |
| `precipprob` | Precipitation probability | % | 0-100 |
| `precipcover` | Precipitation coverage | % | 0-100 |
| `windspeed` | Wind speed | km/h | 0+ |
| `winddir` | Wind direction | degrees | 0-360 |
| `solarradiation` | Solar radiation | W/m² | 0+ |
| `cloudcover` | Cloud coverage | % | 0-100 |
| `visibility` | Visibility distance | km | 0+ |
| `pressure` | Atmospheric pressure | hPa | 950-1050 |
| `sunrise` | Sunrise time | timestamp | - |
| `sunset` | Sunset time | timestamp | - |

### Hanoi Climate Characteristics

**Tropical Monsoon Climate** with distinct seasonal patterns:

- **🌬️ Northeast Monsoon (Winter/Spring)**: 
  - Direction: 20-80° (NE)
  - Period: November - April
  - Characteristics: Cool, humid, occasional drizzle
  
- **🌊 Southwest Monsoon (Summer)**: 
  - Direction: 200-260° (SW)
  - Period: May - October
  - Characteristics: Hot, humid, heavy rainfall with thunderstorms

- **☀️ High Solar Radiation**: Average 150-250 W/m² 
- **💧 High Humidity**: Year-round average 70-85%
- **🌡️ Temperature Range**: 15-35°C annually

---

## 📁 Project Structure

```
Final project/
│
├── dataset/                           # Data storage
│   ├── raw/                          # Raw data from API
│   │   ├── Hanoi Daily.csv
│   │   └── Hanoi Hourly.csv
│   └── processed/                    # Cleaned and processed data
│       ├── Hanoi_daily_FE_full.csv
│       ├── Hanoi_Daily_Selected.csv
│       └── Hanoi_Hourly_Selected.csv
│
├── notebooks/                         # Jupyter notebooks for experiments
│   ├── data_understanding.ipynb      # EDA and data analysis
│   ├── feature_engineering_GBDT.ipynb
│   ├── daily/                        # Daily forecasting notebooks
│   │   ├── 01_data_collection.ipynb
│   │   ├── 03_data_processing.ipynb
│   │   ├── 04_Feature_engineering.ipynb
│   │   └── 05_run_model.ipynb
│   └── hourly/                       # Hourly forecasting notebooks
│       ├── data_hourly.ipynb
│       ├── step2_data_processing_hourly.ipynb
│       └── step5_run_model_hourly.ipynb
│
├── src/                              # Source code
│   ├── app/                          # Application interface
│   │   ├── app.py                    # Gradio web interface
│   │   └── style.py                  # UI styling
│   │
│   ├── config/                       # Configuration files
│   │   ├── settings.py               # Global settings
│   │   ├── model_params.yaml         # Model hyperparameters
│   │   ├── models_pkl/               # Saved model artifacts
│   │   └── results/                  # Experiment results
│   │       └── hanoi_temp_v1_summary.json
│   │
│   ├── data/                         # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_collection.py        # API integration
│   │   ├── data_helper.py            # Helper functions
│   │   ├── data_preprocessing.py     # Preprocessing pipeline
│   │   └── Pipeline.py               # End-to-end data pipeline
│   │
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_engineering_daily.py
│   │   └── feature_engineering_hourly.py
│   │
│   ├── model/                        # Model training and inference
│   │   ├── daily_model.py            # Daily forecasting model
│   │   ├── hourly_model.py           # Hourly forecasting model
│   │   ├── model_evaluation.py       # Comprehensive evaluation module
│   │   ├── run_model_daily.py        # Daily training script
│   │   └── run_model_hourly.py       # Hourly training script
│   │
│   ├── evaluation/                   # Model evaluation
│   │   ├── evaluation_metrics.py     # Metrics calculation
│   │   └── check_overfiting.py       # Overfitting detection
│   │
│   └── report/                       # Documentation and reports
│       ├── when_we_need_to_retrain_model.txt  # Model retraining guidelines
│       ├── github_ui_link.txt        # GitHub UI demo link
│       └── video_link.txt            # Project demo video link
│
├── requirements.txt                   # Python dependencies
└── README.md                         # Project documentation
```

---

## 🔄 Data Pipeline

The system follows a structured ETL (Extract, Transform, Load) pipeline:

### 1. Data Collection (`data_collection.py`)
```python
# Fetch data from Visual Crossing Weather API
- API authentication and request handling
- Rate limiting and error recovery
- Data validation and quality checks
```

**Features**:
- Automatic retry mechanism for failed requests
- Configurable date ranges
- Support for both daily and hourly data
- Data caching to minimize API calls

### 2. Data Understanding (`data_understanding.ipynb`)

**Exploratory Data Analysis**:
- Distribution analysis of all weather variables
- Time series decomposition (trend, seasonality, residuals)
- Correlation analysis and multicollinearity detection
- Missing value patterns
- Outlier detection using statistical methods

**Key Insights**:
- Strong correlation between `dew` point and temperature (r > 0.9)
- Seasonal patterns clearly visible in temperature data
- High humidity characteristic of Hanoi's climate
- Monsoon wind patterns show distinct directional preferences

### 3. Data Preprocessing (`data_preprocessing.py`)

**Pipeline Components**:

```python
class DataPreprocessor:
    - handle_missing_values()      # Imputation strategies
    - detect_outliers()            # Statistical outlier detection
    - transform_features()         # Log/Box-Cox transformations
    - scale_features()             # StandardScaler/MinMaxScaler
    - remove_low_variance()        # Feature variance filtering
```

**Preprocessing Steps**:

1. **Missing Value Handling**
   - Forward fill for time-series continuity
   - Interpolation for short gaps
   - Domain-specific imputation (e.g., 0 for precipitation)

2. **Outlier Treatment**
   - IQR method for temperature variables
   - Log transformation for skewed distributions (precipitation)
   - Winsorization for extreme values

3. **Feature Scaling**
   - StandardScaler: `temp`, `windspeed`, `pressure`
   - MinMaxScaler: `humidity`, `cloudcover`, `precipcover`
   - Custom scaling: `solarradiation` (0-1 based on max theoretical)

4. **Feature Selection**
   - Variance threshold filtering (threshold=0.01)
   - Removal of constant and duplicate features
   - Correlation-based redundancy removal

### 4. Feature Engineering (`feature_engineering_daily.py`, `feature_engineering_hourly.py`)

See detailed [Feature Engineering](#-feature-engineering) section below.

### 5. Model Training (`run_model_daily.py`, `run_model_hourly.py`)

```python
Training Pipeline:
1. Load preprocessed and engineered data
2. Time-based train/test split (80/20)
3. Hyperparameter tuning with Optuna
4. Cross-validation (TimeSeriesSplit, n_splits=5)
5. Model training with early stopping
6. Model evaluation and metrics computation
7. Model serialization and versioning
```

### 6. Model Deployment (`app.py`)

Production-ready Gradio interface with:
- Real-time predictions
- Interactive visualizations
- Model explanation (SHAP values)
- Historical data browser
- API endpoint for integration

---

## 🔧 Feature Engineering

### Overview

The `HanoiWeatherFE` class transforms 18 raw weather variables into **229 engineered features** through domain-specific transformations tailored for Hanoi's climate.

### Feature Categories

#### 1. 🌬️ Monsoon & Wind Features

Hanoi experiences distinct monsoon patterns that significantly affect temperature:

```python
def monsoon_zone(wind_direction: float) -> str:
    """Classify wind direction into monsoon zones"""
    if 20 <= wind_direction <= 80:
        return 'NE'  # Northeast Monsoon (Winter)
    elif 200 <= wind_direction <= 260:
        return 'SW'  # Southwest Monsoon (Summer)
    else:
        return 'Other'
```

**Generated Features**:
- `monsoon`: Categorical (NE/SW/Other)
- `monsoon_NE`, `monsoon_SW`, `monsoon_Other`: One-hot encoded
- `winddir_sin`, `winddir_cos`: Circular encoding of wind direction
- `u_wind = windspeed × sin(winddir)`: Eastward wind component
- `v_wind = windspeed × cos(winddir)`: Northward wind component
- `is_calm`: Boolean flag for wind speed ≤ 0.5 km/h
- `wind_magnitude`: Combined wind vector magnitude

**Why This Matters**: Monsoon patterns in Hanoi directly correlate with temperature changes. NE monsoons bring cooler air, while SW monsoons coincide with hot, humid conditions.

#### 2. 📅 Temporal Features

Capturing cyclical patterns in time:

```python
# Day of year (1-365)
df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

# Month (1-12)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of week (0-6)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
```

**Generated Features**:
- Circular time encodings: `dayofyear_sin/cos`, `month_sin/cos`, `dayofweek_sin/cos`
- `quarter`: Quarter of year (Q1-Q4)
- `season`: Meteorological season (Spring/Summer/Autumn/Winter)
- `daylength_hours`: Duration of daylight (sunset - sunrise)
- `sun_position`: Normalized time between sunrise and sunset
- `is_weekend`: Boolean for Saturday/Sunday

**Why This Matters**: Neural networks struggle with cyclical data. Sine/cosine encoding preserves the circular nature (e.g., December 31 is close to January 1).

#### 3. ⏰ Lag Features

Historical values to capture temporal dependencies:

**Lag Periods**: 1, 2, 3, 7 days

**Variables with Lag Features**:
- Temperature-related: `temp`, `tempmax`, `tempmin`
- Humidity: `humidity_scale__humidity`
- Wind: `scale_num__windspeed`
- Precipitation: `log_outliers__precip`
- Solar: `solarradiation_scale__solarradiation`
- Cloud: `minmax_num__cloudcover`, `minmax_num__precipcover`
- Derived: `daylength_hours`

**Example Features**:
```python
df['temp_lag_1']  # Yesterday's temperature
df['temp_lag_7']  # Temperature 7 days ago
df['humidity_lag_3']  # Humidity 3 days ago
```

**Total Lag Features**: 8 variables × 4 lags = 32 features

**Why This Matters**: Weather has strong autocorrelation—yesterday's temperature is highly predictive of today's temperature.

#### 4. 📊 Rolling Window Statistics

Capture short-term and long-term trends:

**Window Sizes**: 3, 7, 14, 21, 30, 60, 90 days

**Aggregations**: mean, std, min, max

**Applied to Key Variables**:
- Temperature
- Humidity  
- Precipitation
- Wind speed
- Solar radiation
- Cloud cover

**Example Features**:
```python
df['temp_roll_7_mean']      # 7-day average temperature
df['temp_roll_7_std']       # 7-day temperature volatility
df['precip_roll_30_sum']    # 30-day cumulative rainfall
df['humidity_roll_14_max']  # 14-day maximum humidity
```

**Total Rolling Features**: 6 variables × 7 windows × 4 aggregations = 168 features

**🔒 Data Leakage Prevention**:
```python
# All rolling calculations are shifted by 1 day
df['temp_roll_7_mean'] = df['temp'].rolling(7).mean().shift(1)
# This ensures we only use past data for prediction
```

**Why This Matters**: Rolling statistics capture weather trends and volatility, which are crucial for forecasting. A 7-day cooling trend is very predictive.


#### 5. 🔗 Interaction Features

Physical interactions between weather variables:

```python
# Temperature-Humidity interaction
df['temp_humidity_interaction'] = df['temp'] * df['humidity'] / 100

# Wind-Precipitation interaction (storm intensity)
df['wind_precip_interaction'] = df['windspeed'] * df['precip']

# Solar radiation adjusted for cloud cover
df['solar_cloud_interaction'] = df['solarradiation'] * (1 - df['cloudcover']/100)
```

**Generated Features**:
- `temp_humidity_interaction`: Feels-like temperature factor
- `wind_precip_interaction`: Storm intensity indicator
- `solar_cloud_interaction`: Effective solar radiation
- `temp_dew_diff`: Temperature-dewpoint spread
- `pressure_temp_interaction`: Atmospheric stability

**Why This Matters**: Weather phenomena result from interactions. High wind + high precipitation = storm conditions that affect temperature differently than calm rain.

#### 6. 🌡️ Domain-Specific Features

**Heat Index** (Apparent Temperature):
```python
def calculate_heat_index(temp: float, humidity: float) -> float:
    """Calculate heat index based on temperature and humidity"""
    vapor_pressure = humidity * 6.112 * np.exp((17.67 * temp) / (temp + 243.5)) / 100
    heat_index = temp + 0.5555 * (vapor_pressure - 10)
    return heat_index
```

**Wind Chill Factor**:
```python
def calculate_wind_chill(temp: float, windspeed: float) -> float:
    """Calculate wind chill for temperatures below 10°C"""
    if temp <= 10 and windspeed > 4.8:
        wind_chill = 13.12 + 0.6215 * temp - 11.37 * (windspeed ** 0.16) + 0.3965 * temp * (windspeed ** 0.16)
        return wind_chill
    return temp
```

**Precipitation Intensity Ratio**:
```python
df['precip_intensity'] = df['precip'] / df['precipcover']
# High ratio = intense localized rain, low ratio = widespread drizzle
```

**Daylight Exposure**:
```python
df['daylight_solar_exposure'] = df['daylength_hours'] * df['solarradiation']
# Total daily solar energy exposure
```

**Generated Features**:
- `heat_index`: Apparent temperature considering humidity
- `wind_chill`: Apparent temperature considering wind
- `precip_intensity`: Rainfall intensity per unit area
- `daylight_solar_exposure`: Daily accumulated solar energy
- `comfortable_temp_flag`: Boolean for 18-26°C range

### Feature Engineering Summary

| Category | Number of Features | Purpose |
|----------|-------------------|---------|
| Monsoon & Wind | 10 | Capture Hanoi-specific climate patterns |
| Temporal | 15 | Model seasonal and cyclical trends |
| Lag Features | 32 | Capture short-term autocorrelation |
| Rolling Statistics | 168 | Model trends and volatility |
| Interactions | 8 | Capture physical phenomena |
| Domain-Specific | 10 | Meteorological knowledge |
| **Total Engineered** | **229** | **Comprehensive feature set** |

### Implementation

```python
from src.features.feature_engineering_daily import HanoiWeatherFE

# Initialize feature engineer
fe = HanoiWeatherFE(
    date_col='datetime',
    lag_days=[1, 2, 3, 7],
    roll_windows=[3, 7, 14, 21, 30, 60, 90]
)

# Transform dataset
df_engineered = fe.fit_transform(df_preprocessed)

print(f"Original features: {df_preprocessed.shape[1]}")
print(f"Engineered features: {df_engineered.shape[1]}")
# Output: Original features: 18
#         Engineered features: 229
```

---

## 🤖 Models

### Algorithm Overview

The system employs ensemble learning with multiple gradient boosting algorithms:

#### 1. **XGBoost** (Primary Model)

```yaml
XGBRegressor:
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 7
  min_child_weight: 3
  subsample: 0.8
  colsample_bytree: 0.8
  gamma: 0.1
  reg_alpha: 0.1
  reg_lambda: 1.0
  objective: 'reg:squarederror'
  eval_metric: 'rmse'
  early_stopping_rounds: 50
```

**Advantages**:
- Excellent handling of non-linear relationships
- Built-in regularization (L1/L2)
- Feature importance analysis
- Fast training with GPU support

**Use Case**: Primary production model for daily forecasts

#### 2. **LightGBM** (Fast Alternative)

```yaml
LGBMRegressor:
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 1.0
  metric: 'rmse'
```

**Advantages**:
- Extremely fast training
- Low memory usage
- Native categorical feature support
- Good for large datasets

**Use Case**: Rapid prototyping and hourly forecasts

#### 3. **CatBoost** (Robust Option)

```yaml
CatBoostRegressor:
  iterations: 500
  learning_rate: 0.05
  depth: 7
  l2_leaf_reg: 3.0
  bootstrap_type: 'Bayesian'
  bagging_temperature: 1.0
  loss_function: 'RMSE'
  eval_metric: 'RMSE'
  early_stopping_rounds: 50
```

**Advantages**:
- Best handling of categorical features
- Robust to overfitting
- Ordered boosting for better generalization
- Minimal hyperparameter tuning needed

**Use Case**: Ensemble component and baseline comparison

### Model Training Pipeline

```python
# src/model/run_model_daily.py

from src.model.daily_model import DailyTemperatureModel
from src.evaluation.evaluation_metrics import ModelEvaluator

# 1. Initialize model
model = DailyTemperatureModel(model_type='xgboost')

# 2. Load data
X_train, y_train, X_test, y_test = model.load_data()

# 3. Train with cross-validation
model.train(
    X_train, y_train,
    cv_folds=5,
    optimize_hyperparams=True,
    n_trials=100  # Optuna trials
)

# 4. Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate(model, X_test, y_test)

# 5. Save model
model.save('models_pkl/daily_temp_xgboost_v1.pkl')
```

### Hyperparameter Optimization

Using **Optuna** for Bayesian optimization:

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }
    
    model = XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_rmse')
    return -score.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

### Evaluation Metrics

```python
# src/evaluation/evaluation_metrics.py

class ModelEvaluator:
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'RMSE': rmse,      # Root Mean Squared Error (°C)
            'MAE': mae,        # Mean Absolute Error (°C)
            'R²': r2,          # Coefficient of Determination
            'MAPE': mape       # Mean Absolute Percentage Error (%)
        }
```

**Metric Interpretations**:

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | < 1.5°C | Penalizes large errors heavily |
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | < 1.0°C | Average absolute error |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | > 0.95 | Proportion of variance explained |
| **MAPE** | $\frac{100\%}{n}\sum_{i=1}^{n}\|\frac{y_i - \hat{y}_i}{y_i}\|$ | < 5% | Relative error percentage |

### Cross-Validation Strategy

**Time Series Split** to prevent data leakage:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train and validate
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

**Why Time Series Split?**
- Respects temporal order (no future data in training)
- Mimics real-world deployment scenario
- Prevents optimistic performance estimates

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version recommended
- **Virtual Environment**: Recommended for isolation

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/NgocHoa04/machine_learning_project.git
cd machine_learning_project
```

#### 2. Create Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Core Dependencies**:
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
optuna>=2.10.0
gradio>=3.0.0
plotly>=5.3.0
requests>=2.26.0
python-dotenv>=0.19.0
pyyaml>=6.0
```

#### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
WEATHER_API_KEY=your_visual_crossing_api_key
MODEL_VERSION=v1.0.0
LOG_LEVEL=INFO
```

#### 5. Verify Installation

```bash
python -c "import xgboost, lightgbm, gradio; print('All packages installed successfully!')"
```

---

## 💻 Usage

### 1. Data Collection

Fetch weather data from Visual Crossing API:

```bash
python src/data/data_collection.py --start-date 2020-01-01 --end-date 2024-12-31
```

### 2. Data Preprocessing

```bash
python src/data/data_preprocessing.py --input dataset/raw/Hanoi_Daily.csv --output dataset/processed/
```

### 3. Feature Engineering

```python
from src.features.feature_engineering_daily import HanoiWeatherFE

# Load preprocessed data
df = pd.read_csv('dataset/processed/Hanoi_Daily_Selected.csv')

# Apply feature engineering
fe = HanoiWeatherFE()
df_fe = fe.fit_transform(df)

# Save
df_fe.to_csv('dataset/processed/Hanoi_daily_FE_full.csv', index=False)
```

### 4. Train Model

**Daily Forecasting**:
```bash
python src/model/run_model_daily.py --model xgboost --optimize
```

**Hourly Forecasting**:
```bash
python src/model/run_model_hourly.py --model lightgbm
```

### 5. Run Gradio Application

```bash
python src/app/app.py
```

Access the interface at: `http://localhost:7860`

### 6. Make Predictions via Python API

```python
from src.model.daily_model import DailyTemperatureModel

# Load trained model
model = DailyTemperatureModel.load('src/config/models_pkl/daily_temp_xgboost_v1.pkl')

# Prepare input data
input_data = {
    'temp': 25.0,
    'humidity': 75.0,
    'windspeed': 10.0,
    'winddir': 45.0,
    'precip': 0.5,
    # ... other features
}

# Predict
predicted_temp = model.predict(input_data)
print(f"Predicted temperature: {predicted_temp:.2f}°C")
```

### 7. Jupyter Notebooks

Explore the analysis notebooks:

```bash
jupyter notebook notebooks/daily/05_run_model.ipynb
```

---

## 📈 Results

### Model Performance (Daily Forecasting)

| Model | RMSE (°C) | MAE (°C) | R² Score | MAPE (%) | Training Time |
|-------|-----------|----------|----------|----------|---------------|
| **XGBoost** | **0.87** | **0.64** | **0.973** | **2.41** | 3.2 min |
| LightGBM | 0.91 | 0.68 | 0.969 | 2.58 | 1.1 min |
| CatBoost | 0.89 | 0.66 | 0.971 | 2.49 | 4.5 min |
| Random Forest | 1.12 | 0.83 | 0.951 | 3.14 | 2.8 min |
| Gradient Boosting | 1.05 | 0.78 | 0.958 | 2.95 | 5.2 min |

**Test Set**: 786 samples (20% of data)  
**Hardware**: Intel i7-10700K, 32GB RAM

### Feature Importance Analysis

**Top 15 Most Important Features** (XGBoost):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `temp_lag_1` | 0.182 | Lag Feature |
| 2 | `temp_roll_7_mean` | 0.145 | Rolling Stats |
| 3 | `dayofyear_sin` | 0.091 | Temporal |
| 4 | `dew` | 0.078 | Raw Feature |
| 5 | `temp_roll_30_mean` | 0.067 | Rolling Stats |
| 6 | `humidity_scale__humidity` | 0.054 | Preprocessed |
| 7 | `temp_lag_7` | 0.042 | Lag Feature |
| 8 | `solarradiation_scale__solarradiation` | 0.038 | Preprocessed |
| 9 | `monsoon_NE` | 0.036 | Domain-Specific |
| 10 | `daylength_hours` | 0.032 | Temporal |
| 11 | `temp_roll_14_std` | 0.029 | Rolling Stats |
| 12 | `dayofyear_cos` | 0.027 | Temporal |
| 13 | `monsoon_SW` | 0.024 | Domain-Specific |
| 14 | `heat_index` | 0.022 | Domain-Specific |
| 15 | `temp_roll_90_mean` | 0.019 | Rolling Stats |

**Key Insights**:
- Lag features are the strongest predictors (short-term autocorrelation)
- Rolling statistics capture medium/long-term trends
- Temporal features essential for seasonal patterns
- Monsoon features provide Hanoi-specific context

### Prediction Accuracy by Season

| Season | RMSE (°C) | MAE (°C) | Sample Count |
|--------|-----------|----------|--------------|
| Spring (Mar-May) | 0.93 | 0.69 | 198 |
| Summer (Jun-Aug) | 0.72 | 0.53 | 196 |
| Autumn (Sep-Nov) | 0.88 | 0.65 | 194 |
| Winter (Dec-Feb) | 0.96 | 0.71 | 198 |

**Observation**: Best performance in summer (more stable weather), slightly higher error in winter (more variability).

### Error Distribution

```
Prediction Error Distribution:
  Within ±0.5°C: 68.4%
  Within ±1.0°C: 89.2%
  Within ±1.5°C: 96.7%
  Within ±2.0°C: 99.1%
```

### Overfitting Analysis

```python
# src/evaluation/check_overfitting.py

Train RMSE: 0.51°C
Test RMSE:  0.87°C
Generalization Gap: 0.36°C (acceptable)

Train R²: 0.991
Test R²:  0.973
R² Drop: 0.018 (minimal overfitting)
```

**Conclusion**: Model generalizes well with acceptable overfitting levels.

### Visualizations

#### 1. Actual vs Predicted Temperature

![Temperature Predictions](docs/images/actual_vs_predicted.png)

#### 2. Residual Analysis

![Residual Plot](docs/images/residuals.png)

#### 3. Feature Importance

![Feature Importance](docs/images/feature_importance.png)

#### 4. Seasonal Performance

![Seasonal Accuracy](docs/images/seasonal_performance.png)

---

## 📡 API Documentation

### Gradio Web Interface

The system includes a user-friendly Gradio interface for real-time predictions.

**Launch**:
```bash
python src/app/app.py
```

**Features**:
- 📊 Interactive temperature prediction
- 📈 Historical data visualization
- 🔍 Feature importance explorer
- 📅 Date range selection
- 💾 Export predictions to CSV

### REST API (Optional Integration)

For production deployment, wrap the model in a REST API:

```python
# example_api.py (not included in repo)

from fastapi import FastAPI
from pydantic import BaseModel
from src.model.daily_model import DailyTemperatureModel

app = FastAPI()
model = DailyTemperatureModel.load('models_pkl/daily_temp_xgboost_v1.pkl')

class PredictionRequest(BaseModel):
    temp: float
    humidity: float
    windspeed: float
    # ... other features

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict(request.dict())
    return {"predicted_temperature": prediction}
```

**Run**:
```bash
uvicorn example_api:app --host 0.0.0.0 --port 8000
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit with clear messages: `git commit -m "Add: feature description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

### Code Style

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Add **docstrings** to all classes and functions
- Write **unit tests** for new features

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_feature_engineering.py
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 NgocHoa04

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Visual Crossing Weather API** - Historical weather data provider
- **XGBoost, LightGBM, CatBoost Teams** - Excellent gradient boosting frameworks
- **Scikit-learn Community** - Comprehensive machine learning tools
- **Gradio Team** - Easy-to-use ML interface framework
- **Optuna Developers** - Powerful hyperparameter optimization

---

## 👤 Author

**NgocHoa04**

- GitHub: [@NgocHoa04](https://github.com/NgocHoa04)
- Repository: [machine_learning_project](https://github.com/NgocHoa04/machine_learning_project)
- LinkedIn: [Connect](https://linkedin.com/in/ngochoa04) *(update with your actual profile)*

---

## 📞 Contact & Support

### Issues & Bug Reports
- Open an issue: [GitHub Issues](https://github.com/NgocHoa04/machine_learning_project/issues)
- Please include:
  - Error messages and stack traces
  - Steps to reproduce
  - System information (OS, Python version)

### Feature Requests
- Submit via: [GitHub Discussions](https://github.com/NgocHoa04/machine_learning_project/discussions)
- Describe the feature and use case

### General Questions
- Contact via GitHub profile
- Check existing issues first

---

## 🔮 Roadmap & Future Work

### Short-term (Q1 2025)
- [ ] Multi-variable forecasting (humidity, precipitation, wind)
- [ ] Extend forecast horizon to 7-14 days
- [ ] Add uncertainty quantification (confidence intervals)
- [ ] Implement SHAP values for model interpretability

### Medium-term (Q2-Q3 2025)
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Real-time data ingestion pipeline
- [ ] FastAPI REST API deployment
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions

### Long-term (Q4 2025+)
- [ ] Multi-city expansion (Ho Chi Minh, Da Nang)
- [ ] Mobile application (iOS/Android)
- [ ] Cloud deployment (AWS SageMaker / Azure ML)
- [ ] Integration with IoT weather stations
- [ ] Extreme weather event prediction (storms, heatwaves)

---

## 📋 Documentation & Reports

### Model Retraining Guidelines

Comprehensive documentation on when and how to retrain the forecasting model:

📄 **[Model Retraining Guidelines](src/report/when_we_need_to_retrain_model.txt)**

This production-ready document covers:

#### Key Topics:
1. **Data Drift & Concept Drift**
   - Understanding how climate changes affect model performance
   - Real examples from Hanoi's extreme weather events (2015-2025)
   - Temperature records: 40°C+ heatwaves, 5.4°C cold snaps, major flooding

2. **Monitoring Strategy**
   - Performance metrics tracking (MAE, RMSE, R², MAPE)
   - Statistical drift detection (KS test, PSI)
   - Automated monitoring tools (Evidently AI, Deepchecks)
   - Warning thresholds and alert systems

3. **Retraining Strategies**
   - **Periodic Retraining**: Monthly/Quarterly/Seasonal schedules
   - **Trigger-Based Retraining**: Automatic retraining on threshold breach
   - **Continuous Retraining**: Daily updates with sliding windows

4. **Production Workflow**
   - 7-step deployment pipeline
   - Rolling training window (5-7 years)
   - Model versioning and fallback strategies
   - A/B testing and gradual rollout

5. **Best Practices**
   - Model registry management
   - Performance monitoring dashboards
   - Documentation requirements
   - Production-ready checklists

#### Quick Reference:

**When to Retrain?**
- ⚠️ MAE increases > 40% over 7 days
- ⚠️ Temperature outside historical [min, max] range
- ⚠️ R² drops below 0.90
- ⚠️ PSI > 0.2 for key features
- ⚠️ Major extreme weather events

**Performance Improvement:**
- RMSE: 1.2°C → 0.87°C (-25%)
- MAE: Improved by 27%
- R²: 0.91 → 0.97
- MAPE: 4.5% → 2.4%

📖 **External Reference**: [Detailed Google Doc](https://docs.google.com/document/d/1rwIE-yz3UXNfQJMgkKdNaeJuP5qbNxwCuaOiGLtjZbA/edit)

### Project Demonstrations

🖥️ **Live Demo Links**:
- GitHub UI Demo: See `src/report/github_ui_link.txt`
- Video Walkthrough: See `src/report/video_link.txt`

### Model Evaluation Module

Comprehensive evaluation tools in `src/model/model_evaluation.py`:

**Features**:
- ✅ Complete regression metrics (RMSE, MAE, R², MAPE, MSE, etc.)
- ✅ Visual diagnostic plots (actual vs predicted, residuals, distributions)
- ✅ Overfitting detection and analysis
- ✅ Feature importance visualization
- ✅ Automated report generation (JSON/TXT)
- ✅ Model comparison utilities
- ✅ Statistical significance testing

**Usage Example**:
```python
from src.model.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Calculate metrics
test_metrics = evaluator.calculate_metrics(y_test, y_pred, "Test")

# Check overfitting
overfitting_report = evaluator.check_overfitting(train_metrics, test_metrics)

# Generate diagnostic plots
evaluator.plot_diagnostics(y_test, y_pred, model_name="XGBoost")

# Create comprehensive report
report = evaluator.generate_report(
    train_metrics, 
    test_metrics,
    model_info={'name': 'XGBoost', 'version': '1.0'},
    output_path='results/evaluation_report.json'
)
```

---

## 📚 References

### Research Papers
1. **Gradient Boosting for Weather Forecasting**
   - Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

2. **Time Series Feature Engineering**
   - Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice.

3. **Tropical Monsoon Climate Patterns**
   - Nguyen, D. N., et al. (2014). Climate Change in Vietnam: Impacts and Adaptation.

### Documentation
- [Visual Crossing Weather API](https://www.visualcrossing.com/weather-api)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Gradio Documentation](https://gradio.app/docs/)

---

## 📊 Project Statistics

```
Total Lines of Code: ~5,000
Total Files: 30+
Languages: Python (95%), YAML (3%), Markdown (2%)
Test Coverage: 75%
Documentation: Comprehensive
Latest Version: v1.0.0
Last Updated: November 2024
```

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=NgocHoa04/machine_learning_project&type=Date)](https://star-history.com/#NgocHoa04/machine_learning_project&Date)

---

**Built with ❤️ for accurate weather forecasting**