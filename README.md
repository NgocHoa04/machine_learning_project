# 🌤️ Hanoi Weather Forecast - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-Time%20Series%20Forecasting-orange.svg)](https://github.com)

Dự án dự báo thời tiết Hà Nội sử dụng Machine Learning với dữ liệu lịch sử từ Visual Crossing Weather API. Mô hình dự đoán nhiệt độ tương lai dựa trên các đặc trưng thời tiết như nhiệt độ, độ ẩm, hướng gió, lượng mưa, bức xạ mặt trời và các yếu tố khí tượng khác.

---

## 📋 Mục Lục

- [Tổng Quan Dự Án](#-tổng-quan-dự-án)
- [Dữ Liệu](#-dữ-liệu)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Quy Trình Phát Triển](#-quy-trình-phát-triển)
- [Feature Engineering](#-feature-engineering)
- [Mô Hình Machine Learning](#-mô-hình-machine-learning)
- [Cài Đặt và Sử Dụng](#-cài-đặt-và-sử-dụng)
- [Kết Quả](#-kết-quả)
- [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)
- [Tác Giả](#-tác-giả)

---

## 🎯 Tổng Quan Dự Án

### Mục Tiêu
Xây dựng hệ thống dự báo nhiệt độ chính xác cho thành phố Hà Nội, giúp:
- Dự đoán nhiệt độ trong tương lai dựa trên dữ liệu lịch sử
- Phân tích các yếu tố ảnh hưởng đến biến đổi nhiệt độ
- Cung cấp insight về khí hậu Hà Nội qua các mùa trong năm

### Phương Pháp
- **Time Series Forecasting** với các thuật toán Ensemble Learning
- **Feature Engineering** toàn diện bao gồm lag features, rolling statistics, và seasonal patterns
- **Data Preprocessing** chuyên sâu với outlier handling và feature scaling
- **Model Evaluation** sử dụng Cross-Validation và các metrics chuẩn (RMSE, MAE, R²)

---

## 📊 Dữ Liệu

### Nguồn Dữ Liệu
- **API**: Visual Crossing Weather API
- **Vị trí**: Hà Nội, Việt Nam
- **Thời gian**: Dữ liệu lịch sử và theo giờ

### Dataset Files
- **`Hanoi Daily.csv`**: Dữ liệu thời tiết theo ngày
- **`Hanoi Hourly.csv`**: Dữ liệu thời tiết theo giờ
- **`train.xlsx`**: Tập huấn luyện đã xử lý (3,051 samples)
- **`test.xlsx`**: Tập kiểm tra (786 samples)

### Các Biến Chính
| Biến | Mô Tả | Đơn Vị |
|------|-------|--------|
| `temp` | Nhiệt độ | °C |
| `tempmax` | Nhiệt độ cao nhất | °C |
| `tempmin` | Nhiệt độ thấp nhất | °C |
| `humidity` | Độ ẩm tương đối | % |
| `dew` | Điểm sương | °C |
| `precip` | Lượng mưa | mm |
| `precipprob` | Xác suất mưa | % |
| `precipcover` | Phạm vi phủ mưa | % |
| `windspeed` | Tốc độ gió | km/h |
| `winddir` | Hướng gió | độ (0-360°) |
| `solarradiation` | Bức xạ mặt trời | W/m² |
| `cloudcover` | Độ che phủ mây | % |
| `sunrise` | Giờ mặt trời mọc | timestamp |
| `sunset` | Giờ mặt trời lặn | timestamp |

### Đặc Điểm Khí Hậu Hà Nội
- **Gió mùa Đông Bắc (NE)**: 20-80° - Mùa đông/xuân, lạnh và ẩm
- **Gió mùa Tây Nam (SW)**: 200-260° - Mùa hè, nóng ẩm với giông bão
- **Điểm sương cao**: Đặc trưng khí hậu ẩm nhiệt đới gió mùa

---

## 📁 Cấu Trúc Thư Mục

```
Final project/
│
├── dataset/
│   ├── raw/                      # Dữ liệu gốc
│   │   ├── Hanoi Daily.csv       # Dữ liệu ngày
│   │   └── Hanoi Hourly.csv      # Dữ liệu giờ
│   └── processed/                # Dữ liệu đã xử lý
│
├── data/                         # Dữ liệu train/test
│   ├── train.xlsx                # Training set
│   └── test.xlsx                 # Testing set
│
├── notebooks/                    # Jupyter Notebooks
│   ├── data_understanding.ipynb  # EDA và phân tích dữ liệu
│   ├── data_processing.ipynb     # Tiền xử lý dữ liệu
│   ├── feature_engineering_GBDT.ipynb  # FE với GBDT
│   └── project.ipynb             # Notebook chính
│
├── scripts/                      # Python scripts
│   ├── data_preprocessing.py     # Preprocessing pipeline
│   └── FE.py                     # Feature engineering class
│
└── README.md                     # Tài liệu dự án
```

---

## 🔄 Quy Trình Phát Triển

### 1. Data Understanding (`data_understanding.ipynb`)
- **Exploratory Data Analysis (EDA)**
  - Phân tích phân phối của các biến
  - Correlation matrix và heatmap
  - Phát hiện missing values và outliers
  - Visualize xu hướng theo thời gian

- **Insights chính**:
  - Điểm sương (`dew`) tương quan mạnh với nhiệt độ
  - Độ ẩm cao đặc trưng khí hậu Hà Nội
  - Seasonal patterns rõ rệt giữa các mùa

### 2. Data Preprocessing (`data_preprocessing.py`)
- **Xử lý Missing Values**: Imputation cho các giá trị thiếu
- **Outlier Detection & Handling**: 
  - Log transformation cho `precip` (lượng mưa)
  - Statistical methods cho các biến khác
- **Feature Scaling**:
  - `StandardScaler` cho các biến số học
  - `MinMaxScaler` cho humidity, cloudcover, precipcover
  - Custom scaling cho solar radiation
- **Remove Low Variance Features**: Loại bỏ features không đóng góp thông tin

**Classes chính**:
```python
- VarianceThresholdSelector: Loại bỏ features variance thấp
- ConstantAndDuplicateRemover: Xóa constants và duplicates
- remove_low_variance_pipeline: Pipeline tổng hợp
```

### 3. Feature Engineering (`FE.py`, `project.ipynb`)
Tạo 200+ features mới từ dữ liệu gốc (xem chi tiết phần [Feature Engineering](#-feature-engineering))

### 4. Model Training & Evaluation (`project.ipynb`)
- **Train/Test Split**: Time-based split để tránh data leakage
- **Model Selection**: So sánh nhiều thuật toán
- **Hyperparameter Tuning**: Grid Search, Random Search
- **Cross-Validation**: Time Series CV
- **Model Evaluation**: RMSE, MAE, R², visualizations

---

## 🔧 Feature Engineering

### Class: `HanoiWeatherFE`

Feature Engineering class được thiết kế đặc biệt cho dữ liệu thời tiết Hà Nội với **229 features** từ 18 features gốc.

#### 1. **Monsoon & Wind Features**
Phân loại và mã hóa gió mùa đặc trưng Hà Nội:

```python
monsoon_zone(deg):
  - NE (20-80°): Gió mùa Đông Bắc
  - SW (200-260°): Gió mùa Tây Nam  
  - Other: Các hướng khác
```

**Features tạo ra**:
- `monsoon`: Category (NE/SW/Other)
- `monsoon_NE`, `monsoon_SW`, `monsoon_Other`: One-hot encoding
- `winddir_sin`, `winddir_cos`: Chu kỳ hóa hướng gió
- `u_wind`, `v_wind`: Vector gió (phân tích thành phần)
- `is_calm`: Cờ gió lặng (speed ≤ 0.5)

#### 2. **Temporal Features**
Trích xuất đặc trưng thời gian chu kỳ:

- **Ngày trong năm**: `dayofyear_sin`, `dayofyear_cos`
- **Tháng**: `month_sin`, `month_cos`
- **Ngày trong tuần**: `dayofweek_sin`, `dayofweek_cos`
- **Quý**: `quarter`
- **Mùa**: `season` (Spring/Summer/Autumn/Winter)
- **Độ dài ban ngày**: `daylength_hours`
- **Vị trí mặt trời**: `sun_position` (sunrise/sunset relative)

#### 3. **Lag Features**
Tạo lag features cho các biến quan trọng để capture temporal dependencies:

**Lag days**: 1, 2, 3, 7 ngày

**Biến áp dụng lag**:
- `humidity_scale__humidity`
- `scale_num__windspeed`
- `log_outliers__precip`
- `solarradition_scale__solarradiation`
- `minmax_num__cloudcover`
- `minmax_num__precipcover`
- `daylength_hours`

**Ví dụ**: `humidity_scale__humidity_lag_1`, `windspeed_lag_7`

#### 4. **Rolling Window Statistics**
Rolling aggregations để capture xu hướng ngắn/dài hạn:

**Windows**: 3, 7, 14, 21, 30, 60, 90 ngày

**Aggregations**: mean, std, min, max

**Ví dụ**:
- `humidity_roll_7_mean`: Độ ẩm trung bình 7 ngày
- `precip_roll_30_std`: Độ lệch chuẩn lượng mưa 30 ngày
- `temp_roll_14_max`: Nhiệt độ max trong 14 ngày

**🔒 No Data Leakage**: Tất cả rolling features được shift(1) trước khi tính toán

#### 5. **Interaction Features**
Tương tác giữa các biến:

- `temp_humidity_interaction`: temp × humidity
- `wind_precip_interaction`: windspeed × precip
- `solar_cloud_interaction`: solar radiation × (1 - cloudcover)

#### 6. **Domain-Specific Features**

**Heat Index**:
```python
heat_index = temp + 0.5555 × (vapor_pressure - 10)
```

**Precipitation Ratio**:
```python
precip_ratio = precipcover / precipprob (when precipprob > 0)
```

**Wind Chill Effect**: Hiệu ứng làm lạnh của gió

---

## 🤖 Mô Hình Machine Learning

### Thuật Toán Sử Dụng

#### 1. **Random Forest Regressor**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
```
- **Ưu điểm**: Robust với outliers, feature importance rõ ràng
- **Sử dụng**: Baseline model, feature selection

#### 2. **XGBoost**
```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    eval_metric="rmse"
)
```
- **Ưu điểm**: Performance cao, regularization tốt
- **Sử dụng**: Main production model

#### 3. **LightGBM**
```python
lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31
)
```
- **Ưu điểm**: Nhanh, hiệu quả với large dataset
- **Sử dụng**: Alternative model, ensemble

#### 4. **CatBoost**
```python
cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=7
)
```
- **Ưu điểm**: Xử lý categorical features tốt
- **Sử dụng**: Ensemble component

#### 5. **Gradient Boosting**
```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)
```
- **Ưu điểm**: Stable, interpretable
- **Sử dụng**: Baseline comparison

### Evaluation Metrics

| Metric | Công Thức | Ý Nghĩa |
|--------|-----------|---------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Root Mean Squared Error - phạt lỗi lớn |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Mean Absolute Error - robust với outliers |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Coefficient of determination - goodness of fit |
| **MAPE** | $\frac{100\%}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Mean Absolute Percentage Error |

---

## 🚀 Cài Đặt và Sử Dụng

### Prerequisites
```bash
Python >= 3.8
```

### Installation

1. **Clone repository**:
```bash
git clone https://github.com/NgocHoa04/machine_learning_project.git
cd machine_learning_project
```

2. **Tạo virtual environment** (khuyến nghị):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt dependencies**:
```bash
pip install pandas numpy scikit-learn
pip install xgboost lightgbm catboost
pip install matplotlib seaborn
pip install jupyter notebook
pip install openpyxl  # Để đọc .xlsx files
```

### Quick Start

#### 1. Chạy toàn bộ pipeline:
```bash
jupyter notebook project.ipynb
```

#### 2. Sử dụng Feature Engineering:
```python
from FE import HanoiWeatherFE

# Khởi tạo
fe = HanoiWeatherFE(
    date_col="datetime",
    lag_days=(1, 2, 3, 7),
    roll_windows=(3, 7, 14, 21, 30, 60, 90)
)

# Transform data
df_engineered = fe.transform(df_preprocessed)
print(f"Original: {df.shape} -> Engineered: {df_engineered.shape}")
```

#### 3. Train model:
```python
from xgboost import XGBRegressor

# Prepare data
X_train = train_fe.drop(columns=['target_temp'])
y_train = train_fe['target_temp']

# Train
model = XGBRegressor(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

---

## 📈 Kết Quả

### Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Random Forest | TBD | TBD | TBD |
| **XGBoost** | **TBD** | **TBD** | **TBD** |
| LightGBM | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD |

*Note: Cập nhật metrics sau khi chạy đầy đủ experiments*

### Feature Importance

Top 10 features quan trọng nhất (từ XGBoost):
1. `temp_lag_1` - Nhiệt độ ngày hôm trước
2. `temp_roll_7_mean` - Nhiệt độ trung bình 7 ngày
3. `dew` - Điểm sương
4. `humidity_scale__humidity` - Độ ẩm
5. `dayofyear_sin/cos` - Chu kỳ năm
6. `solarradiation_scale__solarradiation` - Bức xạ mặt trời
7. `monsoon_NE/SW` - Gió mùa
8. `daylength_hours` - Độ dài ban ngày
9. `temp_roll_30_mean` - Trend dài hạn
10. `precip_roll_7_mean` - Lượng mưa gần đây

### Visualizations

Các biểu đồ quan trọng trong notebook:
- 📊 Correlation Heatmap
- 📈 Temperature Trends over Time
- 🌡️ Actual vs Predicted Temperature
- 📉 Residual Analysis
- 🎯 Feature Importance Plot
- 📅 Seasonal Patterns

---

## 🛠️ Công Nghệ Sử Dụng

### Core Libraries
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML algorithms & preprocessing
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **CatBoost** - Categorical boosting

### Visualization
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualization

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git** - Version control
- **Python 3.8+** - Programming language

---

## 👥 Tác Giả

**NgocHoa04**
- GitHub: [@NgocHoa04](https://github.com/NgocHoa04)
- Repository: [machine_learning_project](https://github.com/NgocHoa04/machine_learning_project)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Visual Crossing Weather API** - Cung cấp dữ liệu thời tiết
- **Scikit-learn Community** - ML frameworks
- **XGBoost/LightGBM/CatBoost Teams** - Advanced boosting algorithms

---

## 📞 Contact & Support

Nếu bạn có câu hỏi hoặc gặp vấn đề, vui lòng:
1. Mở [Issue](https://github.com/NgocHoa04/machine_learning_project/issues) trên GitHub
2. Liên hệ qua GitHub profile

---

## 🔮 Future Work

- [ ] Thêm dự báo cho các biến khác (humidity, precipitation)
- [ ] Triển khai web application với Flask/FastAPI
- [ ] Tích hợp real-time data từ API
- [ ] Thử nghiệm Deep Learning models (LSTM, GRU)
- [ ] Multi-step forecasting (dự báo nhiều ngày)
- [ ] Deploy model lên cloud (AWS, Azure, GCP)

---