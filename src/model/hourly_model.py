import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
from datetime import datetime # Sửa lại import
import pickle

# ======================== JSON ENCODER (Giữ nguyên) ========================
class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder để xử lý các kiểu dữ liệu của NumPy
    (vd: np.float64, np.int64) mà thư viện json tiêu chuẩn không xử lý được.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ======================== HOURLY PIPELINE CLASS ========================

class MultiHorizonHourly_WalkForwardOptuna_XGBoost_Pipeline:
    """
    Pipeline chuyên dụng cho dự báo ĐA TẦM NHÌN (Multi-Horizon) HÀNG GIỜ.

    Sử dụng Walk-Forward Cross-Validation (với cửa sổ Mở rộng hoặc Trượt)
    kết hợp Optuna để tìm MỘT bộ siêu tham số (shared parameters) tối ưu
    cho tất cả các tầm nhìn (horizons).

    Sau đó, huấn luyện MỘT MÔ HÌNH CUỐI CÙNG (final model) riêng biệt
    cho TỪNG TẦM NHÌN (ví dụ: model cho t+1h, model cho t+3h,...).
    """
    
    def __init__(
        self,
        df,
        date_col,
        target_col,
        feature_cols,
        n_splits=5,
        test_size=720,       # THAY ĐỔI: Mặc định 720 giờ (30 ngày)
        gap_size_hours=24,    # THAY ĐỔI: Thêm Gap (giờ)
        mode="expanding",
        horizons=(1, 2, 3, 4 , 5), # THAY ĐỔI: Mặc định (giờ)
    ):
        """
        Khởi tạo pipeline.
        Args:
            df (pd.DataFrame): DataFrame chứa toàn bộ dữ liệu (đã qua FE).
            date_col (str): Tên cột datetime (đã sort).
            target_col (str): Tên cột mục tiêu (ví dụ: 'temp').
            feature_cols (list): Danh sách tất cả các cột feature.
            n_splits (int): Số lượng Fold cho Walk-Forward CV.
            test_size (int): Kích thước (số giờ) của mỗi tập test/validation.
            gap_size_hours (int): Số giờ chừa trống (gap) giữa train và test.
            mode (str): 'expanding' (cửa sổ mở rộng) hoặc 'rolling' (cửa sổ trượt).
            horizons (tuple): Các tầm nhìn dự đoán (ví dụ: t+1h, t+3h...).
        """
        self.df = df.copy().sort_values(date_col).reset_index(drop=True)
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_size_hours = gap_size_hours # Đã thêm
        self.mode = mode
        self.horizons = tuple(horizons)

        self.walkfolds = {h: [] for h in self.horizons}
        self.walkfold_dates = {h: [] for h in self.horizons}

        self.final_models = {h: None for h in self.horizons}

        self.best_params = None
        self.fold_history = [] # Lưu lịch sử các fold của Optuna

        # Tự động lọc ra các cột X (features)
        self.X_cols = [
            c for c in self.feature_cols
            if c not in [
                self.date_col,
                self.target_col,
                # Loại bỏ các cột target/date đã shift (nếu có)
                self.target_col + "_next",
                self.date_col + "_next",
            ]
        ]
        # Đảm bảo không có cột target_hX nào trong X_cols
        for h in self.horizons:
            target_h_name = f"{self.target_col}_h{h}"
            date_h_name = f"{self.date_col}_h{h}"
            if target_h_name in self.X_cols:
                self.X_cols.remove(target_h_name)
            if date_h_name in self.X_cols:
                self.X_cols.remove(date_h_name)

    # ======================== 1. TARGET SHIFT FOR ALL HORIZONS (TRAIN) ========================

    def add_target_shifts(self):
        """
        Tạo các cột target đã dịch chuyển (shifted) cho từng horizon.
        Ví dụ: 'temp_h1' (temp 1 giờ sau), 'temp_h24' (temp 24 giờ sau).
        
        Xóa các hàng NaN ở cuối DataFrame (không có target).
        """
        print("Đang tạo các cột target (shifted)...")
        for h in self.horizons:
            # Dịch chuyển target (giá trị tương lai) về hàng hiện tại
            self.df[f"{self.target_col}_h{h}"] = self.df[self.target_col].shift(-h)
            # Dịch chuyển cả date của target (để dễ kiểm tra)
            self.df[f"{self.date_col}_h{h}"] = self.df[self.date_col].shift(-h)

        # Xóa các hàng ở cuối mà không có đủ target (ví dụ: 24h cuối)
        target_cols = [f"{self.target_col}_h{h}" for h in self.horizons]
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)
        print(f"Hình dạng DF sau khi shift và dropna: {self.df.shape}")

    # ======================== 1b. TARGET SHIFT CHO TEST SET ========================

    def prepare_test_dataset(self, test_df_raw):
        """
        Chuẩn bị (shift target) cho một DataFrame test (hold-out) bên ngoài.
        """
        test_df = test_df_raw.copy().sort_values(self.date_col).reset_index(drop=True)

        for h in self.horizons:
            test_df[f"{self.target_col}_h{h}"] = test_df[self.target_col].shift(-h)
            test_df[f"{self.date_col}_h{h}"] = test_df[self.date_col].shift(-h)

        target_cols = [f"{self.target_col}_h{h}" for h in self.horizons]
        test_df = test_df.dropna(subset=target_cols).reset_index(drop=True)
        return test_df

    # ======================== 2. CREATE WALK-FORWARD FOLDS ========================

    def create_walkforward_folds(self):
        """
        Tạo ra các fold (cặp Train/Validation) theo chiến lược Walk-Forward.
        Lưu các fold (data) và ngày (dates) vào self.walkfolds
        """
        df_len = len(self.df)
        
        # Tính toán kích thước của mỗi 'bước' huấn luyện
        total_test_gap_span = (self.test_size + self.gap_size_hours) * self.n_splits
        train_span = df_len - total_test_gap_span
        
        if train_span <= 0:
             raise ValueError(
                 f"Dữ liệu quá ngắn ({df_len} giờ) cho n_splits={self.n_splits}, "
                 f"test_size={self.test_size}, gap={self.gap_size_hours}"
             )
        
        # 'step' là kích thước của block huấn luyện *ban đầu*
        # và cũng là kích thước của mỗi "bước nhảy"
        step = train_span // self.n_splits
        if step <= 0:
            raise ValueError(
                f"Không thể tạo Folds. 'step' (kích thước train) = {step}. "
                f"Hãy giảm n_splits hoặc test_size."
            )

        print(f"Đang tạo {self.n_splits} folds, mỗi fold test_size={self.test_size}h, gap={self.gap_size_hours}h...")

        self.walkfolds = {h: [] for h in self.horizons}
        self.walkfold_dates = {h: [] for h in self.horizons}

        # KHÔNG DÙNG 'initial_train_end' nữa
        # initial_train_end = df_len - total_test_gap_span

        for i in range(self.n_splits):
            
            # ========== SỬA LOGIC Ở ĐÂY ==========
            # train_end tăng thêm 'step' sau mỗi lần lặp
            train_end = step * (i + 1)
            
            if self.mode == "expanding":
                train_start = 0
            else: # 'rolling'
                # Cửa sổ trượt sẽ có kích thước bằng 'step'
                train_start = max(0, train_end - step)
            # ======================================
            
            test_start = train_end + self.gap_size_hours
            test_end = test_start + self.test_size
            
            if test_end > df_len:
                print(f"Bỏ qua fold {i+1} vì test_end ({test_end}) vượt quá độ dài dữ liệu ({df_len})")
                break

            train_df = self.df.iloc[train_start:train_end]
            test_df = self.df.iloc[test_start:test_end]
            
            # ... (Phần còn lại của hàm giữ nguyên) ...
            
            if train_df.empty or test_df.empty:
                print(f"Bỏ qua fold {i+1} vì train hoặc test rỗng.")
                continue

            # Tách X (features) và Y (targets)
            X_train = train_df[self.X_cols].reset_index(drop=True)
            X_val = test_df[self.X_cols].reset_index(drop=True)

            X_train_dates = train_df[self.date_col].reset_index(drop=True)
            X_val_dates = test_df[self.date_col].reset_index(drop=True)

            # Tạo dữ liệu cho từng horizon
            for h in self.horizons:
                y_train_h = train_df[f"{self.target_col}_h{h}"].reset_index(drop=True)
                y_val_h = test_df[f"{self.target_col}_h{h}"].reset_index(drop=True)

                y_train_dates_h = train_df[f"{self.date_col}_h{h}"].reset_index(drop=True)
                y_val_dates_h = test_df[f"{self.date_col}_h{h}"].reset_index(drop=True)

                self.walkfolds[h].append(
                    ((X_train, y_train_h), (X_val, y_val_h))
                )

                self.walkfold_dates[h].append(
                    ((X_train_dates, y_train_dates_h),
                     (X_val_dates, y_val_dates_h))
                )
        print(f"Đã tạo thành công {len(self.walkfolds[self.horizons[0]])} folds.")

    # ======================== 3. OPTUNA OBJECTIVE (SHARED PARAMS) ========================

    def create_objective(self):
        """
        Tạo hàm objective cho Optuna.
        Hàm này sẽ chạy Walk-Forward CV cho TẤT CẢ CÁC HORIZON
        và trả về RMSE TRUNG BÌNH CHUNG (mean over all horizons)
        """
        def objective(trial):
            # Không gian tìm kiếm siêu tham số
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10), # Có thể tăng max_depth
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }

            horizon_results = {} # Lưu kết quả chi tiết của trial này
            overall_rmse_list = [] # Lưu RMSE trung bình của MỖI horizon
            global_step = 0 # Dùng cho Pruning

            # Lặp qua các horizon (ví dụ: h=1, h=3, h=6...)
            for h in self.horizons:
                scores = []
                fold_results = []

                # Lặp qua các fold (ví dụ: 5 folds)
                for i, ((X_train, y_train), (X_val, y_val)) in enumerate(self.walkfolds[h]):

                    model = xgb.XGBRegressor(
                        **params,
                        n_estimators=1000, # Có thể tăng cho dữ liệu hourly
                        random_state=42,
                        tree_method='hist', # Dùng 'gpu_hist' nếu có GPU
                        early_stopping_rounds=100 # Giữ nguyên
                    )

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )

                    preds = model.predict(X_val)
                    
                    # Tính toán các chỉ số
                    rmse = np.sqrt(mean_squared_error(y_val, preds))

                    ss_res = np.sum((y_val - preds) ** 2)
                    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

                    scores.append(rmse)
                    fold_results.append({'rmse': rmse, 'r2': r2})

                    # Báo cáo cho Optuna (Pruning)
                    trial.report(rmse, step=global_step)
                    if trial.should_prune():
                        print(f"Trial {trial.number} pruned tại global_step={global_step}")
                        raise optuna.TrialPruned()
                    global_step += 1

                # Tính RMSE trung bình cho horizon H
                mean_rmse_h = float(np.mean(scores))
                horizon_results[h] = {
                    'mean_rmse': mean_rmse_h,
                    'fold_scores': fold_results,
                }
                overall_rmse_list.append(mean_rmse_h) # Thêm vào danh sách chung

            # Tính RMSE trung bình của các RMSE trung bình
            overall_mean_rmse = float(np.mean(overall_rmse_list))
            print(
                f"Trial {trial.number} finished, "
                f"overall mean RMSE across horizons={overall_mean_rmse:.4f}\n"
            )

            # Lưu lại lịch sử để debug
            self.fold_history.append({
                'trial': trial.number,
                'params': params,
                'horizon_results': horizon_results,
                'overall_mean_rmse': overall_mean_rmse,
            })
            
            # Trả về 1 con số duy nhất để Optuna tối ưu
            return overall_mean_rmse

        return objective

    # ======================== 4. RUN OPTUNA ========================

    def run_optuna(self, n_trials=50):
        """
        Chạy quá trình tối ưu hóa Optuna.
        """
        if not self.walkfolds[self.horizons[0]]:
            raise ValueError("Chưa tạo folds. Hãy gọi add_target_shifts() và create_walkforward_folds() trước.")
            
        objective_fn = self.create_objective()

        study = optuna.create_study(
            direction='minimize', # Tối ưu (giảm) RMSE
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=len(self.horizons) # Chờ ít nhất 1 horizon hoàn thành
            )
        )
        study.optimize(objective_fn, n_trials=n_trials)
        self.best_params = study.best_params
        return study

    # ======================== 5. TRAIN A FINAL MODEL PER HORIZON ========================

    def train_final_models(self):
        """
        Huấn luyện mô hình cuối cùng cho TỪNG horizon
        sử dụng bộ tham số tốt nhất (best_params) tìm được từ Optuna
        trên TOÀN BỘ self.df.
        """
        if self.best_params is None:
            raise ValueError("Bạn phải chạy run_optuna() trước khi train_final_models().")

        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 1500, # Có thể tăng n_estimators cho mô hình cuối cùng
            'early_stopping_rounds': 150 # Dùng (X_full, y_full_h) làm eval_set
        })

        X_full = self.df[self.X_cols]

        for h in self.horizons:
            print(f"Đang huấn luyện mô hình cuối cùng cho horizon h={h}...")
            y_full_h = self.df[f"{self.target_col}_h{h}"]

            model_h = xgb.XGBRegressor(
                **final_params,
                random_state=42,
                tree_method='hist' # Dùng 'gpu_hist' nếu có GPU
            )

            # Sử dụng early stopping ngay cả trên mô hình cuối cùng
            # bằng cách dùng chính tập full làm eval_set
            model_h.fit(
                X_full, 
                y_full_h, 
                eval_set=[(X_full, y_full_h)],
                verbose=False
            )
            
            self.final_models[h] = model_h
            print(f"Final model cho horizon {h} đã huấn luyện xong.")

    # ======================== 6. PREDICTION HELPERS ========================

    def predict_horizon(self, X_today, h):
        """
        Dự đoán cho MỘT horizon H, sử dụng MỘT dòng/nhiều dòng dữ liệu đầu vào.
        """
        if h not in self.horizons:
            raise ValueError(f"Horizon {h} không nằm trong {self.horizons}.")
        if self.final_models[h] is None:
            raise ValueError(f"Model cho horizon {h} chưa được huấn luyện. Gọi train_final_models().")

        # Đảm bảo X_today là DataFrame và chỉ lấy các cột cần thiết
        if isinstance(X_today, pd.Series):
             X_input = X_today[self.X_cols].to_frame().T
        else:
             X_input = X_today[self.X_cols]
             
        return self.final_models[h].predict(X_input)

    def predict_all_horizons(self, X_today):
        """
        Dự đoán cho TẤT CẢ các horizon.
        """
        results = {}
        for h in self.horizons:
            results[h] = self.predict_horizon(X_today, h)
        return results

    # ======================== 6b. BUILD PREDICTIONS FRAME (Plot) ========================

    def get_predictions_frame(self, df_with_shift, h):
        """
        Tạo một DataFrame chứa (y_true, y_pred, feature_time, target_time)
        cho một horizon H cụ thể, dựa trên một DataFrame đầu vào.
        """
        if h not in self.horizons:
            raise ValueError(f"Horizon {h} không nằm trong {self.horizons}.")
        if self.final_models[h] is None:
            raise ValueError(f"Model cho horizon {h} chưa được huấn luyện.")

        X = df_with_shift[self.X_cols]
        y_true = df_with_shift[f"{self.target_col}_h{h}"]
        y_pred = self.final_models[h].predict(X)

        feature_time = df_with_shift[self.date_col]
        
        # Đảm bảo cột target_time tồn tại
        if f"{self.date_col}_h{h}" in df_with_shift.columns:
            target_time = df_with_shift[f"{self.date_col}_h{h}"]
        else:
            # Nếu không có, tự tính toán (chậm hơn)
            target_time = pd.to_datetime(feature_time) + pd.to_timedelta(h, unit='h')

        df_pred = pd.DataFrame({
            "feature_time": feature_time.values,
            "target_time": target_time.values,
            "y_true": y_true.values,
            "y_pred": y_pred,
        })
        return df_pred

    def plot_predictions(self, df_with_shift, h, use_target_time=True, n_points=None):
        """
        Vẽ biểu đồ so sánh True vs Predicted cho horizon H.
        """
        df_pred = self.get_predictions_frame(df_with_shift, h)

        df_plot = df_pred.copy()
        if use_target_time and "target_time" in df_plot.columns:
            df_plot["x_axis"] = pd.to_datetime(df_plot["target_time"])
            xlabel = f"Target Time (Dự đoán cho lúc...)"
        else:
            df_plot["x_axis"] = pd.to_datetime(df_plot["feature_time"])
            xlabel = f"Feature Time (Dự đoán từ lúc...)"

        if n_points is not None:
            df_plot = df_plot.iloc[-n_points:] # Lấy N điểm cuối cùng

        plt.figure(figsize=(15, 5)) # Tăng kích thước
        plt.plot(df_plot["x_axis"], df_plot["y_true"], label="True", alpha=0.8)
        plt.plot(df_plot["x_axis"], df_plot["y_pred"], label="Predicted", linestyle='--')
        plt.xlabel(xlabel)
        plt.ylabel(self.target_col)
        plt.title(f"Dự báo cho tầm nhìn {h} giờ (Horizon h={h})")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # ======================== 7. EVAL ON TRAIN SET (CHECK OVERFIT) ========================

    def evaluate_train_models(self):
        """
        Đánh giá các mô hình cuối cùng trên TOÀN BỘ tập huấn luyện (self.df)
        để kiểm tra overfitting.
        """
        print("Đang đánh giá mô hình trên toàn bộ tập TRAIN (để kiểm tra overfit)...")
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("Tất cả models phải được huấn luyện trước khi đánh giá.")

        metrics = {}
        X_train_full = self.df[self.X_cols]

        for h in self.horizons:
            y_train_h = self.df[f"{self.target_col}_h{h}"]
            preds = self.final_models[h].predict(X_train_full)

            mse = mean_squared_error(y_train_h, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_train_h, preds)
            eps = 1e-8
            mape = np.mean(np.abs((y_train_h - preds) / (y_train_h + eps))) * 100

            ss_res = np.sum((y_train_h - preds) ** 2)
            ss_tot = np.sum((y_train_h - np.mean(y_train_h)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

            print(
                f"[TRAIN] Horizon h={h} -> "
                f"RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}"
            )

            metrics[h] = {
                'rmse': rmse, 'mae': mae, 'mape': mape, 'mse': mse, 'r2': r2,
            }

        return metrics

    # ======================== 8. EVALUATION ON A TEST SET ========================

    def evaluate_final_models(self, test_df):
        """
        Đánh giá các mô hình cuối cùng trên một tập TEST (hold-out) bên ngoài.
        """
        print("Đang đánh giá mô hình trên tập TEST (hold-out)...")
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("Tất cả models phải được huấn luyện trước khi đánh giá.")

        metrics = {}
        X_test = test_df[self.X_cols]

        for h in self.horizons:
            y_test_h = test_df[f"{self.target_col}_h{h}"]
            preds = self.final_models[h].predict(X_test)

            mse = mean_squared_error(y_test_h, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_h, preds)
            eps = 1e-8
            mape = np.mean(np.abs((y_test_h - preds) / (y_test_h + eps))) * 100

            ss_res = np.sum((y_test_h - preds) ** 2)
            ss_tot = np.sum((y_test_h - np.mean(y_test_h)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

            print(
                f"[TEST] Horizon h={h} -> "
                f"RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}"
            )

            metrics[h] = {
                'rmse': rmse, 'mae': mae, 'mape': mape, 'mse': mse, 'r2': r2,
            }

        return metrics

    # ======================== 9. SAVE FINAL MODELS (PICKLE) ========================

    def save_all_final_models_pkl(self, base_filename_prefix):
        """
        Lưu tất cả các model cuối cùng (final_models) ra đĩa dùng PICKLE (.pkl).
        """
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("Phải train model trước khi lưu. Hãy gọi train_final_models().")
        
        saved_paths = {}
        for h in self.horizons:
            model = self.final_models[h]
            # Tên file sẽ là prefix + _h1.pkl, _h3.pkl ...
            filename = f"{base_filename_prefix}_h{h}.pkl"
            
            try:
                with open(filename, 'wb') as f: 
                    pickle.dump(model, f)
                saved_paths[h] = filename
                print(f"Model (Pickle) cho horizon {h} đã được lưu vào: {filename}")
            except Exception as e:
                print(f"Lỗi khi lưu file pickle {filename}: {e}")
        
        return saved_paths

    # ======================== 10. SAVE RESULTS SUMMARY (JSON) ========================
    
    def save_results_to_json(
        self,
        filename,
        model_name,
        train_metrics,
        test_metrics,
        saved_model_paths=None
    ):
        """
        Lưu kết quả tóm tắt (best param, metrics, timestamp,...) vào 1 file JSON.
        """
        
        results_summary = {
            "model_name": model_name,
            "run_timestamp_utc": datetime.utcnow().isoformat(), # Đã sửa (xóa 1 .datetime)
            "target_column": self.target_col,
            "feature_columns": self.X_cols,
            "horizons_hours": self.horizons, # Đổi tên cho rõ
            "cross_validation_setup": {
                "n_splits": self.n_splits,
                "test_size_per_fold_hours": self.test_size, # Đổi tên
                "gap_size_hours": self.gap_size_hours, # THÊM
                "mode": self.mode,
            },
            "optuna_best_params": self.best_params,
            "evaluation_train_set": train_metrics,
            "evaluation_test_set": test_metrics,
            "saved_model_files": saved_model_paths if saved_model_paths else {},
            "optuna_fold_history": self.fold_history 
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, indent=4, cls=NpEncoder)
            print(f"Kết quả tóm tắt đã được lưu vào: {filename}")
        except Exception as e:
            print(f"Lỗi khi lưu file JSON: {e}")