import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
from datetime import datetime # Fixed import
import pickle

# ======================== JSON ENCODER (Keep as is) ========================
class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types
    (e.g., np.float64, np.int64) that the standard json library cannot handle.
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
    Specialized pipeline for MULTI-HORIZON HOURLY forecasting.

    Uses Walk-Forward Cross-Validation (with Expanding or Rolling window)
    combined with Optuna to find ONE optimal set of hyperparameters (shared parameters)
    for all horizons.

    Then, trains ONE FINAL MODEL separately
    for EACH HORIZON (e.g., model for t+1h, model for t+3h, ...).
    """
    
    def __init__(
        self,
        df,
        date_col,
        target_col,
        feature_cols,
        n_splits=5,
        test_size=720,       # CHANGE: Default 720 hours (30 days)
        gap_size_hours=24,    # CHANGE: Added Gap (hours)
        mode="expanding",
        horizons=(1, 2, 3, 4 , 5), # CHANGE: Default (hours)
    ):
        """
        Initialize pipeline.
        Args:
            df (pd.DataFrame): DataFrame containing all data (after FE).
            date_col (str): Name of datetime column (already sorted).
            target_col (str): Name of target column (e.g., 'temp').
            feature_cols (list): List of all feature columns.
            n_splits (int): Number of Folds for Walk-Forward CV.
            test_size (int): Size (hours) of each test/validation set.
            gap_size_hours (int): Number of hours gap between train and test.
            mode (str): 'expanding' (expanding window) or 'rolling' (rolling window).
            horizons (tuple): Forecast horizons (e.g., t+1h, t+3h...).
        """
        self.df = df.copy().sort_values(date_col).reset_index(drop=True)
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_size_hours = gap_size_hours # Added
        self.mode = mode
        self.horizons = tuple(horizons)

        self.walkfolds = {h: [] for h in self.horizons}
        self.walkfold_dates = {h: [] for h in self.horizons}

        self.final_models = {h: None for h in self.horizons}

        self.best_params = None
        self.fold_history = [] # Store Optuna fold history

        # Automatically filter out X columns (features)
        self.X_cols = [
            c for c in self.feature_cols
            if c not in [
                self.date_col,
                self.target_col,
                # Remove shifted target/date columns (if any)
                self.target_col + "_next",
                self.date_col + "_next",
            ]
        ]
        # Ensure no target_hX column is in X_cols
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
        Create shifted target columns for each horizon.
        Example: 'temp_h1' (temp 1 hour ahead), 'temp_h24' (temp 24 hours ahead).
        
        Remove NaN rows at the end of DataFrame (no target available).
        """
        print("Creating shifted target columns...")
        for h in self.horizons:
            # Shift target (future value) to current row
            self.df[f"{self.target_col}_h{h}"] = self.df[self.target_col].shift(-h)
            # Also shift date of target (for easy verification)
            self.df[f"{self.date_col}_h{h}"] = self.df[self.date_col].shift(-h)

        # Remove rows at the end without sufficient targets (e.g., last 24h)
        target_cols = [f"{self.target_col}_h{h}" for h in self.horizons]
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)
        print(f"DataFrame shape after shift and dropna: {self.df.shape}")

    # ======================== 1b. TARGET SHIFT FOR TEST SET ========================

    def prepare_test_dataset(self, test_df_raw):
        """
        Prepare (shift target) for an external test DataFrame (hold-out).
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
        Create folds (Train/Validation pairs) using Walk-Forward strategy.
        Store folds (data) and dates into self.walkfolds
        """
        df_len = len(self.df)
        
        # Calculate size of each training 'step'
        total_test_gap_span = (self.test_size + self.gap_size_hours) * self.n_splits
        train_span = df_len - total_test_gap_span
        
        if train_span <= 0:
             raise ValueError(
                 f"Data too short ({df_len} hours) for n_splits={self.n_splits}, "
                 f"test_size={self.test_size}, gap={self.gap_size_hours}"
             )
        
        # 'step' is the size of *initial* training block
        # and also the size of each "jump"
        step = train_span // self.n_splits
        if step <= 0:
            raise ValueError(
                f"Cannot create Folds. 'step' (train size) = {step}. "
                f"Please reduce n_splits or test_size."
            )

        print(f"Creating {self.n_splits} folds, each fold test_size={self.test_size}h, gap={self.gap_size_hours}h...")

        self.walkfolds = {h: [] for h in self.horizons}
        self.walkfold_dates = {h: [] for h in self.horizons}

        # NO LONGER USE 'initial_train_end'
        # initial_train_end = df_len - total_test_gap_span

        for i in range(self.n_splits):
            
            # ========== FIXED LOGIC HERE ==========
            # train_end increases by 'step' after each iteration
            train_end = step * (i + 1)
            
            if self.mode == "expanding":
                train_start = 0
            else: # 'rolling'
                # Rolling window will have size equal to 'step'
                train_start = max(0, train_end - step)
            # ======================================
            
            test_start = train_end + self.gap_size_hours
            test_end = test_start + self.test_size
            
            if test_end > df_len:
                print(f"Skipping fold {i+1} because test_end ({test_end}) exceeds data length ({df_len})")
                break

            train_df = self.df.iloc[train_start:train_end]
            test_df = self.df.iloc[test_start:test_end]
            
            # ... (Rest of function remains unchanged) ...
            
            if train_df.empty or test_df.empty:
                print(f"Skipping fold {i+1} because train or test is empty.")
                continue

            # Split X (features) and Y (targets)
            X_train = train_df[self.X_cols].reset_index(drop=True)
            X_val = test_df[self.X_cols].reset_index(drop=True)

            X_train_dates = train_df[self.date_col].reset_index(drop=True)
            X_val_dates = test_df[self.date_col].reset_index(drop=True)

            # Create data for each horizon
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
        print(f"Successfully created {len(self.walkfolds[self.horizons[0]])} folds.")

    # ======================== 3. OPTUNA OBJECTIVE (SHARED PARAMS) ========================

    def create_objective(self):
        """
        Create objective function for Optuna.
        This function will run Walk-Forward CV for ALL HORIZONS
        and return OVERALL MEAN RMSE (mean over all horizons)
        """
        def objective(trial):
            # Hyperparameter search space
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10), # Can increase max_depth
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }

            horizon_results = {} # Store detailed results of this trial
            overall_rmse_list = [] # Store mean RMSE of EACH horizon
            global_step = 0 # Used for Pruning

            # Loop through horizons (e.g., h=1, h=3, h=6...)
            for h in self.horizons:
                scores = []
                fold_results = []

                # Loop through folds (e.g., 5 folds)
                for i, ((X_train, y_train), (X_val, y_val)) in enumerate(self.walkfolds[h]):

                    model = xgb.XGBRegressor(
                        **params,
                        n_estimators=1000,
                        random_state=42,
                        tree_method='hist', 
                        early_stopping_rounds=100 
                    )

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )

                    preds = model.predict(X_val)
                    
                    rmse = np.sqrt(mean_squared_error(y_val, preds))

                    ss_res = np.sum((y_val - preds) ** 2)
                    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

                    scores.append(rmse)
                    fold_results.append({'rmse': rmse, 'r2': r2})

                    trial.report(rmse, step=global_step)
                    if trial.should_prune():
                        print(f"Trial {trial.number} pruned at global_step={global_step}")
                        raise optuna.TrialPruned()
                    global_step += 1

                mean_rmse_h = float(np.mean(scores))
                horizon_results[h] = {
                    'mean_rmse': mean_rmse_h,
                    'fold_scores': fold_results,
                }
                overall_rmse_list.append(mean_rmse_h)

            overall_mean_rmse = float(np.mean(overall_rmse_list))
            print(
                f"Trial {trial.number} finished, "
                f"overall mean RMSE across horizons={overall_mean_rmse:.4f}\n"
            )

            self.fold_history.append({
                'trial': trial.number,
                'params': params,
                'horizon_results': horizon_results,
                'overall_mean_rmse': overall_mean_rmse,
            })
            
            return overall_mean_rmse

        return objective

    # ======================== 4. RUN OPTUNA ========================

    def run_optuna(self, n_trials=50):
        """
        Run Optuna optimization process.
        """
        if not self.walkfolds[self.horizons[0]]:
            raise ValueError("Folds not created yet. Call add_target_shifts() and create_walkforward_folds() first.")
            
        objective_fn = self.create_objective()

        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=len(self.horizons)
            )
        )
        study.optimize(objective_fn, n_trials=n_trials)
        self.best_params = study.best_params
        return study

    # ======================== 5. TRAIN A FINAL MODEL PER HORIZON ========================

    def train_final_models(self):
        """
        Train final model for EACH horizon
        using best parameters (best_params) found from Optuna
        on ENTIRE self.df.
        """
        if self.best_params is None:
            raise ValueError("You must run run_optuna() before train_final_models().")

        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 1500,
            'early_stopping_rounds': 150
        })

        X_full = self.df[self.X_cols]

        for h in self.horizons:
            print(f"Training final model for horizon h={h}...")
            y_full_h = self.df[f"{self.target_col}_h{h}"]

            model_h = xgb.XGBRegressor(
                **final_params,
                random_state=42,
                tree_method='hist'
            )

            model_h.fit(
                X_full, 
                y_full_h, 
                eval_set=[(X_full, y_full_h)],
                verbose=False
            )
            
            self.final_models[h] = model_h
            print(f"Final model for horizon {h} training completed.")

    # ======================== 6. PREDICTION HELPERS ========================

    def predict_horizon(self, X_today, h):
        """
        Predict for ONE horizon H, using ONE row/multiple rows of input data.
        """
        if h not in self.horizons:
            raise ValueError(f"Horizon {h} is not in {self.horizons}.")
        if self.final_models[h] is None:
            raise ValueError(f"Model for horizon {h} not trained yet. Call train_final_models().")

        if isinstance(X_today, pd.Series):
             X_input = X_today[self.X_cols].to_frame().T
        else:
             X_input = X_today[self.X_cols]
             
        return self.final_models[h].predict(X_input)

    def predict_all_horizons(self, X_today):
        """
        Predict for ALL horizons.
        """
        results = {}
        for h in self.horizons:
            results[h] = self.predict_horizon(X_today, h)
        return results

    # ======================== 6b. BUILD PREDICTIONS FRAME (Plot) ========================

    def get_predictions_frame(self, df_with_shift, h):
        """
        Create a DataFrame containing (y_true, y_pred, feature_time, target_time)
        for a specific horizon H, based on an input DataFrame.
        """
        if h not in self.horizons:
            raise ValueError(f"Horizon {h} is not in {self.horizons}.")
        if self.final_models[h] is None:
            raise ValueError(f"Model for horizon {h} not trained yet.")

        X = df_with_shift[self.X_cols]
        y_true = df_with_shift[f"{self.target_col}_h{h}"]
        y_pred = self.final_models[h].predict(X)

        feature_time = df_with_shift[self.date_col]
        
        # Ensure target_time column exists
        if f"{self.date_col}_h{h}" in df_with_shift.columns:
            target_time = df_with_shift[f"{self.date_col}_h{h}"]
        else:
            # If not available, calculate it (slower)
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
        Plot comparison chart of True vs Predicted for horizon H.
        """
        df_pred = self.get_predictions_frame(df_with_shift, h)

        df_plot = df_pred.copy()
        if use_target_time and "target_time" in df_plot.columns:
            df_plot["x_axis"] = pd.to_datetime(df_plot["target_time"])
            xlabel = f"Target Time (Prediction for...)"
        else:
            df_plot["x_axis"] = pd.to_datetime(df_plot["feature_time"])
            xlabel = f"Feature Time (Prediction from...)"

        if n_points is not None:
            df_plot = df_plot.iloc[-n_points:] # Take last N points

        plt.figure(figsize=(15, 5)) # Increase size
        plt.plot(df_plot["x_axis"], df_plot["y_true"], label="True", alpha=0.8)
        plt.plot(df_plot["x_axis"], df_plot["y_pred"], label="Predicted", linestyle='--')
        plt.xlabel(xlabel)
        plt.ylabel(self.target_col)
        plt.title(f"Forecast for {h}-hour horizon (Horizon h={h})")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # ======================== 7. EVAL ON TRAIN SET (CHECK OVERFIT) ========================

    def evaluate_train_models(self):
        """
        Evaluate final models on ENTIRE training set (self.df)
        to check for overfitting.
        """
        print("Evaluating model on entire TRAIN set (to check overfitting)...")
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("All models must be trained before evaluation.")

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
        Evaluate final models on an external TEST set (hold-out).
        """
        print("Evaluating model on TEST set (hold-out)...")
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("All models must be trained before evaluation.")

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
        Save all final models (final_models) to disk using PICKLE (.pkl).
        """
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("Must train models before saving. Call train_final_models().")
        
        saved_paths = {}
        for h in self.horizons:
            model = self.final_models[h]
            # Filename will be prefix + _h1.pkl, _h3.pkl ...
            filename = f"{base_filename_prefix}_h{h}.pkl"
            
            try:
                with open(filename, 'wb') as f: 
                    pickle.dump(model, f)
                saved_paths[h] = filename
                print(f"Model (Pickle) for horizon {h} saved to: {filename}")
            except Exception as e:
                print(f"Error saving pickle file {filename}: {e}")
        
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
        Save results summary (best params, metrics, timestamp, ...) to a JSON file.
        """
        
        results_summary = {
            "model_name": model_name,
            "run_timestamp_utc": datetime.utcnow().isoformat(), # Fixed (removed one .datetime)
            "target_column": self.target_col,
            "feature_columns": self.X_cols,
            "horizons_hours": self.horizons,
            "cross_validation_setup": {
                "n_splits": self.n_splits,
                "test_size_per_fold_hours": self.test_size, 
                "gap_size_hours": self.gap_size_hours,
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
            print(f"Results summary saved to: {filename}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")