import pandas as pd
import numpy as np
import xgboost as xgb
import optuna  
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class MultiHorizonWalkForwardOptuna_XGBoost_Pipeline:
    def __init__(
        self,
        df,
        date_col,
        target_col,
        feature_cols,
        n_splits=5,
        test_size=30,
        mode="expanding",
        horizons=(1, 2, 3, 4, 5),   # t+1 ... t+5
    ):
        """
        df          : pandas DataFrame in chronological order
        date_col    : name of date column (datetime)
        target_col  : name of target column (e.g., 'temp')
        feature_cols: list of feature columns (SHOULD NOT include target)
        n_splits    : number of walk-forward folds
        test_size   : number of days in each test fold
        mode        : "expanding" or "rolling"
        horizons    : iterable of forecast horizons (days ahead), e.g. (1,2,3,4,5)
        """
        self.df = df.copy().sort_values(date_col).reset_index(drop=True)
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.n_splits = n_splits
        self.test_size = test_size
        self.mode = mode
        self.horizons = tuple(horizons)

        # For each horizon h, we will have:
        #   self.walkfolds[h]      : list of ((X_train, y_train_h), (X_val, y_val_h))
        #   self.walkfold_dates[h] : list of ((X_train_dates, y_train_dates_h),
        #                                     (X_val_dates,   y_val_dates_h))
        self.walkfolds = {h: [] for h in self.horizons}
        self.walkfold_dates = {h: [] for h in self.horizons}

        # Final model per horizon: self.final_models[h] = trained XGBRegressor
        self.final_models = {h: None for h in self.horizons}

        self.best_params = None
        self.fold_history = []  # store optuna trial results

        # Actual feature columns: remove date/target if accidentally passed in
        self.X_cols = [
            c for c in self.feature_cols
            if c not in [
                self.date_col,
                self.target_col,
                self.target_col + "_next",
                self.date_col + "_next",
            ]
        ]

    # ======================== 1. TARGET SHIFT FOR ALL HORIZONS (TRAIN) ========================

    def add_target_shifts(self):
        """
        Create shifted targets for all horizons, e.g. for target='temp':
        - temp_h1 = temp at t+1
        - temp_h2 = temp at t+2
        - ...
        Also create corresponding date columns date_h{h} for reference.
        After that, drop rows with any NaN in those shifted targets.
        """
        for h in self.horizons:
            self.df[f"{self.target_col}_h{h}"] = self.df[self.target_col].shift(-h)
            self.df[f"{self.date_col}_h{h}"] = self.df[self.date_col].shift(-h)

        target_cols = [f"{self.target_col}_h{h}" for h in self.horizons]
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)

    # ======================== 1b. TARGET SHIFT CHO TEST SET ========================

    def prepare_test_dataset(self, test_df_raw):
        """
        Create shifted targets for all horizons on a separate test DataFrame.

        Parameters
        ----------
        test_df_raw : pd.DataFrame
            Raw test DataFrame containing at least [date_col, target_col] and feature_cols.

        Returns
        -------
        test_df : pd.DataFrame
            A new DataFrame with:
            - original columns
            - {target_col}_h{h}
            - {date_col}_h{h}
          Rows with NaN in any shifted target are dropped.
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
        Create walk-forward folds for each horizon.

        For each horizon h:
          self.walkfolds[h] = [
              ((X_train, y_train_h), (X_val, y_val_h)),
              ...
          ]

          self.walkfold_dates[h] = [
              ((X_train_dates, y_train_dates_h), (X_val_dates, y_val_dates_h)),
              ...
          ]
        """
        df_len = len(self.df)
        step = (df_len - self.test_size) // self.n_splits

        # reset containers
        self.walkfolds = {h: [] for h in self.horizons}
        self.walkfold_dates = {h: [] for h in self.horizons}

        for i in range(self.n_splits):
            train_end = step * (i + 1)
            test_start = train_end + 91   # 91-day gap
            test_end = test_start + self.test_size
            if test_end > df_len:
                break

            if self.mode == "expanding":
                train_start = 0
            else:  # rolling
                train_start = max(0, train_end - step * 2)

            train_df = self.df.iloc[train_start:train_end]
            test_df = self.df.iloc[test_start:test_end]

            # Features (same for all horizons)
            X_train = train_df[self.X_cols].reset_index(drop=True)
            X_val = test_df[self.X_cols].reset_index(drop=True)

            # Dates of "feature time" (t)
            X_train_dates = train_df[self.date_col].reset_index(drop=True)
            X_val_dates = test_df[self.date_col].reset_index(drop=True)

            # For each horizon, build targets & date of target
            for h in self.horizons:
                y_train_h = train_df[f"{self.target_col}_h{h}"].reset_index(drop=True)
                y_val_h = test_df[f"{self.target_col}_h{h}"].reset_index(drop=True)

                y_train_dates_h = train_df[f"{self.date_col}_h{h}"].reset_index(drop=True)
                y_val_dates_h = test_df[f"{self.date_col}_h{h}"].reset_index(drop=True)

                # Walkfolds
                self.walkfolds[h].append(
                    ((X_train, y_train_h), (X_val, y_val_h))
                )

                # Date folds
                self.walkfold_dates[h].append(
                    ((X_train_dates, y_train_dates_h),
                     (X_val_dates,   y_val_dates_h))
                )

    # ======================== 3. OPTUNA OBJECTIVE (SHARED PARAMS) ========================

    def create_objective(self):
        """
        Optuna objective.

        A single set of hyperparameters is shared for all horizons.
        The objective value is the mean of the mean-RMSE across horizons.

        - Uses early_stopping_rounds in XGBRegressor constructor (compatible with new XGBoost).
        - Uses trial.report + trial.should_prune to prune poor trials.
        """
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
            }

            horizon_results = {}
            overall_rmse_list = []
            global_step = 0  # step index for Optuna pruning

            for h in self.horizons:
                scores = []
                fold_results = []

                for i, ((X_train, y_train), (X_val, y_val)) in enumerate(self.walkfolds[h]):

                    model = xgb.XGBRegressor(
                        **params,
                        n_estimators=800,
                        random_state=42,
                        tree_method='hist',
                        # early stopping for new XGBoost (passed via constructor)
                        early_stopping_rounds=100
                    )

                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )

                    preds = model.predict(X_val)
                    rmse = mean_squared_error(y_val, preds)
                    ss_res = np.sum((y_val - preds) ** 2)
                    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

                    scores.append(rmse)
                    fold_results.append({'rmse': rmse, 'r2': r2})

                    print(
                        f"Trial {trial.number}, Horizon h={h}, "
                        f"Fold {i+1}: RMSE={rmse:.4f}, R²={r2:.4f}"
                    )

                    # Report to Optuna and possibly prune
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
        Run Optuna to find shared hyperparameters for all horizons.

        Uses MedianPruner to prune poor trials.
        """
        objective_fn = self.create_objective()

        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=5  # number of steps before pruning is enabled
            )
        )
        study.optimize(objective_fn, n_trials=n_trials)
        self.best_params = study.best_params
        return study

    # ======================== 5. TRAIN A FINAL MODEL PER HORIZON ========================

    def train_final_models(self):
        """
        Train one final model per horizon on the entire development set
        using the best hyperparameters from Optuna.
        """
        if self.best_params is None:
            raise ValueError("You must run run_optuna() before train_final_models().")

        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 800
        })

        X_full = self.df[self.X_cols]

        for h in self.horizons:
            y_full_h = self.df[f"{self.target_col}_h{h}"]

            model_h = xgb.XGBRegressor(
                **final_params,
                random_state=42,
                tree_method='hist'
            )

            model_h.fit(X_full, y_full_h, verbose=False)
            self.final_models[h] = model_h

    # ======================== 6. PREDICTION HELPERS ========================

    def predict_horizon(self, X_today, h):
        """
        Predict target at t+h for a single row or DataFrame X_today.
        """
        if h not in self.horizons:
            raise ValueError(f"Horizon {h} not in configured horizons {self.horizons}.")
        if self.final_models[h] is None:
            raise ValueError(
                f"Model for horizon {h} is not trained. "
                f"Call train_final_models() first."
            )

        X_input = X_today[self.X_cols]
        return self.final_models[h].predict(X_input)

    def predict_all_horizons(self, X_today):
        """
        Predict for all configured horizons for a given X_today (1 row or small DataFrame).
        Returns a dict: {h: prediction_array}
        """
        results = {}
        for h in self.horizons:
            results[h] = self.predict_horizon(X_today, h)
        return results 

    # ======================== 6b. BUILD PREDICTIONS FRAME (Plot) ========================

    def get_predictions_frame(self, df_with_shift, h):
        """
        Build a DataFrame with dates, true values and predictions for horizon h.

        Parameters
        ----------
        df_with_shift : pd.DataFrame
            DataFrame (train or test) that already has the columns:
            - self.date_col
            - f"{self.target_col}_h{h}"
            - f"{self.date_col}_h{h}" (if available)

        Returns
        -------
        df_pred : pd.DataFrame with columns:
            - feature_time : time t (where features are taken from)
            - target_time  : time t+h (if available), else feature_time
            - y_true       : true target at t+h
            - y_pred       : model prediction at t+h
        """
        if h not in self.horizons:
            raise ValueError(f"Horizon {h} not in configured horizons {self.horizons}.")
        if self.final_models[h] is None:
            raise ValueError(
                f"Model for horizon {h} is not trained. "
                f"Call train_final_models() first."
            )

        X = df_with_shift[self.X_cols]
        y_true = df_with_shift[f"{self.target_col}_h{h}"]
        y_pred = self.final_models[h].predict(X)

        feature_time = df_with_shift[self.date_col]
        if f"{self.date_col}_h{h}" in df_with_shift.columns:
            target_time = df_with_shift[f"{self.date_col}_h{h}"]
        else:
            target_time = feature_time

        df_pred = pd.DataFrame({
            "feature_time": feature_time.values,
            "target_time": target_time.values,
            "y_true": y_true.values,
            "y_pred": y_pred,
        })
        return df_pred

    def plot_predictions(self, df_with_shift, h, use_target_time=True, n_points=None):
        """
        Plot true vs predicted for a given horizon h.

        Parameters
        ----------
        df_with_shift : pd.DataFrame
            DataFrame (typically test_df already shifted via prepare_test_dataset).
        h : int
            Horizon to plot (must be in self.horizons).
        use_target_time : bool, default True
            If True and target_time exists, x-axis = target_time (t+h).
            If False, x-axis = feature_time (t).
        n_points : int or None
            If not None, only the first n_points are plotted.
        """
        df_pred = self.get_predictions_frame(df_with_shift, h)

        df_plot = df_pred.copy()
        if use_target_time and "target_time" in df_plot.columns:
            df_plot["x"] = df_plot["target_time"]
        else:
            df_plot["x"] = df_plot["feature_time"]

        if n_points is not None:
            df_plot = df_plot.iloc[:n_points]

        plt.figure(figsize=(10, 4))
        plt.plot(df_plot["x"], df_plot["y_true"], label="True")
        plt.plot(df_plot["x"], df_plot["y_pred"], label="Predicted")
        plt.xlabel("Date")
        plt.ylabel(self.target_col)
        plt.title(f"Horizon h={h} forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ======================== 7. EVAL ON TRAIN SET (CHECK OVERFIT) ========================

    def evaluate_train_models(self):
        """
        Evaluate all final models on the training (development) set.
        Returns a dict: {h: {'rmse','mae','mape','mse','r2'}}
        """
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("All horizon models must be trained before evaluation.")

        metrics = {}
        X_train_full = self.df[self.X_cols]

        for h in self.horizons:
            y_train_h = self.df[f"{self.target_col}_h{h}"]
            preds = self.final_models[h].predict(X_train_full)

            rmse = mean_squared_error(y_train_h, preds)
            mae = mean_absolute_error(y_train_h, preds)
            mse = mean_squared_error(y_train_h, preds)

            # avoid division by zero for MAPE
            eps = 1e-8
            mape = np.mean(np.abs((y_train_h - preds) / (y_train_h + eps))) * 100

            ss_res = np.sum((y_train_h - preds) ** 2)
            ss_tot = np.sum((y_train_h - np.mean(y_train_h)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

            print(
                f"[TRAIN] Horizon h={h} -> "
                f"RMSE: {rmse:.4f}, R²: {r2:.4f}"
            )

            metrics[h] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'mse': mse,
                'r2': r2,
            }

        return metrics

    # ======================== 8. EVALUATION ON A TEST SET ========================

    def evaluate_final_models(self, test_df):
        """
        Evaluate all final models on a separate test DataFrame that already contains
        the shifted targets (i.e., you must have created _h1.._h5 on it).
        Returns a dict: {h: {'rmse','mae','mape','mse','r2'}}
        """
        if any(self.final_models[h] is None for h in self.horizons):
            raise ValueError("All horizon models must be trained before evaluation.")

        metrics = {}
        X_test = test_df[self.X_cols]

        for h in self.horizons:
            y_test_h = test_df[f"{self.target_col}_h{h}"]
            preds = self.final_models[h].predict(X_test)

            rmse = mean_squared_error(y_test_h, preds)
            mae = mean_absolute_error(y_test_h, preds)
            mse = mean_squared_error(y_test_h, preds)

            eps = 1e-8
            mape = np.mean(np.abs((y_test_h - preds) / (y_test_h + eps))) * 100

            ss_res = np.sum((y_test_h - preds) ** 2)
            ss_tot = np.sum((y_test_h - np.mean(y_test_h)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

            print(f"Horizon h={h} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

            metrics[h] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'mse': mse,
                'r2': r2,
            }

        return metrics
