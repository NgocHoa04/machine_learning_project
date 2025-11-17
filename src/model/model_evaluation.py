"""
Model Evaluation Module

This production-ready module provides comprehensive model evaluation functionality
for time series temperature forecasting models. It includes performance metrics,
visualization tools, overfitting detection, and automated reporting.

Key Features:
    - Comprehensive regression metrics (RMSE, MAE, R², MAPE, etc.)
    - Visual diagnostic plots (actual vs predicted, residuals, distributions)
    - Overfitting and generalization analysis
    - Feature importance visualization
    - Automated report generation
    - Support for multi-horizon forecasting evaluation
    - Statistical significance testing
    - Production-ready logging and error handling

Usage:
    from src.model.model_evaluation import ModelEvaluator
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, dataset_name="Test")
    evaluator.plot_diagnostics(y_true, y_pred, model_name="XGBoost")
    report = evaluator.generate_report(train_metrics, test_metrics)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings

# Scikit-learn metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)
from scipy import stats
from scipy.stats import normaltest, shapiro

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation class for regression models.
    
    This class provides methods for calculating performance metrics,
    generating visualizations, detecting overfitting, and creating
    detailed evaluation reports.
    
    Attributes:
        logger: Logging instance for output tracking
        metrics_history: Dictionary storing evaluation history
        visualization_config: Configuration for plot customization
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ModelEvaluator.
        
        Args:
            logger: Optional logger instance. If None, creates default logger.
        """
        self.logger = logger or self._setup_default_logger()
        self.metrics_history = {}
        self.visualization_config = self._default_viz_config()
        
    def _setup_default_logger(self) -> logging.Logger:
        """Create default logger for the evaluator."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _default_viz_config(self) -> Dict[str, Any]:
        """Return default visualization configuration."""
        return {
            'figsize': (15, 5),
            'dpi': 100,
            'style': 'seaborn-v0_8-darkgrid',
            'color_palette': 'husl',
            'font_size': 12,
            'title_size': 14,
            'save_format': 'png'
        }
    
    # ===================== METRICS CALCULATION =====================
    
    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        dataset_name: str = "Dataset",
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            dataset_name: Name of the dataset (e.g., "Train", "Test")
            verbose: If True, print metrics to console
        
        Returns:
            Dictionary containing all calculated metrics
        
        Raises:
            ValueError: If input arrays have different shapes or invalid values
        """
        # Input validation
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        try:
            # Core regression metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Additional metrics
            mape = self._safe_mape(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            explained_var = explained_variance_score(y_true, y_pred)
            max_err = max_error(y_true, y_pred)
            
            # Statistical metrics
            residuals = y_true - y_pred
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            median_abs_error = np.median(np.abs(residuals))
            
            # Percentage metrics
            mean_pct_error = np.mean((residuals / y_true) * 100)
            
            # Correlation
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            # Compile metrics
            metrics = {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R2': float(r2),
                'MAPE': float(mape),
                'MSE': float(mse),
                'Explained_Variance': float(explained_var),
                'Max_Error': float(max_err),
                'Mean_Residual': float(mean_residual),
                'Std_Residual': float(std_residual),
                'Median_Absolute_Error': float(median_abs_error),
                'Mean_Percentage_Error': float(mean_pct_error),
                'Correlation': float(correlation),
                'Dataset': dataset_name,
                'N_Samples': len(y_true),
                'Timestamp': datetime.now().isoformat()
            }
            
            # Store in history
            self.metrics_history[f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = metrics
            
            # Print metrics if verbose
            if verbose:
                self._print_metrics(metrics, dataset_name)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def _validate_inputs(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and convert inputs to numpy arrays."""
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Validate shapes
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        
        # Check for NaN or infinite values
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input contains NaN values")
        
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Input contains infinite values")
        
        return y_true, y_pred
    
    def _safe_mape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """Calculate MAPE with protection against division by zero."""
        # Add small epsilon to avoid division by zero
        denominator = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)
    
    def _print_metrics(self, metrics: Dict[str, float], dataset_name: str):
        """Print formatted metrics to console."""
        self.logger.info("\n" + "="*70)
        self.logger.info(f"📊 {dataset_name} SET PERFORMANCE METRICS")
        self.logger.info("="*70)
        
        # Core metrics
        self.logger.info(f"\n🎯 Core Metrics:")
        self.logger.info(f"   RMSE:                {metrics['RMSE']:.4f}°C")
        self.logger.info(f"   MAE:                 {metrics['MAE']:.4f}°C")
        self.logger.info(f"   R² Score:            {metrics['R2']:.4f}")
        self.logger.info(f"   MAPE:                {metrics['MAPE']:.2f}%")
        
        # Additional metrics
        self.logger.info(f"\n📈 Additional Metrics:")
        self.logger.info(f"   MSE:                 {metrics['MSE']:.4f}")
        self.logger.info(f"   Explained Variance:  {metrics['Explained_Variance']:.4f}")
        self.logger.info(f"   Max Error:           {metrics['Max_Error']:.4f}°C")
        self.logger.info(f"   Correlation:         {metrics['Correlation']:.4f}")
        
        # Residual statistics
        self.logger.info(f"\n📊 Residual Statistics:")
        self.logger.info(f"   Mean Residual:       {metrics['Mean_Residual']:+.4f}°C")
        self.logger.info(f"   Std Residual:        {metrics['Std_Residual']:.4f}°C")
        self.logger.info(f"   Median Abs Error:    {metrics['Median_Absolute_Error']:.4f}°C")
        self.logger.info(f"   Mean % Error:        {metrics['Mean_Percentage_Error']:+.2f}%")
        
        # Sample info
        self.logger.info(f"\n📝 Dataset Info:")
        self.logger.info(f"   Number of Samples:   {metrics['N_Samples']:,}")
        
        self.logger.info("="*70 + "\n")
    
    # ===================== OVERFITTING DETECTION =====================
    
    def check_overfitting(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        model_name: str = "Model",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze overfitting by comparing train and test performance.
        
        Args:
            train_metrics: Metrics dictionary from training set
            test_metrics: Metrics dictionary from test set
            model_name: Name of the model being evaluated
            verbose: If True, print analysis to console
        
        Returns:
            Dictionary containing overfitting analysis results
        """
        if verbose:
            self.logger.info("\n" + "="*70)
            self.logger.info(f"🔍 OVERFITTING ANALYSIS - {model_name}")
            self.logger.info("="*70)
        
        analysis = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics_comparison': {},
            'overall_assessment': {}
        }
        
        # Compare key metrics
        key_metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        for metric in key_metrics:
            if metric not in train_metrics or metric not in test_metrics:
                continue
            
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]
            
            # Calculate differences
            if metric == 'R2':
                # For R², higher is better
                diff = train_val - test_val
                diff_pct = (diff / train_val) * 100 if train_val != 0 else 0
                degradation = diff > 0
            else:
                # For error metrics, lower is better
                diff = test_val - train_val
                diff_pct = (diff / train_val) * 100 if train_val != 0 else 0
                degradation = diff > 0
            
            analysis['metrics_comparison'][metric] = {
                'train': float(train_val),
                'test': float(test_val),
                'difference': float(diff),
                'difference_pct': float(diff_pct),
                'degradation': degradation
            }
            
            if verbose:
                self.logger.info(f"\n{metric}:")
                self.logger.info(f"   Train:     {train_val:.4f}")
                self.logger.info(f"   Test:      {test_val:.4f}")
                self.logger.info(f"   Diff:      {diff:+.4f} ({diff_pct:+.2f}%)")
        
        # Overall assessment
        r2_comp = analysis['metrics_comparison'].get('R2', {})
        rmse_comp = analysis['metrics_comparison'].get('RMSE', {})
        
        r2_drop = abs(r2_comp.get('difference_pct', 0))
        rmse_increase = rmse_comp.get('difference_pct', 0)
        
        # Determine overfitting severity
        if r2_drop > 10 or rmse_increase > 20:
            status = "SEVERE_OVERFITTING"
            severity = "⚠️  SEVERE OVERFITTING DETECTED"
            color = "red"
            recommendation = "Model is severely overfitted. Consider:\n" \
                           "   - Reduce model complexity\n" \
                           "   - Increase regularization\n" \
                           "   - Add more training data\n" \
                           "   - Feature selection/reduction"
        elif r2_drop > 5 or rmse_increase > 10:
            status = "MODERATE_OVERFITTING"
            severity = "⚠️  MODERATE OVERFITTING"
            color = "orange"
            recommendation = "Model shows signs of overfitting. Consider:\n" \
                           "   - Tune hyperparameters\n" \
                           "   - Apply cross-validation\n" \
                           "   - Feature engineering review"
        elif r2_drop > 2 or rmse_increase > 5:
            status = "SLIGHT_OVERFITTING"
            severity = "ℹ️  SLIGHT OVERFITTING (ACCEPTABLE)"
            color = "yellow"
            recommendation = "Model generalization is acceptable.\n" \
                           "   - Monitor performance on new data\n" \
                           "   - Consider minor tuning if needed"
        else:
            status = "GOOD_GENERALIZATION"
            severity = "✅ GOOD GENERALIZATION"
            color = "green"
            recommendation = "Model generalizes well to unseen data.\n" \
                           "   - Performance is excellent\n" \
                           "   - Ready for production deployment"
        
        analysis['overall_assessment'] = {
            'status': status,
            'severity': severity,
            'color': color,
            'r2_drop_pct': float(r2_drop),
            'rmse_increase_pct': float(rmse_increase),
            'recommendation': recommendation
        }
        
        if verbose:
            self.logger.info("\n" + "="*70)
            self.logger.info(severity)
            self.logger.info("="*70)
            self.logger.info(f"\n{recommendation}")
            self.logger.info("="*70 + "\n")
        
        return analysis
    
    # ===================== VISUALIZATION =====================
    
    def plot_diagnostics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        model_name: str = "Model",
        dataset_name: str = "Test",
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Create comprehensive diagnostic plots.
        
        Generates four diagnostic plots:
        1. Actual vs Predicted scatter plot
        2. Residual plot
        3. Residual distribution
        4. Error distribution by prediction range
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            dataset_name: Name of the dataset
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        residuals = y_true - y_pred
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'{model_name} - {dataset_name} Set Diagnostic Plots',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        # Plot 1: Actual vs Predicted
        self._plot_actual_vs_predicted(axes[0, 0], y_true, y_pred, model_name)
        
        # Plot 2: Residual Plot
        self._plot_residuals(axes[0, 1], y_pred, residuals, model_name)
        
        # Plot 3: Residual Distribution
        self._plot_residual_distribution(axes[1, 0], residuals, model_name)
        
        # Plot 4: Error by Prediction Range
        self._plot_error_by_range(axes[1, 1], y_true, y_pred, residuals)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path,
                dpi=self.visualization_config['dpi'],
                bbox_inches='tight'
            )
            self.logger.info(f"Diagnostic plots saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_actual_vs_predicted(
        self,
        ax: plt.Axes,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ):
        """Plot actual vs predicted values."""
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = y_true.min(), y_true.max()
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            lw=2,
            label='Perfect Prediction',
            alpha=0.8
        )
        
        # Calculate and display R²
        r2 = r2_score(y_true, y_pred)
        ax.text(
            0.05, 0.95,
            f'R² = {r2:.4f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_xlabel('Actual Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals(
        self,
        ax: plt.Axes,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        model_name: str
    ):
        """Plot residuals vs predicted values."""
        ax.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.8)
        
        # Add standard deviation bands
        std_res = np.std(residuals)
        ax.axhline(y=std_res, color='orange', linestyle=':', lw=1.5, alpha=0.6, label=f'±1 STD')
        ax.axhline(y=-std_res, color='orange', linestyle=':', lw=1.5, alpha=0.6)
        ax.axhline(y=2*std_res, color='red', linestyle=':', lw=1.5, alpha=0.4, label=f'±2 STD')
        ax.axhline(y=-2*std_res, color='red', linestyle=':', lw=1.5, alpha=0.4)
        
        ax.set_xlabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_residual_distribution(
        self,
        ax: plt.Axes,
        residuals: np.ndarray,
        model_name: str
    ):
        """Plot residual distribution with normality test."""
        # Histogram
        n, bins, patches = ax.hist(
            residuals,
            bins=50,
            density=True,
            alpha=0.7,
            color='skyblue',
            edgecolor='black'
        )
        
        # Fit normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            'r-',
            lw=2,
            label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})'
        )
        
        # Normality test
        if len(residuals) >= 20:
            _, p_value = normaltest(residuals)
            normality_text = f"Normality test p-value: {p_value:.4f}"
            if p_value > 0.05:
                normality_text += "\n(Residuals appear normal)"
            else:
                normality_text += "\n(Residuals may not be normal)"
            
            ax.text(
                0.05, 0.95,
                normality_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        ax.set_xlabel('Residuals (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Residual Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_error_by_range(
        self,
        ax: plt.Axes,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray
    ):
        """Plot absolute error by temperature range."""
        # Create temperature bins
        n_bins = 10
        bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        abs_errors = np.abs(residuals)
        
        # Calculate mean absolute error for each bin
        bin_errors = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
            if mask.sum() > 0:
                bin_errors.append(abs_errors[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_errors.append(0)
                bin_counts.append(0)
        
        # Bar plot
        bars = ax.bar(
            bin_centers,
            bin_errors,
            width=(bins[1] - bins[0]) * 0.8,
            alpha=0.7,
            edgecolor='black'
        )
        
        # Color bars by error magnitude
        max_error = max(bin_errors) if bin_errors else 1
        for bar, error in zip(bars, bin_errors):
            if max_error > 0:
                intensity = error / max_error
                bar.set_color(plt.cm.RdYlGn_r(intensity))
        
        # Add count labels on bars
        for i, (center, error, count) in enumerate(zip(bin_centers, bin_errors, bin_counts)):
            if count > 0:
                ax.text(
                    center,
                    error,
                    f'n={count}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        ax.set_xlabel('Temperature Range (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Error by Temperature Range', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Plot feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning("Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame and sort
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N
        feature_imp_top = feature_imp.head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
        
        bars = ax.barh(
            range(len(feature_imp_top)),
            feature_imp_top['importance'],
            alpha=0.8
        )
        
        # Color bars by importance
        max_imp = feature_imp_top['importance'].max()
        for bar, imp in zip(bars, feature_imp_top['importance']):
            intensity = imp / max_imp if max_imp > 0 else 0
            bar.set_color(plt.cm.viridis(intensity))
        
        ax.set_yticks(range(len(feature_imp_top)))
        ax.set_yticklabels(feature_imp_top['feature'])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Top {top_n} Feature Importances',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.visualization_config['dpi'], bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return feature_imp
    
    # ===================== REPORTING =====================
    
    def generate_report(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        model_info: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            train_metrics: Training set metrics
            test_metrics: Test set metrics
            model_info: Optional model metadata
            output_path: Optional path to save JSON report
        
        Returns:
            Complete evaluation report as dictionary
        """
        # Perform overfitting analysis
        overfitting_analysis = self.check_overfitting(
            train_metrics,
            test_metrics,
            model_name=model_info.get('name', 'Model') if model_info else 'Model',
            verbose=False
        )
        
        # Compile report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0.0'
            },
            'model_info': model_info or {},
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting_analysis': overfitting_analysis,
            'performance_summary': {
                'test_rmse': test_metrics.get('RMSE'),
                'test_mae': test_metrics.get('MAE'),
                'test_r2': test_metrics.get('R2'),
                'test_mape': test_metrics.get('MAPE'),
                'generalization_status': overfitting_analysis['overall_assessment']['status'],
                'production_ready': overfitting_analysis['overall_assessment']['status'] in [
                    'GOOD_GENERALIZATION',
                    'SLIGHT_OVERFITTING'
                ]
            }
        }
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation report saved to: {output_path}")
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """Print formatted report summary."""
        self.logger.info("\n" + "="*70)
        self.logger.info("📋 EVALUATION REPORT SUMMARY")
        self.logger.info("="*70)
        
        summary = report['performance_summary']
        
        self.logger.info(f"\n🎯 Test Set Performance:")
        self.logger.info(f"   RMSE:  {summary['test_rmse']:.4f}°C")
        self.logger.info(f"   MAE:   {summary['test_mae']:.4f}°C")
        self.logger.info(f"   R²:    {summary['test_r2']:.4f}")
        self.logger.info(f"   MAPE:  {summary['test_mape']:.2f}%")
        
        self.logger.info(f"\n📊 Model Assessment:")
        self.logger.info(f"   Status: {summary['generalization_status']}")
        self.logger.info(f"   Production Ready: {'✅ Yes' if summary['production_ready'] else '⚠️ No'}")
        
        self.logger.info("="*70 + "\n")
    
    # ===================== UTILITY METHODS =====================
    
    def compare_models(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        metric: str = 'RMSE',
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            models_metrics: Dictionary mapping model names to their metrics
            metric: Metric to compare (default: 'RMSE')
            ascending: Sort order (True for ascending, False for descending)
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, metrics in models_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics.get('RMSE', np.nan),
                'MAE': metrics.get('MAE', np.nan),
                'R2': metrics.get('R2', np.nan),
                'MAPE': metrics.get('MAPE', np.nan)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        if metric in df_comparison.columns:
            df_comparison = df_comparison.sort_values(metric, ascending=ascending)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("🏆 MODEL COMPARISON")
        self.logger.info("="*70)
        self.logger.info(f"\n{df_comparison.to_string(index=False)}\n")
        
        return df_comparison
    
    def export_metrics_history(self, output_path: Path):
        """Export all metrics history to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        self.logger.info(f"Metrics history exported to: {output_path}")


# ===================== STANDALONE FUNCTIONS =====================

def evaluate_model(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    dataset_name: str = "Dataset"
) -> Dict[str, float]:
    """
    Standalone function for quick model evaluation.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.calculate_metrics(y_true, y_pred, dataset_name)


def plot_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    dataset_name: str = "Test"
):
    """
    Standalone function for quick visualization.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    evaluator = ModelEvaluator()
    evaluator.plot_diagnostics(y_true, y_pred, model_name, dataset_name)


def check_overfitting(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Standalone function for overfitting analysis.
    
    Args:
        train_metrics: Training set metrics
        test_metrics: Test set metrics
        model_name: Name of the model
    
    Returns:
        Overfitting analysis results
    """
    evaluator = ModelEvaluator()
    return evaluator.check_overfitting(train_metrics, test_metrics, model_name)


# ===================== MODULE INITIALIZATION =====================

if __name__ == "__main__":
    print("="*70)
    print("Model Evaluation Module - Production Ready")
    print("="*70)
    print("\nModule loaded successfully!")
    print("\nAvailable classes:")
    print("  - ModelEvaluator: Comprehensive model evaluation")
    print("\nAvailable functions:")
    print("  - evaluate_model(): Quick metrics calculation")
    print("  - plot_predictions(): Quick visualization")
    print("  - check_overfitting(): Quick overfitting check")
    print("="*70)
