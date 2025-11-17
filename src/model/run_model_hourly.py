"""
Hourly Temperature Prediction Model Training Pipeline

This production-ready script orchestrates the complete machine learning pipeline for
multi-horizon hourly temperature forecasting using XGBoost models with walk-forward
cross-validation and Optuna hyperparameter optimization.

Features:
    - Automated data loading and preprocessing
    - Walk-forward cross-validation for time series
    - Bayesian hyperparameter optimization with Optuna
    - Multi-horizon forecasting (1-5 hours ahead)
    - Comprehensive model evaluation and metrics tracking
    - Automated model persistence and result logging
    - Production-ready error handling and logging

Usage:
    python src/model/run_model_hourly.py
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings

import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the training pipeline."""
    
    # Model parameters
    N_SPLITS = 5
    TEST_SIZE = 365
    CV_MODE = "expanding"
    HORIZONS = (1, 2, 3, 4, 5)
    N_TRIALS = 50
    
    # Data parameters
    DATE_COL = "datetime"
    TARGET_COL = "temp"
    EXCLUDE_COLS = ["datetime", "temp", "temp_next", "datetime_next"]
    
    # Model metadata
    MODEL_NAME = "Hanoi Temp Predictor v1.0 (XGBoost)"
    MODEL_VERSION = "1.0.0"
    
    # File paths (relative to project root)
    DATA_FILE = "dataset/raw/Hanoi_Hourly_Selected.csv"
    MODELS_DIR = "src/config/models_pkl"
    RESULTS_DIR = "src/config/results"


# LOGGING SETUP
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# PATH MANAGEMENT
def setup_project_paths() -> Tuple[Path, Path]:
    """
    Configure project root and source paths dynamically.
    
    Returns:
        Tuple of (PROJECT_ROOT, SRC_PATH) as Path objects
    
    Raises:
        RuntimeError: If project structure is invalid
    """
    try:
        current_file = Path(__file__).resolve()
        PROJECT_ROOT = current_file.parent.parent.parent
        SRC_PATH = PROJECT_ROOT / "src"
        
        # Validate paths exist
        if not PROJECT_ROOT.exists():
            raise RuntimeError(f"Project root does not exist: {PROJECT_ROOT}")
        if not SRC_PATH.exists():
            raise RuntimeError(f"Source path does not exist: {SRC_PATH}")
        
        # Add source path to sys.path
        if str(SRC_PATH) not in sys.path:
            sys.path.insert(0, str(SRC_PATH))
        
        return PROJECT_ROOT, SRC_PATH
    
    except Exception as e:
        raise RuntimeError(f"Failed to setup project paths: {str(e)}")



def load_and_preprocess_data(
    data_path: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data and apply preprocessing pipeline.
    
    Args:
        data_path: Path to the raw CSV data file
        logger: Logger instance for output
    
    Returns:
        Tuple of (train_dataset, test_dataset) as DataFrames
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data preprocessing fails
    """
    logger.info("="*80)
    logger.info("DATA LOADING AND PREPROCESSING")
    logger.info("="*80)
    
    # Import after path setup
    import data.Pipeline as pipeline_module
    
    # Load data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    df_hourly = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df_hourly):,} records")
    
    # Train-test split
    logger.info("Splitting data into train and test sets...")
    train_dataset, test_dataset = pipeline_module.train_test_split_hourly(df_hourly)
    logger.info(f"Train set: {len(train_dataset):,} records")
    logger.info(f"Test set: {len(test_dataset):,} records")
    
    # Apply preprocessing pipeline
    logger.info("Applying preprocessing pipeline...")
    preprocessing_pipeline = pipeline_module.before_model_pipeline_hourly
    preprocessing_pipeline.fit(train_dataset)
    
    train_dataset = preprocessing_pipeline.transform(train_dataset).reset_index()
    test_dataset = preprocessing_pipeline.transform(test_dataset).reset_index()
    
    logger.info("Preprocessing completed successfully")
    logger.info(f"Final train shape: {train_dataset.shape}")
    logger.info(f"Final test shape: {test_dataset.shape}")
    
    return train_dataset, test_dataset


def prepare_features(
    df: pd.DataFrame,
    exclude_cols: list,
    logger: logging.Logger
) -> list:
    """
    Extract feature columns from dataframe.
    
    Args:
        df: Input dataframe
        exclude_cols: List of columns to exclude
        logger: Logger instance
    
    Returns:
        List of feature column names
    """
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Identified {len(feature_cols)} feature columns")
    logger.debug(f"Features: {feature_cols}")
    
    return feature_cols


# MODEL TRAINING
def initialize_pipeline(
    train_data: pd.DataFrame,
    config: Config,
    feature_cols: list,
    logger: logging.Logger
) -> Any:
    """
    Initialize and configure the XGBoost training pipeline.
    
    Args:
        train_data: Training dataset
        config: Configuration object
        feature_cols: List of feature columns
        logger: Logger instance
    
    Returns:
        Initialized pipeline object
    """
    logger.info("="*80)
    logger.info("MODEL INITIALIZATION")
    logger.info("="*80)
    
    import model.hourly_model as hourly_model
    
    logger.info("Creating multi-horizon XGBoost pipeline...")
    pipeline = hourly_model.MultiHorizonHourly_WalkForwardOptuna_XGBoost_Pipeline(
        df=train_data,
        date_col=config.DATE_COL,
        target_col=config.TARGET_COL,
        feature_cols=feature_cols,
        n_splits=config.N_SPLITS,
        test_size=config.TEST_SIZE,
        mode=config.CV_MODE,
        horizons=config.HORIZONS
    )
    
    logger.info(f"Pipeline configured:")
    logger.info(f"  - Horizons: {config.HORIZONS}")
    logger.info(f"  - CV splits: {config.N_SPLITS}")
    logger.info(f"  - CV mode: {config.CV_MODE}")
    logger.info(f"  - Test size: {config.TEST_SIZE}")
    
    return pipeline


def train_models(
    pipeline: Any,
    config: Config,
    logger: logging.Logger
) -> Any:
    """
    Execute the complete model training workflow.
    
    Args:
        pipeline: Initialized pipeline object
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Trained pipeline object
    """
    # Feature engineering
    logger.info("="*80)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*80)
    
    logger.info("Adding target shifts for multi-horizon forecasting...")
    pipeline.add_target_shifts()
    logger.info("Target shifts created successfully")
    
    logger.info("Creating walk-forward cross-validation folds...")
    pipeline.create_walkforward_folds()
    logger.info("CV folds created successfully")
    
    # Hyperparameter optimization
    logger.info("="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    
    logger.info(f"Starting Optuna optimization with {config.N_TRIALS} trials...")
    logger.info("This process may take several minutes to hours depending on data size...")
    
    pipeline.run_optuna(n_trials=config.N_TRIALS)
    logger.info(f"Optimization completed ({config.N_TRIALS} trials)")
    
    # Final model training
    logger.info("="*80)
    logger.info("FINAL MODEL TRAINING")
    logger.info("="*80)
    
    logger.info("Training final models with optimized hyperparameters...")
    pipeline.train_final_models()
    logger.info(f"Successfully trained {len(config.HORIZONS)} models (one per horizon)")
    
    return pipeline


# MODEL EVALUATION
def evaluate_models(
    pipeline: Any,
    test_data: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[Dict, Dict]:
    """
    Evaluate trained models on train and test datasets.
    
    Args:
        pipeline: Trained pipeline object
        test_data: Test dataset
        logger: Logger instance
    
    Returns:
        Tuple of (train_metrics, test_metrics) dictionaries
    """
    logger.info("="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    # Training set evaluation
    logger.info("\nEvaluating on TRAINING set...")
    train_metrics = pipeline.evaluate_train_models()
    
    logger.info("\nTraining Set Performance:")
    logger.info("-" * 60)
    for horizon, metrics in sorted(train_metrics.items()):
        rmse = metrics.get('rmse', float('nan'))
        mae = metrics.get('mae', float('nan'))
        r2 = metrics.get('r2', float('nan'))
        logger.info(f"  Horizon {horizon}h | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    
    # Test set evaluation
    logger.info("\nEvaluating on TEST set...")
    test_df_shifted = pipeline.prepare_test_dataset(test_data)
    test_metrics = pipeline.evaluate_final_models(test_df_shifted)
    
    logger.info("\nTest Set Performance:")
    logger.info("-" * 60)
    for horizon, metrics in sorted(test_metrics.items()):
        rmse = metrics.get('rmse', float('nan'))
        mae = metrics.get('mae', float('nan'))
        r2 = metrics.get('r2', float('nan'))
        logger.info(f"  Horizon {horizon}h | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    
    return train_metrics, test_metrics



# RESULT PERSISTENCE
def save_results(
    pipeline: Any,
    train_metrics: Dict,
    test_metrics: Dict,
    project_root: Path,
    config: Config,
    logger: logging.Logger
) -> Tuple[str, Path]:
    """
    Save trained models and results to disk.
    
    Args:
        pipeline: Trained pipeline object
        train_metrics: Training metrics dictionary
        test_metrics: Test metrics dictionary
        project_root: Project root directory path
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Tuple of (run_name, summary_path)
    """
    logger.info("="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    # Generate unique run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hanoi_temp_v1_{timestamp}"
    
    # Setup directories
    models_dir = project_root / config.MODELS_DIR
    results_dir = project_root / config.RESULTS_DIR
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model artifacts
    logger.info(f"\nSaving model files to: {models_dir}")
    model_paths = pipeline.save_all_final_models_pkl(
        base_filename_prefix=str(models_dir / run_name)
    )
    logger.info(f"Saved {len(model_paths)} model file(s):")
    for path in model_paths:
        logger.info(f"  - {Path(path).name}")
    
    # Save results summary
    summary_path = results_dir / f"{run_name}_summary.json"
    logger.info(f"\nSaving results summary to: {summary_path}")
    
    pipeline.save_results_to_json(
        filename=str(summary_path),
        model_name=config.MODEL_NAME,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        saved_model_paths=model_paths
    )
    
    logger.info(f"Results successfully saved: {summary_path.name}")
    
    return run_name, summary_path


# MAIN EXECUTION
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train multi-horizon hourly temperature prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python src/model/run_model_hourly.py
    python src/model/run_model_hourly.py --n-trials 100 --log-level DEBUG
    python src/model/run_model_hourly.py --data-file dataset/raw/custom_data.csv
        """
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=Config.N_TRIALS,
        help=f'Number of Optuna trials (default: {Config.N_TRIALS})'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=Config.DATA_FILE,
        help=f'Path to data file relative to project root (default: {Config.DATA_FILE})'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--cv-splits',
        type=int,
        default=Config.N_SPLITS,
        help=f'Number of cross-validation splits (default: {Config.N_SPLITS})'
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main execution function for the training pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Display pipeline header
        logger.info("\n" + "="*80)
        logger.info("HOURLY TEMPERATURE PREDICTION PIPELINE")
        logger.info(f"Version: {Config.MODEL_VERSION}")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        # Setup paths
        logger.info("\nInitializing environment...")
        PROJECT_ROOT, SRC_PATH = setup_project_paths()
        logger.info(f"Project root: {PROJECT_ROOT}")
        logger.info(f"Source path: {SRC_PATH}")
        
        # Update config with CLI arguments
        config = Config()
        config.N_TRIALS = args.n_trials
        config.N_SPLITS = args.cv_splits
        config.DATA_FILE = args.data_file
        
        # Load and preprocess data
        data_path = PROJECT_ROOT / config.DATA_FILE
        train_data, test_data = load_and_preprocess_data(data_path, logger)
        
        # Prepare features
        feature_cols = prepare_features(train_data, config.EXCLUDE_COLS, logger)
        
        # Initialize pipeline
        pipeline = initialize_pipeline(train_data, config, feature_cols, logger)
        
        # Train models
        pipeline = train_models(pipeline, config, logger)
        
        # Evaluate models
        train_metrics, test_metrics = evaluate_models(pipeline, test_data, logger)
        
        # Save results
        run_name, summary_path = save_results(
            pipeline,
            train_metrics,
            test_metrics,
            PROJECT_ROOT,
            config,
            logger
        )
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Run name: {run_name}")
        logger.info(f"Models saved: {len(config.HORIZONS)}")
        logger.info(f"Results: {summary_path}")
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Please check that all required files exist in the correct locations.")
        return 1
    
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        logger.error("Please check your input data and configuration.")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

