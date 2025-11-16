from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(y_true, y_pred, dataset_name="Dataset"):
    """
    Đánh giá hiệu suất mô hình với các metrics phổ biến
    
    Parameters:
    -----------
    y_true : array-like
        Giá trị thực tế
    y_pred : array-like
        Giá trị dự đoán
    dataset_name : str
        Tên của dataset (Train/Test)
        
    Returns:
    --------
    metrics : dict
        Dictionary chứa các metrics
    """
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    print(f"\n📊 {dataset_name} Performance:")
    print(f"   RMSE: {rmse:.4f}°C")
    print(f"   MAE:  {mae:.4f}°C")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return metrics

print("Evaluation function defined!")

def plot_predictions(y_true, y_pred, model_name="Model", dataset_name="Test"):
    """
    Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
    
    Parameters:
    -----------
    y_true : array-like
        Giá trị thực tế
    y_pred : array-like
        Giá trị dự đoán
    model_name : str
        Tên mô hình
    dataset_name : str
        Tên dataset
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted scatter
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Temperature (°C)', fontsize=12)
    axes[0].set_ylabel('Predicted Temperature (°C)', fontsize=12)
    axes[0].set_title(f'{model_name} - {dataset_name} Set\nActual vs Predicted', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Temperature (°C)', fontsize=12)
    axes[1].set_ylabel('Residuals (°C)', fontsize=12)
    axes[1].set_title(f'{model_name} - {dataset_name} Set\nResidual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Residual statistics
    print(f"\n📊 Residual Statistics:")
    print(f"   Mean:   {residuals.mean():.4f}°C")
    print(f"   Std:    {residuals.std():.4f}°C")
    print(f"   Min:    {residuals.min():.4f}°C")
    print(f"   Max:    {residuals.max():.4f}°C")

print("Visualization function defined!")

