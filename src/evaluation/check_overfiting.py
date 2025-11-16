def check_overfitting(train_metrics, test_metrics, model_name="Model"):
    """
    Kiểm tra overfitting bằng cách so sánh performance giữa train và test
    
    Parameters:
    -----------
    train_metrics : dict
        Metrics từ tập train
    test_metrics : dict
        Metrics từ tập test
    model_name : str
        Tên mô hình
        
    Returns:
    --------
    overfitting_report : dict
        Báo cáo về overfitting
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 OVERFITTING ANALYSIS - {model_name}")
    print(f"{'='*60}")
    
    overfitting_report = {}
    
    for metric in ['RMSE', 'MAE', 'R2', 'MAPE']:
        train_val = train_metrics[metric]
        test_val = test_metrics[metric]
        
        if metric == 'R2':
            # For R2, higher is better
            diff = train_val - test_val
            diff_pct = (diff / train_val) * 100 if train_val != 0 else 0
        else:
            # For RMSE, MAE, MAPE, lower is better
            diff = test_val - train_val
            diff_pct = (diff / train_val) * 100 if train_val != 0 else 0
        
        overfitting_report[metric] = {
            'train': train_val,
            'test': test_val,
            'difference': diff,
            'difference_pct': diff_pct
        }
        
        print(f"\n{metric}:")
        print(f"   Train: {train_val:.4f}")
        print(f"   Test:  {test_val:.4f}")
        print(f"   Diff:  {diff:+.4f} ({diff_pct:+.2f}%)")
    
    # Overall overfitting assessment
    print(f"\n{'='*60}")
    r2_drop = overfitting_report['R2']['difference_pct']
    rmse_increase = overfitting_report['RMSE']['difference_pct']
    
    if r2_drop > 10 or rmse_increase > 20:
        status = "⚠️  SEVERE OVERFITTING DETECTED"
        color = "red"
    elif r2_drop > 5 or rmse_increase > 10:
        status = "⚠️  MODERATE OVERFITTING"
        color = "orange"
    else:
        status = "✅ GOOD GENERALIZATION"
        color = "green"
    
    print(f"{status}")
    print(f"{'='*60}")
    
    overfitting_report['status'] = status
    overfitting_report['color'] = color
    
    return overfitting_report

print("Overfitting check function defined!")