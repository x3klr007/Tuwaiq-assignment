"""
Medical Insurance Cost Analysis - Production Utilities Module.

This module provides comprehensive data processing, statistical analysis,
and visualization utilities for the insurance cost prediction pipeline.

Author: Data Science Capstone
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union, Any
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set global visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# DATA CLEANING AND VALIDATION
# =============================================================================

def validate_data_quality(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None
) -> Dict[str, Union[bool, str, int]]:
    """
    Perform comprehensive data quality validation checks.
    
    Checks:
    1. Schema validation - all expected columns present
    2. Missing values - no nulls in dataset
    3. Data integrity - charges column has positive values
    
    Args:
        df: Input DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        Dictionary containing validation results
        
    Raises:
        ValueError: If input is not a DataFrame or is empty
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    results: Dict[str, Any] = {
        'passed_all': True,
        'checks': [],
        'details': ''
    }
    
    # Check 1: Schema validation
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        schema_passed = len(missing_cols) == 0
        results['checks'].append({
            'name': 'Schema Validation',
            'passed': schema_passed,
            'message': f"Missing columns: {missing_cols}" if not schema_passed else "All columns present"
        })
        if not schema_passed:
            results['passed_all'] = False
            results['details'] += f"Missing columns: {missing_cols}. "
    
    # Check 2: Missing values
    total_missing = df.isnull().sum().sum()
    missing_passed = total_missing == 0
    results['checks'].append({
        'name': 'Missing Values Check',
        'passed': missing_passed,
        'message': f"Total missing: {total_missing}" if not missing_passed else "No missing values"
    })
    if not missing_passed:
        results['passed_all'] = False
        results['details'] += f"Missing values detected: {total_missing}. "
    
    # Check 3: Data integrity (charges must be positive)
    if 'charges' in df.columns:
        negative_charges = (df['charges'] <= 0).sum()
        integrity_passed = negative_charges == 0
        results['checks'].append({
            'name': 'Data Integrity Check',
            'passed': integrity_passed,
            'message': f"Non-positive charges: {negative_charges}" if not integrity_passed else "All charges positive"
        })
        if not integrity_passed:
            results['passed_all'] = False
            results['details'] += f"Non-positive charges found: {negative_charges}. "
    
    return results


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print formatted validation report to console."""
    print("=" * 60)
    print("DATA QUALITY VALIDATION REPORT")
    print("=" * 60)
    for check in results['checks']:
        status = "[PASS]" if check['passed'] else "[FAIL]"
        print(f"{status} | {check['name']}: {check['message']}")
    print("=" * 60)
    if results['passed_all']:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: CHECKS FAILED - {results['details']}")
    print("=" * 60)


def clean_data(
    df: pd.DataFrame,
    outlier_columns: Optional[List[str]] = None,
    upper_percentile: float = 0.99,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data cleaning function.
    
    Cleaning steps:
    1. Handle missing values (median for numeric, mode for categorical)
    2. Remove duplicate rows
    3. Cap outliers at specified percentile
    4. Standardize data types
    
    Args:
        df: Raw input DataFrame
        outlier_columns: Columns to cap outliers on (default: ['charges'])
        upper_percentile: Upper percentile for capping (default: 0.99)
        verbose: Print progress information
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report dictionary)
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError("Cannot clean empty DataFrame")
    
    if verbose:
        print("\n" + "=" * 60)
        print("STARTING DATA CLEANING PIPELINE")
        print("=" * 60)
    
    df_clean = df.copy()
    cleaning_report: Dict[str, Any] = {
        'input_rows': len(df_clean),
        'steps_applied': []
    }
    
    # Step 1: Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    
    if missing_before > 0:
        # Numeric: fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                if verbose:
                    print(f"  Filled missing in '{col}' with median: {median_val:.2f}")
        
        # Categorical: fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                if verbose:
                    print(f"  Filled missing in '{col}' with mode: {mode_val}")
    
    missing_after = df_clean.isnull().sum().sum()
    cleaning_report['steps_applied'].append(
        f"Missing values: {missing_before} -> {missing_after}"
    )
    
    # Step 2: Remove duplicates
    duplicates_before = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = duplicates_before - df_clean.duplicated().sum()
    if duplicates_removed > 0:
        cleaning_report['steps_applied'].append(f"Duplicates removed: {duplicates_removed}")
    
    # Step 3: Cap outliers at 99th percentile
    if outlier_columns is None:
        outlier_columns = ['charges']
    
    outlier_stats: Dict[str, Dict[str, float]] = {}
    for col in outlier_columns:
        if col in df_clean.columns:
            upper_bound = df_clean[col].quantile(upper_percentile)
            values_above = (df_clean[col] > upper_bound).sum()
            df_clean[col] = df_clean[col].clip(upper=upper_bound)
            outlier_stats[col] = {
                'upper_bound': float(upper_bound),
                'values_capped': int(values_above)
            }
    
    cleaning_report['outlier_stats'] = outlier_stats
    cleaning_report['steps_applied'].append(
        f"Outliers capped at {upper_percentile*100:.0f}th percentile"
    )
    
    # Step 4: Standardize data types
    if 'age' in df_clean.columns:
        df_clean['age'] = df_clean['age'].astype(int)
    if 'children' in df_clean.columns:
        df_clean['children'] = df_clean['children'].astype(int)
    
    # Standardize categorical strings
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].str.lower().str.strip()
    
    cleaning_report['steps_applied'].append("Data types standardized")
    cleaning_report['output_rows'] = len(df_clean)
    cleaning_report['rows_removed'] = cleaning_report['input_rows'] - cleaning_report['output_rows']
    
    if verbose:
        print(f"\nCLEANING COMPLETE")
        print(f"  Input:  {cleaning_report['input_rows']} rows")
        print(f"  Output: {cleaning_report['output_rows']} rows")
        print(f"  Removed: {cleaning_report['rows_removed']} rows")
        print("  Steps Applied:")
        for step in cleaning_report['steps_applied']:
            print(f"    - {step}")
    
    return df_clean, cleaning_report


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using appropriate methods.
    
    Encoding strategy:
    - One-Hot: region (nominal, no order)
    - Binary: sex, smoker (binary categories)
    - Ordinal: bmi_category (ordered health categories)
    
    Args:
        df: Input DataFrame with categorical columns
        
    Returns:
        DataFrame with encoded features added
    """
    df_encoded = df.copy()
    
    # One-Hot Encoding for region (drop_first=True to avoid dummy variable trap)
    if 'region' in df_encoded.columns:
        region_dummies = pd.get_dummies(df_encoded['region'], prefix='region', drop_first=True)
        df_encoded = pd.concat([df_encoded, region_dummies], axis=1)
    
    # Binary encoding for sex
    if 'sex' in df_encoded.columns:
        df_encoded['sex_male'] = (df_encoded['sex'] == 'male').astype(int)
    
    # Binary encoding for smoker
    if 'smoker' in df_encoded.columns:
        df_encoded['is_smoker'] = (df_encoded['smoker'] == 'yes').astype(int)
    
    # Ordinal encoding for BMI categories
    if 'bmi' in df_encoded.columns:
        def categorize_bmi(bmi: float) -> int:
            if bmi < 18.5:
                return 0  # underweight
            elif bmi < 25:
                return 1  # normal
            elif bmi < 30:
                return 2  # overweight
            else:
                return 3  # obese
        
        df_encoded['bmi_category'] = df_encoded['bmi'].apply(categorize_bmi)
    
    return df_encoded


def create_interaction_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Create domain-specific interaction features.
    
    CRITICAL: This function does NOT use 'charges' (target) to create features
    to prevent data leakage. All features are computable at inference time
    without knowing the target value.
    
    New features:
    - age_bmi_risk: age * bmi (compound risk factor)
    - bmi_smoker_interaction: bmi * is_smoker (health risk interaction)
    - family_size: children + 1 (household scale)
    - age_squared: age^2 (non-linear age effect)
    - bmi_squared: bmi^2 (non-linear BMI effect)
    
    Args:
        df: Input DataFrame with base features
        is_training: Whether this is training phase (for target transform)
        
    Returns:
        DataFrame with interaction features added (no target leakage)
    """
    df_features = df.copy()
    
    # Age-BMI risk compound (no target leakage)
    if 'age' in df_features.columns and 'bmi' in df_features.columns:
        df_features['age_bmi_risk'] = df_features['age'] * df_features['bmi']
    
    # BMI-Smoker interaction (no target leakage)
    if 'bmi' in df_features.columns and 'is_smoker' in df_features.columns:
        df_features['bmi_smoker_interaction'] = df_features['bmi'] * df_features['is_smoker']
    
    # Family size (no target leakage)
    if 'children' in df_features.columns:
        df_features['family_size'] = df_features['children'] + 1
    
    # Non-linear age effect (no target leakage)
    if 'age' in df_features.columns:
        df_features['age_squared'] = df_features['age'] ** 2
    
    # Non-linear BMI effect (no target leakage)
    if 'bmi' in df_features.columns:
        df_features['bmi_squared'] = df_features['bmi'] ** 2
    
    # Log transform target for modeling (only if charges exists - for training)
    if is_training and 'charges' in df_features.columns:
        df_features['charges_log'] = np.log1p(df_features['charges'])
    
    return df_features


def calculate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for multicollinearity detection.
    
    VIF Interpretation:
    - VIF < 5: Low multicollinearity (acceptable)
    - VIF 5-10: Moderate multicollinearity (monitor)
    - VIF > 10: High multicollinearity (remove feature)
    
    Args:
        df: DataFrame containing features
        features: List of feature column names
        
    Returns:
        DataFrame with columns ['Feature', 'VIF', 'Concern_Level']
        Sorted by VIF in descending order
        
    Raises:
        ImportError: If statsmodels is not installed
        ValueError: If insufficient samples for calculation
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        raise ImportError("statsmodels required. Install: pip install statsmodels")
    
    X = df[features].dropna()
    
    if len(X) < len(features) + 1:
        raise ValueError(f"Need >= {len(features) + 1} samples, got {len(X)}")
    
    # Add constant for intercept
    X_with_const = X.assign(constant=1)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                       for i in range(len(features))]
    
    def interpret_vif(vif: float) -> str:
        if vif < 5:
            return "Low (OK)"
        elif vif < 10:
            return "Moderate"
        else:
            return "High (Remove)"
    
    vif_data["Concern_Level"] = vif_data["VIF"].apply(interpret_vif)
    
    return vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def manual_numpy_stats(arr: np.ndarray) -> Dict[str, Union[float, int]]:
    """
    Calculate mean and standard deviation manually using NumPy.
    
    Uses sample standard deviation with Bessel's correction (n-1 denominator).
    
    Args:
        arr: Input array of numeric values (1-D)
        
    Returns:
        Dictionary with statistics including manual and numpy-calculated values
        
    Raises:
        ValueError: If array is empty or has fewer than 2 elements
        TypeError: If array contains non-numeric values
    """
    arr = np.asarray(arr)
    
    if arr.size == 0:
        raise ValueError("Cannot compute statistics on empty array")
    
    if arr.size < 2:
        raise ValueError("Need >= 2 elements for sample standard deviation")
    
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("Array must contain numeric values")
    
    # Handle NaN values
    if np.any(np.isnan(arr)):
        original_size = arr.size
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            raise ValueError(f"After removing NaN, only {arr.size} elements remain")
        print(f"Warning: Removed {original_size - arr.size} NaN values")
    
    n = len(arr)
    
    # Manual mean: E[X] = sum(x) / n
    mean_val = np.sum(arr) / n
    
    # Manual std: s = sqrt(sum((x - mean)^2) / (n-1))
    # Bessel's correction uses (n-1) for sample standard deviation
    squared_diff = np.sum((arr - mean_val) ** 2)
    std_val = np.sqrt(squared_diff / (n - 1))
    
    return {
        'mean_manual': float(mean_val),
        'std_manual': float(std_val),
        'mean_numpy': float(np.mean(arr)),
        'std_numpy': float(np.std(arr, ddof=1)),
        'sample_size': int(n)
    }


def manual_zscore_standardization(
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Z-score standardization manually using NumPy broadcasting.
    
    Formula: z = (X - mean) / std
    
    Args:
        X: Input array (n_samples, n_features)
        
    Returns:
        Tuple of (X_scaled, mean_vector, std_vector)
        
    Raises:
        ValueError: If any feature has zero std
    """
    X = np.asarray(X)
    
    # Calculate mean for each feature (axis=0 for column-wise)
    mean_vec = np.sum(X, axis=0) / X.shape[0]
    
    # Calculate std for each feature
    std_vec = np.sqrt(np.sum((X - mean_vec) ** 2, axis=0) / X.shape[0])
    
    # Check for zero std
    if np.any(std_vec == 0):
        zero_features = np.where(std_vec == 0)[0]
        raise ValueError(f"Zero std detected in features: {zero_features}")
    
    # Z-score using broadcasting
    X_scaled = (X - mean_vec) / std_vec
    
    return X_scaled, mean_vec, std_vec


def cosine_similarity_manual(
    vec1: np.ndarray,
    vec2: np.ndarray,
    scale_features: bool = False
) -> float:
    """
    Calculate cosine similarity between two vectors manually.
    
    Formula: cos(theta) = (A . B) / (||A|| * ||B||)
    
    IMPORTANT: For vectors with different scales, use scale_features=True.
    
    Args:
        vec1: First vector (1-D array)
        vec2: Second vector (1-D array)
        scale_features: If True, standardize vectors before comparison
        
    Returns:
        Cosine similarity between -1 and 1
        Returns 0.0 if either vector is a zero vector (guard against NaN)
        
    Raises:
        ValueError: If vectors have different shapes
    """
    vec1 = np.asarray(vec1, dtype=float)
    vec2 = np.asarray(vec2, dtype=float)
    
    if vec1.shape != vec2.shape:
        raise ValueError(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")
    
    if vec1.ndim != 1:
        raise ValueError(f"Vectors must be 1-D, got {vec1.ndim}D")
    
    # Optional scaling
    if scale_features:
        from sklearn.preprocessing import StandardScaler
        combined = np.vstack([vec1, vec2])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined.T).T
        vec1, vec2 = scaled[0], scaled[1]
    
    # Dot product
    dot_product = np.dot(vec1, vec2)
    
    # Magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    # GUARD AGAINST ZERO VECTORS - return 0.0 instead of NaN
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Cosine similarity with clipping for floating point errors
    cos_sim = dot_product / (mag1 * mag2)
    return float(np.clip(cos_sim, -1.0, 1.0))


# =============================================================================
# MODEL VALIDATION AND RESIDUAL ANALYSIS
# =============================================================================

def calculate_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Calculate residual metrics for model evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with residuals and metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    
    standardized = residuals / rmse if rmse > 0 else np.zeros_like(residuals)
    
    return {
        'residuals': residuals,
        'standardized_residuals': standardized,
        'mse': float(mse),
        'rmse': float(rmse),
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals, ddof=1))
    }


def plot_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive residual analysis plots (2x2 grid).
    
    Plots:
    1. Residuals vs Fitted (homoscedasticity check)
    2. Q-Q Plot (normality check)
    3. Histogram of residuals (distribution)
    4. Standardized residuals (outlier detection)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        model_name: Name for plot titles
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    resid_data = calculate_residuals(y_true, y_pred)
    residuals = resid_data['residuals']
    standardized = resid_data['standardized_residuals']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Residual Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Residuals vs Fitted\n(Homoscedasticity Check)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot\n(Normality Check)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, 
                    density=True, color='steelblue')
    if len(residuals) > 1:
        mu, std = stats.norm.fit(residuals)
        x = np.linspace(axes[1, 0].get_xlim()[0], axes[1, 0].get_xlim()[1], 100)
        axes[1, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2,
                       label=f'Normal Fit')
    axes[1, 0].set_xlabel('Residual Value', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('Residual Distribution', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Standardized Residuals
    axes[1, 1].scatter(y_pred, standardized, alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axhline(y=2, color='orange', linestyle=':', linewidth=1.5, label='+/- 2 sigma')
    axes[1, 1].axhline(y=-2, color='orange', linestyle=':', linewidth=1.5)
    axes[1, 1].axhline(y=3, color='darkred', linestyle=':', linewidth=1.5, label='+/- 3 sigma')
    axes[1, 1].axhline(y=-3, color='darkred', linestyle=':', linewidth=1.5)
    axes[1, 1].set_xlabel('Fitted Values', fontsize=11)
    axes[1, 1].set_ylabel('Standardized Residuals', fontsize=11)
    axes[1, 1].set_title('Standardized Residuals\n(Outlier Detection)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add summary text
    outlier_count = np.sum(np.abs(standardized) > 3)
    summary = f"Mean: {resid_data['mean_residual']:.2f}\n"
    summary += f"RMSE: {resid_data['rmse']:.2f}\n"
    summary += f"Outliers: {outlier_count} ({100*outlier_count/len(residuals):.1f}%)"
    axes[1, 1].text(0.02, 0.98, summary, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def analyze_model_bias_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    group_name: str = "Group"
) -> pd.DataFrame:
    """
    Analyze model bias across different groups using T-test.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        group_labels: Group membership array
        group_name: Name of grouping variable
        
    Returns:
        DataFrame with metrics by group
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    group_labels = np.asarray(group_labels).flatten()
    
    results = []
    
    for group in np.unique(group_labels):
        mask = group_labels == group
        group_residuals = y_true[mask] - y_pred[mask]
        
        results.append({
            group_name: group,
            'count': int(np.sum(mask)),
            'mean_actual': float(np.mean(y_true[mask])),
            'mean_predicted': float(np.mean(y_pred[mask])),
            'mean_residual': float(np.mean(group_residuals)),
            'std_residual': float(np.std(group_residuals, ddof=1)),
            'rmse': float(np.sqrt(np.mean(group_residuals ** 2))),
            'mae': float(np.mean(np.abs(group_residuals)))
        })
    
    return pd.DataFrame(results)


def perform_residual_ttest(
    residuals_group1: np.ndarray,
    residuals_group2: np.ndarray,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2"
) -> Dict[str, Union[float, str]]:
    """
    Perform T-test to compare residuals between two groups.
    
    Tests if model has systematic bias between groups.
    
    Args:
        residuals_group1: Residuals from first group
        residuals_group2: Residuals from second group
        group1_name: Name of first group
        group2_name: Name of second group
        
    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    t_stat, p_value = stats.ttest_ind(residuals_group1, residuals_group2)
    
    if p_value < 0.05:
        interpretation = f"BIAS DETECTED: Model treats {group1_name} and {group2_name} differently"
    else:
        interpretation = "No significant bias detected between groups"
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'interpretation': interpretation
    }


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def create_dashboard_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create 2x2 dashboard summarizing the dataset.
    
    Panels:
    1. Key metrics summary (text)
    2. Charges by smoker status (heatmap)
    3. Age vs BMI scatter (colored by charges)
    4. Distribution comparison (smoker vs non-smoker)
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Insurance Cost Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Panel 1: Key Metrics
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    metrics_text = f"""
    KEY METRICS SUMMARY
    ===================
    
    Dataset Overview:
      Total Records: {len(df):,}
      Average Charges: ${df['charges'].mean():,.2f}
      Median Charges: ${df['charges'].median():,.2f}
    
    Demographics:
      Average Age: {df['age'].mean():.1f} years
      Average BMI: {df['bmi'].mean():.1f}
      Smokers: {(df['smoker'] == 'yes').mean()*100:.1f}%
    """
    
    ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax1.set_title('Panel 1: Executive Summary', fontsize=13, fontweight='bold')
    
    # Panel 2: Heatmap
    ax2 = axes[0, 1]
    if 'smoker' in df.columns and 'region' in df.columns:
        pivot = df.pivot_table(values='charges', index='smoker', columns='region', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Panel 2: Avg Charges by Smoker & Region', fontsize=13, fontweight='bold')
    
    # Panel 3: Scatter
    ax3 = axes[1, 0]
    if 'age' in df.columns and 'bmi' in df.columns:
        scatter = ax3.scatter(df['age'], df['bmi'], c=df['charges'], 
                            cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, ax=ax3, label='Charges ($)')
        ax3.set_xlabel('Age (years)')
        ax3.set_ylabel('BMI (kg/m^2)')
        ax3.set_title('Panel 3: Age vs BMI (Colored by Charges)', fontsize=13, fontweight='bold')
    
    # Panel 4: Distribution
    ax4 = axes[1, 1]
    if 'smoker' in df.columns:
        smokers = df[df['smoker'] == 'yes']['charges']
        non_smokers = df[df['smoker'] == 'no']['charges']
        ax4.hist(non_smokers, bins=30, alpha=0.6, label='Non-Smoker', density=True, color='steelblue')
        ax4_twin = ax4.twinx()
        ax4_twin.hist(smokers, bins=30, alpha=0.6, label='Smoker', density=True, color='crimson')
        ax4.set_xlabel('Charges ($)')
        ax4.set_ylabel('Density (Non-Smokers)', color='steelblue')
        ax4_twin.set_ylabel('Density (Smokers)', color='crimson')
        ax4.set_title('Panel 4: Charges Distribution by Smoking Status', fontsize=13, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
    model_type: str = 'tree'
) -> pd.DataFrame:
    """
    Extract and format feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        model_type: 'tree' for tree-based, 'linear' for linear models
        
    Returns:
        DataFrame with feature importance sorted by importance
    """
    if model_type == 'tree':
        importance = model.feature_importances_
    elif model_type == 'linear':
        importance = np.abs(model.coef_)
    else:
        raise ValueError("model_type must be 'tree' or 'linear'")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def save_and_show(fig: plt.Figure, filename: str, dpi: int = 150) -> None:
    """Save figure and display it."""
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")
