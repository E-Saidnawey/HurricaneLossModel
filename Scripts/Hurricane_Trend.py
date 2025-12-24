"""
Time Series Analysis: Hurricane Frequency Decomposition
========================================================

Analyzes trends, seasonality, and residuals in annual hurricane counts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from load_data import load_hurricane_data
import os


def analyze_hurricane_timeseries(years, counts, save_dir='Results/TimeSeries'):
    """
    Perform time series decomposition and trend analysis on hurricane counts.
    
    Args:
        years: Array of years
        counts: Array of hurricane counts per year
        save_dir: Directory to save output plots
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create pandas Series with year as index
    ts = pd.Series(counts, index=pd.to_datetime(years, format='%Y'))
    
    print("="*80)
    print("TIME SERIES ANALYSIS: HURRICANE FREQUENCY")
    print("="*80)
    print(f"\nData range: {years[0]} - {years[-1]}")
    print(f"Number of years: {len(years)}")
    print(f"Mean frequency: {np.mean(counts):.2f} hurricanes/year")
    print(f"Std deviation: {np.std(counts):.2f}")
    
    # ========================================================================
    # 1. STATIONARITY TEST (Augmented Dickey-Fuller)
    # ========================================================================
    print("\n" + "-"*80)
    print("STATIONARITY TEST (Augmented Dickey-Fuller)")
    print("-"*80)
    
    adf_result = adfuller(counts)
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in adf_result[4].items():
        print(f"  {key}: {value:.4f}")
    
    if adf_result[1] < 0.05:
        print("\n✓ Series is STATIONARY (p < 0.05)")
        print("  → No significant trend detected by ADF test")
    else:
        print("\n⚠ Series is NON-STATIONARY (p >= 0.05)")
        print("  → May contain trend or random walk component")
    
    # ========================================================================
    # 2. TREND ANALYSIS (Linear Regression)
    # ========================================================================
    print("\n" + "-"*80)
    print("TREND ANALYSIS (Linear Regression)")
    print("-"*80)
    
    # Fit linear trend
    X = years - years[0]  # Time since start
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, counts)
    
    trend_line = slope * X + intercept
    
    print(f"Slope: {slope:.4f} hurricanes/year")
    print(f"Intercept: {intercept:.2f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        if slope > 0:
            print(f"\n✓ SIGNIFICANT INCREASING TREND (p < 0.05)")
            print(f"  → Frequency increasing by {slope:.4f} hurricanes/year")
            print(f"  → Total increase over period: {slope * len(X):.2f} hurricanes")
        else:
            print(f"\n✓ SIGNIFICANT DECREASING TREND (p < 0.05)")
            print(f"  → Frequency decreasing by {abs(slope):.4f} hurricanes/year")
    else:
        print(f"\n✗ NO SIGNIFICANT LINEAR TREND (p >= 0.05)")
        print(f"  → Data consistent with stationary process")
    
    # ========================================================================
    # 3. TIME SERIES DECOMPOSITION
    # ========================================================================
    print("\n" + "-"*80)
    print("TIME SERIES DECOMPOSITION")
    print("-"*80)
    
    # Perform decomposition (using additive model with appropriate period)
    # For annual data with potential multi-year cycles, use period based on data length
    period = min(10, len(counts) // 3)  # Reasonable period for decomposition
    
    print(f"Using period: {period} years for decomposition")
    
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')
        
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        print("\nDecomposition components:")
        print(f"  Trend variance: {np.nanvar(trend):.2f}")
        print(f"  Seasonal variance: {np.nanvar(seasonal):.2f}")
        print(f"  Residual variance: {np.nanvar(residual):.2f}")
        
        # Calculate variance contributions
        total_var = np.var(counts)
        trend_contribution = np.nanvar(trend) / total_var * 100
        seasonal_contribution = np.nanvar(seasonal) / total_var * 100
        residual_contribution = np.nanvar(residual) / total_var * 100
        
        print(f"\nVariance contribution:")
        print(f"  Trend: {trend_contribution:.1f}%")
        print(f"  Seasonal: {seasonal_contribution:.1f}%")
        print(f"  Residual: {residual_contribution:.1f}%")
        
        decomp_success = True
        
    except Exception as e:
        print(f"\n⚠ Decomposition failed: {e}")
        print("  → Proceeding with trend analysis only")
        decomp_success = False
    
    # ========================================================================
    # 4. MOVING AVERAGES
    # ========================================================================
    print("\n" + "-"*80)
    print("MOVING AVERAGES")
    print("-"*80)
    
    ma_5 = pd.Series(counts).rolling(window=5, center=True).mean()
    ma_10 = pd.Series(counts).rolling(window=10, center=True).mean()
    
    print(f"5-year moving average computed")
    print(f"10-year moving average computed")
    
    # ========================================================================
    # 5. VISUALIZATION
    # ========================================================================
    print("\n" + "-"*80)
    print("GENERATING PLOTS")
    print("-"*80)
    
    if decomp_success:
        # Full decomposition plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Original series
        axes[0].plot(years, counts, 'o-', linewidth=1.5, markersize=4, color='steelblue', label='Observed')
        axes[0].plot(years, trend_line, 'r--', linewidth=2, label=f'Linear Trend (slope={slope:.4f})')
        axes[0].set_ylabel('Hurricane Count', fontsize=11)
        axes[0].set_title('Original Time Series with Linear Trend', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Trend component
        axes[1].plot(years, trend, linewidth=2, color='red', label='Trend')
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Seasonal component
        axes[2].plot(years, seasonal, linewidth=2, color='green', label='Seasonal')
        axes[2].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[2].set_ylabel('Seasonal', fontsize=11)
        axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        # Residual component
        axes[3].plot(years, residual, linewidth=1, color='purple', alpha=0.7, label='Residual')
        axes[3].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[3].set_ylabel('Residual', fontsize=11)
        axes[3].set_xlabel('Year', fontsize=11)
        axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
        axes[3].legend()
        axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, 'timeseries_decomposition.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")
    
    # Trend and moving averages plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(years, counts, 'o-', linewidth=1.5, markersize=5, color='steelblue', 
            label='Observed', alpha=0.6)
    ax.plot(years, trend_line, 'r-', linewidth=2.5, label=f'Linear Trend (slope={slope:.4f})')
    ax.plot(years, ma_5, 'g-', linewidth=2, label='5-year Moving Average', alpha=0.8)
    ax.plot(years, ma_10, 'orange', linewidth=2, label='10-year Moving Average', alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hurricane Count', fontsize=12, fontweight='bold')
    ax.set_title('Hurricane Frequency: Trend Analysis', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'trend_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")
    
    # Residual diagnostics (if decomposition succeeded)
    if decomp_success:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual histogram
        axes[0].hist(residual.dropna(), bins=20, density=True, alpha=0.6, 
                    color='purple', edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = np.nanmean(residual), np.nanstd(residual)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                    label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        axes[0].set_xlabel('Residual', fontsize=11)
        axes[0].set_ylabel('Density', fontsize=11)
        axes[0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Q-Q plot for residuals
        stats.probplot(residual.dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot: Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, 'residual_diagnostics.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")
    
    # ========================================================================
    # 6. SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    
    summary_stats = {
        'Metric': [
            'Data Period',
            'Number of Years',
            'Mean Frequency',
            'Std Deviation',
            'Min Count',
            'Max Count',
            '',
            'Linear Trend Slope',
            'Trend p-value',
            'Trend Significant?',
            '',
            'ADF Statistic',
            'ADF p-value',
            'Series Stationary?',
        ],
        'Value': [
            f'{years[0]} - {years[-1]}',
            f'{len(years)}',
            f'{np.mean(counts):.2f}',
            f'{np.std(counts):.2f}',
            f'{np.min(counts)}',
            f'{np.max(counts)}',
            '',
            f'{slope:.4f} hurricanes/year',
            f'{p_value:.4f}',
            'Yes' if p_value < 0.05 else 'No',
            '',
            f'{adf_result[0]:.4f}',
            f'{adf_result[1]:.4f}',
            'Yes' if adf_result[1] < 0.05 else 'No',
        ]
    }
    
    df_summary = pd.DataFrame(summary_stats)
    
    filepath = os.path.join(save_dir, 'timeseries_summary.csv')
    df_summary.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")
    
    print("\n" + "="*80)
    print("TIME SERIES ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'slope': slope,
        'p_value': p_value,
        'r_squared': r_value**2,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'trend_line': trend_line,
        'decomposition': decomposition if decomp_success else None
    }


# Example usage
if __name__ == "__main__":
    
    # Parse the file
    json_folder  = 'Data'
    json_name = 'hurdat2_all_events.csv'

    json_filepath = os.path.join(json_folder, json_name)

    df, df_per_year = load_hurricane_data(json_filepath)

    years = np.asarray(df_per_year['year'])
    counts = np.asarray(df_per_year['hurricane_count'])

    print(years)
    print(counts)
    print(years[-1])

    # Run analysis
    results = analyze_hurricane_timeseries(years, counts)
    