import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, poisson
from scipy.optimize import minimize
import pandas as pd
import os
from load_data import load_hurricane_data


def statistical_testing(hurricanes_per_year):

    output_df = dict()

    print("\n" + "="*80)
    print("NEGATIVE BINOMIAL FIT")
    print("="*80)

    data = hurricanes_per_year['hurricane_count']
    time = hurricanes_per_year['year']

    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)  # Sample variance (n-1)
    dispersion = sample_var / sample_mean

    print(f"\nSample Statistics:")
    print(f"  n = {len(data)} years")
    print(f"  Sample Mean (μ) = {round(sample_mean,2)}")
    print(f"  Sample Variance (σ²) = {round(sample_var,2)}")
    print(f"  Dispersion (σ²/μ) = {round(dispersion,2)}")
    print(f"  Standard Deviation = {round(np.sqrt(sample_var),2)}")
    print(f"  Min = {np.min(data)}, Max = {np.max(data)}")

    r_mom = sample_mean**2 / (sample_var - sample_mean)
    p_mom = r_mom / (r_mom + sample_mean)

    output_df['Negative_Binomial'] = {
        'mean': sample_mean,
        'variance': sample_var,
        'r_mom': r_mom,
        'p_mom': p_mom
    }
    
    print(f"\nMethod of Moments Estimates:")
    print(f"  r (shape parameter) = {round(r_mom,4)}")
    print(f"  p (success probability) = {round(p_mom,4)}")
    print(f"  Implied mean = {round(r_mom * (1-p_mom) / p_mom,2)}")
    print(f"  Implied variance = {round(r_mom * (1-p_mom) / p_mom**2,2)}")

    print("\n" + "="*80)
    print("POISSON FIT")
    print("="*80)
    lambda_mle = sample_mean
    print(f"\nMaximum Likelihood Estimate:")
    print(f"  λ (rate parameter) = {round(lambda_mle,2)}")
    print(f"  Implied variance = {round(lambda_mle,2)} (constraint: Var = Mean)")

    output_df['Poisson'] = {
            'mean': sample_mean,
            'variance': sample_mean
        }

       
    return output_df
    
def plot_statistics(df_per_year, statistics_df):
    
    data = df_per_year['hurricane_count']
    time = df_per_year['year']

    r_mom = statistics_df['Negative_Binomial']['r_mom']
    p_mom = statistics_df['Negative_Binomial']['p_mom']
    lambda_mle = statistics_df['Poisson']['mean']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- LEFT PLOT: Histogram with fitted PMFs ---
    ax1 = axes[0]

    # Histogram of actual data
    counts, bins, _ = ax1.hist(data, bins=15, density=True, alpha=0.6, 
                                color='steelblue', edgecolor='black', 
                                label='Observed Data')

    # Range for PMF
    x_range = np.arange(0, 1000, 1)

    # Negative Binomial PMF
    if r_mom is not None:
        nb_pmf = nbinom.pmf(x_range, n=r_mom, p=p_mom)
        ax1.plot(x_range, nb_pmf, 'r-', linewidth=2, 
                label=f'Negative Binomial (r={r_mom:.2f})', alpha=0.8)

    # Poisson PMF
    poisson_pmf = poisson.pmf(x_range, mu=lambda_mle)
    ax1.plot(x_range, poisson_pmf, 'g--', linewidth=2, 
            label=f'Poisson (λ={lambda_mle:.0f})', alpha=0.8)

    ax1.set_xlabel('Annual Hurricane Count', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Histogram of Annual Counts with Fitted Distributions', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1000)

    # --- RIGHT PLOT: Q-Q style comparison ---
    ax2 = axes[1]

    # Sort observed data
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Theoretical quantiles from Negative Binomial
    if r_mom is not None:
        theoretical_quantiles_nb = nbinom.ppf(empirical_cdf, n=r_mom, p=p_mom)
        ax2.scatter(theoretical_quantiles_nb, sorted_data, alpha=0.6, s=50,
                    label='Negative Binomial', color='red')

    # Theoretical quantiles from Poisson
    theoretical_quantiles_poisson = poisson.ppf(empirical_cdf, mu=lambda_mle)
    ax2.scatter(theoretical_quantiles_poisson, sorted_data, alpha=0.6, s=50,
                label='Poisson', color='green', marker='x')

    # Perfect fit line
    valid_quantiles_nb = theoretical_quantiles_nb[np.isfinite(theoretical_quantiles_nb)]
    valid_quantiles_poisson = theoretical_quantiles_poisson[np.isfinite(theoretical_quantiles_poisson)]
    max_val = max(np.max(sorted_data), np.max(valid_quantiles_nb), np.max(valid_quantiles_poisson))
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect Fit')

    ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax2.set_ylabel('Observed Quantiles', fontsize=12)
    ax2.set_title('Q-Q Plot: Model Fit Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('Results/negbinom_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: negbinom_analysis.png")

# Example usage
if __name__ == "__main__":
    # Parse the file
    json_folder  = 'Data'
    json_name = 'hurdat2_all_events.csv'

    json_filepath = os.path.join(json_folder, json_name)

    df, df_per_year = load_hurricane_data(json_filepath)

    stats_df = statistical_testing(df_per_year)

    plot_statistics(df_per_year, stats_df)