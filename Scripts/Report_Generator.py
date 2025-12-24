"""
Catastrophe Modeling Report Generator
======================================

Generates professional insurance/reinsurance industry reports with:
- Consistent layouts across scenarios
- Key risk metrics (AAL, TVaR, return periods)
- Comparable visualizations
- Summary comparison dashboard
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import pandas as pd

class CatModelingReport:
    """
    Handles storage and visualization of catastrophe modeling results
    across multiple scenarios.
    """
    
    def __init__(self, exposure):
        """
        Initialize report generator.
        
        Args:
            exposure: Total exposure value (e.g., 1e9 for $1B)
        """
        self.exposure = exposure
        self.scenarios = {}
        self.global_limits = {
            'max_loss': 0,
            'max_hurricanes': 0,
            'max_ep_loss': 0
        }
        
    def add_scenario(self, name, hurricanes, losses, params):
        """
        Add a scenario's results to the report.
        
        Args:
            name: Scenario identifier (e.g., 'Original', 'High_Mean')
            hurricanes: Array of simulated hurricane counts
            losses: Array of simulated losses
            params: Dict with 'mean', 'variance' of frequency model
        """
        hurricanes = np.asarray(hurricanes)
        losses = np.asarray(losses)

        # Calculate key metrics
        metrics = self._calculate_metrics(hurricanes, losses)
        
        # Fit distributions
        freq_params = self._fit_frequency_distribution(hurricanes)
        
        # Store everything
        self.scenarios[name] = {
            'hurricanes': hurricanes,
            'losses': losses,
            'params': params,
            'freq_params': freq_params,
            'metrics': metrics
        }
        
        # Update global limits for consistent axes
        self.global_limits['max_loss'] = max(
            self.global_limits['max_loss'],
            np.max(losses)
        )
        self.global_limits['max_hurricanes'] = max(
            self.global_limits['max_hurricanes'],
            np.max(hurricanes)
        )
        self.global_limits['max_ep_loss'] = max(
            self.global_limits['max_ep_loss'],
            np.percentile(losses, 99.9)
        )
        
    def _calculate_metrics(self, hurricanes, losses):
        """Calculate industry-standard risk metrics."""
        
        # Sort losses for percentile calculations
        losses_sorted = np.sort(losses)
        
        # Exceedance probabilities
        ep = 1 - np.arange(1, len(losses_sorted) + 1) / len(losses_sorted)
        
        # Find return period losses
        def get_return_period_loss(years):
            ep_target = 1.0 / years
            idx = np.searchsorted(-ep, -ep_target)  # Find closest
            return losses_sorted[min(idx, len(losses_sorted)-1)]
        
        # Calculate TVaR (Tail Value at Risk) = average loss beyond threshold
        def tvar(losses_array, percentile):
            threshold = np.percentile(losses_array, percentile)
            losses_array = np.asarray(losses_array)
            tail_losses = losses_array[losses_array >= threshold]
            return np.mean(tail_losses) if len(tail_losses) > 0 else 0
        
        metrics = {
            # Basic statistics
            'aal': np.mean(losses),  # Average Annual Loss
            'std': np.std(losses),
            'cv': np.std(losses) / np.mean(losses) if np.mean(losses) > 0 else 0,  # Coefficient of Variation
            
            # Frequency stats
            'avg_hurricanes': np.mean(hurricanes),
            'std_hurricanes': np.std(hurricanes),
            
            # Percentiles
            'p50': np.percentile(losses, 50),
            'p90': np.percentile(losses, 90),
            'p95': np.percentile(losses, 95),
            'p99': np.percentile(losses, 99),
            'p99.9': np.percentile(losses, 99.9),
            
            # Return periods
            'rp_10': get_return_period_loss(10),
            'rp_50': get_return_period_loss(50),
            'rp_100': get_return_period_loss(100),
            'rp_250': get_return_period_loss(250),
            'rp_500': get_return_period_loss(500),
            
            # Tail risk
            'tvar_90': tvar(losses, 90),
            'tvar_95': tvar(losses, 95),
            'tvar_99': tvar(losses, 99),
            
            # Loss ratios
            'aal_ratio': np.mean(losses) / self.exposure * 100,  # AAL as % of exposure
            'max_loss_ratio': np.max(losses) / self.exposure * 100
        }
        
        return metrics
    
    def _fit_frequency_distribution(self, hurricanes):
        """Fit Negative Binomial to frequency data."""
        sample_mean = np.mean(hurricanes)
        sample_var = np.var(hurricanes, ddof=1)
        
        if sample_var > sample_mean:
            r_mom = sample_mean**2 / (sample_var - sample_mean)
            p_mom = r_mom / (r_mom + sample_mean)
            return {'r': r_mom, 'p': p_mom, 'dispersion': sample_var / sample_mean}
        else:
            # Fallback to Poisson if underdispersed
            return {'r': None, 'p': None, 'dispersion': sample_var / sample_mean}
    
    def plot_individual_scenario(self, scenario_name, save_dir):
        """
        Create a comprehensive 4-panel figure for a single scenario.
        
        Panels:
        A. Frequency distribution (hurricanes/year)
        B. Annual loss distribution
        C. Exceedance probability curve
        D. Monte Carlo convergence
        """
        scenario = self.scenarios[scenario_name]
        hurricanes = scenario['hurricanes']
        losses = scenario['losses']
        metrics = scenario['metrics']
        freq_params = scenario['freq_params']
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # ============================================================
        # PANEL A: Frequency Distribution
        # ============================================================
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Histogram
        counts, bins, _ = ax1.hist(hurricanes, bins=30, density=True, 
                                     alpha=0.6, color='steelblue', 
                                     edgecolor='black', label='Simulated')
        
        # Fitted Negative Binomial
        if freq_params['r'] is not None:
            x_range = np.arange(0, int(self.global_limits['max_hurricanes']) + 1)
            nb_pmf = nbinom.pmf(x_range, n=freq_params['r'], p=freq_params['p'])
            ax1.plot(x_range, nb_pmf, 'r-', linewidth=2, 
                    label=f'NegBin (r={freq_params["r"]:.2f})', alpha=0.8)
        
        ax1.axvline(metrics['avg_hurricanes'], color='black', 
                   linestyle='--', linewidth=2, label=f'Mean = {metrics["avg_hurricanes"]:.1f}')
        ax1.set_xlabel('Hurricanes per Year', fontsize=11)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title('A. Frequency Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, self.global_limits['max_hurricanes'] * 1.1)
        
        # ============================================================
        # PANEL B: Annual Loss Distribution
        # ============================================================
        ax2 = fig.add_subplot(gs[0, 1])
        
        ax2.hist(losses / 1e6, bins=50, density=True, alpha=0.6, 
                color='coral', edgecolor='black')
        ax2.axvline(metrics['aal'] / 1e6, color='red', linestyle='--', 
                   linewidth=2, label=f'AAL = ${metrics["aal"]/1e6:.1f}M')
        ax2.axvline(metrics['p99'] / 1e6, color='darkred', linestyle=':', 
                   linewidth=2, label=f'99th % = ${metrics["p99"]/1e6:.1f}M')
        ax2.set_xlabel('Annual Loss ($M)', fontsize=11)
        ax2.set_ylabel('Probability Density', fontsize=11)
        ax2.set_title('B. Annual Loss Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, self.global_limits['max_loss'] / 1e6 * 1.1)
        
        # ============================================================
        # PANEL C: Exceedance Probability Curve
        # ============================================================
        ax3 = fig.add_subplot(gs[1, :])
        
        losses_sorted = np.sort(losses)
        ep = 1 - np.arange(1, len(losses_sorted) + 1) / len(losses_sorted)
        
        ax3.plot(losses_sorted / 1e6, ep, linewidth=2, color='navy', alpha=0.8)
        
        # Mark key return periods
        return_periods = [10, 50, 100, 250, 500]
        colors = ['green', 'orange', 'red', 'darkred', 'purple']
        
        for rp, color in zip(return_periods, colors):
            ep_target = 1.0 / rp
            loss_rp = metrics[f'rp_{rp}']
            ax3.scatter(loss_rp / 1e6, ep_target, s=100, color=color, 
                       marker='o', edgecolors='black', linewidth=2, 
                       label=f'{rp}-yr: ${loss_rp/1e6:.1f}M', zorder=5)
            ax3.axhline(ep_target, color=color, linestyle=':', alpha=0.5)
        
        ax3.set_xlabel('Loss ($M)', fontsize=11)
        ax3.set_ylabel('Exceedance Probability', fontsize=11)
        ax3.set_title('C. Exceedance Probability Curve with Return Periods', 
                     fontsize=12, fontweight='bold')
        ax3.set_yscale('log')
        ax3.set_xlim(0, self.global_limits['max_ep_loss'] / 1e6 * 1.1)
        ax3.set_ylim(1e-4, 1)
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(alpha=0.3, which='both')
        
        # ============================================================
        # PANEL D: Monte Carlo Convergence
        # ============================================================
        ax4 = fig.add_subplot(gs[2, 0])
        
        running_mean = np.cumsum(losses) / np.arange(1, len(losses) + 1)
        ax4.plot(running_mean / 1e6, linewidth=1.5, color='teal')
        ax4.axhline(metrics['aal'] / 1e6, color='red', linestyle='--', 
                   linewidth=2, label=f'Final AAL = ${metrics["aal"]/1e6:.1f}M')
        ax4.set_xlabel('Simulation Number', fontsize=11)
        ax4.set_ylabel('Cumulative Mean Loss ($M)', fontsize=11)
        ax4.set_title('D. Monte Carlo Convergence', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        # ============================================================
        # PANEL E: Key Metrics Table
        # ============================================================
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Create metrics table
        table_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Average Annual Loss (AAL)', f'${metrics["aal"]/1e6:.2f}M'],
            ['AAL as % of Exposure', f'{metrics["aal_ratio"]:.3f}%'],
            ['Standard Deviation', f'${metrics["std"]/1e6:.2f}M'],
            ['Coefficient of Variation', f'{metrics["cv"]:.2f}'],
            ['', ''],
            ['50th Percentile', f'${metrics["p50"]/1e6:.2f}M'],
            ['90th Percentile', f'${metrics["p90"]/1e6:.2f}M'],
            ['99th Percentile', f'${metrics["p99"]/1e6:.2f}M'],
            ['99.9th Percentile', f'${metrics["p99.9"]/1e6:.2f}M'],
            ['', ''],
            ['100-year Return Period', f'${metrics["rp_100"]/1e6:.2f}M'],
            ['250-year Return Period', f'${metrics["rp_250"]/1e6:.2f}M'],
            ['', ''],
            ['TVaR 95%', f'${metrics["tvar_95"]/1e6:.2f}M'],
            ['TVaR 99%', f'${metrics["tvar_99"]/1e6:.2f}M'],
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style separator rows
        for row in [1, 6, 11, 13]:
            for col in range(2):
                table[(row, col)].set_facecolor('#E7E6E6')
        
        ax5.set_title('E. Key Risk Metrics', fontsize=12, fontweight='bold', pad=20)
        
        # ============================================================
        # Add overall title and save
        # ============================================================
        scenario_label = scenario_name.replace('_', ' ')
        fig.suptitle(f'Hurricane Catastrophe Model: {scenario_label} Scenario\n' +
                    f'Exposure: ${self.exposure/1e9:.1f}B | ' +
                    f'Simulations: {len(losses):,}',
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'{scenario_name}_report.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filepath}")
    
    def plot_scenario_comparison(self, save_dir):
        """
        Create comparison plots across all scenarios.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Define colors for each scenario
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        scenario_names = list(self.scenarios.keys())
        
        # ============================================================
        # PANEL 1: EP Curves Overlay
        # ============================================================
        ax1 = fig.add_subplot(gs[0, :])
        
        for i, (name, scenario) in enumerate(self.scenarios.items()):
            losses_sorted = np.sort(scenario['losses'])
            ep = 1 - np.arange(1, len(losses_sorted) + 1) / len(losses_sorted)
            label = name.replace('_', ' ')
            ax1.plot(losses_sorted / 1e6, ep, linewidth=2.5, 
                    color=colors[i % len(colors)], label=label, alpha=0.8)
        
        ax1.set_xlabel('Loss ($M)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Exceedance Probability', fontsize=12, fontweight='bold')
        ax1.set_title('Exceedance Probability Curves - All Scenarios', 
                     fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.set_xlim(0, self.global_limits['max_ep_loss'] / 1e6 * 1.1)
        ax1.set_ylim(1e-4, 1)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(alpha=0.3, which='both')
        
        # ============================================================
        # PANEL 2: AAL Comparison
        # ============================================================
        ax2 = fig.add_subplot(gs[1, 0])
        
        aals = [scenario['metrics']['aal'] / 1e6 for scenario in self.scenarios.values()]
        labels = [name.replace('_', ' ') for name in scenario_names]
        
        bars = ax2.bar(range(len(aals)), aals, color=colors[:len(aals)], 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_xticks(range(len(aals)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Average Annual Loss ($M)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Annual Loss Comparison', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.1f}M', ha='center', va='bottom', fontsize=9)
        
        # ============================================================
        # PANEL 3: Return Period Comparison
        # ============================================================
        ax3 = fig.add_subplot(gs[1, 1])
        
        return_periods = ['rp_100', 'rp_250', 'rp_500']
        rp_labels = ['100-yr', '250-yr', '500-yr']
        x = np.arange(len(rp_labels))
        width = 0.8 / len(scenario_names)
        
        for i, (name, scenario) in enumerate(self.scenarios.items()):
            values = [scenario['metrics'][rp] / 1e6 for rp in return_periods]
            offset = (i - len(scenario_names)/2 + 0.5) * width
            ax3.bar(x + offset, values, width, label=name.replace('_', ' '),
                   color=colors[i % len(colors)], edgecolor='black', 
                   linewidth=1, alpha=0.8)
        
        ax3.set_xlabel('Return Period', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Loss ($M)', fontsize=11, fontweight='bold')
        ax3.set_title('Return Period Loss Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(rp_labels)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, axis='y')
        
        # ============================================================
        # Save
        # ============================================================
        fig.suptitle(f'Hurricane Catastrophe Model: Scenario Comparison\n' +
                    f'Exposure: ${self.exposure/1e9:.1f}B',
                    fontsize=14, fontweight='bold', y=0.98)
        
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'scenario_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filepath}")
    
    def generate_summary_table(self, save_dir):
        """
        Generate a CSV summary table of all scenarios.
        """
        rows = []
        
        for name, scenario in self.scenarios.items():
            metrics = scenario['metrics']
            params = scenario['params']
            
            row = {
                'Scenario': name.replace('_', ' '),
                'Input_Mean': params['mean'],
                'Input_Variance': params['variance'],
                'Simulated_Mean_Hurricanes': metrics['avg_hurricanes'],
                'AAL_$M': metrics['aal'] / 1e6,
                'AAL_%_Exposure': metrics['aal_ratio'],
                'Std_Dev_$M': metrics['std'] / 1e6,
                'CV': metrics['cv'],
                '50th_Pctl_$M': metrics['p50'] / 1e6,
                '90th_Pctl_$M': metrics['p90'] / 1e6,
                '99th_Pctl_$M': metrics['p99'] / 1e6,
                '100yr_RP_$M': metrics['rp_100'] / 1e6,
                '250yr_RP_$M': metrics['rp_250'] / 1e6,
                '500yr_RP_$M': metrics['rp_500'] / 1e6,
                'TVaR_95%_$M': metrics['tvar_95'] / 1e6,
                'TVaR_99%_$M': metrics['tvar_99'] / 1e6,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'scenario_summary.csv')
        df.to_csv(filepath, index=False, float_format='%.2f')
        
        print(f"✓ Saved: {filepath}")
        
        return df
    
    def generate_full_report(self, output_dir='Results/Report'):
        """
        Generate complete report: individual scenario pages + comparison + summary.
        """
        print("\n" + "="*80)
        print("GENERATING CATASTROPHE MODELING REPORT")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate individual scenario reports
        print("\nGenerating individual scenario reports...")
        for name in self.scenarios.keys():
            self.plot_individual_scenario(name, output_dir)
        
        # Generate comparison plots
        print("\nGenerating scenario comparison...")
        self.plot_scenario_comparison(output_dir)
        
        # Generate summary table
        print("\nGenerating summary table...")
        summary_df = self.generate_summary_table(output_dir)
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {output_dir}/")
        print(f"  - {len(self.scenarios)} individual scenario reports")
        print(f"  - 1 comparison dashboard")
        print(f"  - 1 summary CSV table")
        
        # Display summary table
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        return summary_df



def example_usage():
    """
    Example of how to use the CatModelingReport class.
    """
    # Initialize report
    exposure = 1e9  # $1 billion
    report = CatModelingReport(exposure)
    
    # Simulate some dummy data for demonstration
    np.random.seed(42)
    
    # Scenario 1: Original
    hurricanes_orig = np.random.negative_binomial(5, 0.02, 100000)
    losses_orig = np.random.lognormal(15, 2, 100000) * 1e4
    report.add_scenario('Original', hurricanes_orig, losses_orig, 
                       {'mean': 250, 'variance': 12500})
    
    # Scenario 2: High Mean
    hurricanes_high = np.random.negative_binomial(6, 0.017, 100000)
    losses_high = np.random.lognormal(15, 2, 100000) * 1e4
    report.add_scenario('High_Mean', hurricanes_high, losses_high,
                       {'mean': 300, 'variance': 12500})
    
    # Generate full report
    report.generate_full_report()


if __name__ == "__main__":
    print("Catastrophe Modeling Report Generator")
    print("Import this module and use CatModelingReport class in your wrapper script.")
    print("\nSee example_usage() for demonstration.")