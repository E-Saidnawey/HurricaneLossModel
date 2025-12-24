"""
Updated Monte Carlo Simulation Wrapper
=======================================

Uses the CatModelingReport class to generate professional industry reports.
"""

import os
import numpy as np
from monte_carlo_simulation import monte_carlo_simulation
from load_data import load_hurricane_data
from Severity_Model import model_windspeed
from Report_Generator import CatModelingReport

if __name__ == "__main__":

    print('='*80)
    print('HURRICANE CATASTROPHE MODEL')
    print('='*80)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print('\n1. Loading historical data...')
    
    json_folder = 'Data'
    json_name = 'hurdat2_all_events.csv'
    json_filepath = os.path.join(json_folder, json_name)
    
    df, df_per_year = load_hurricane_data(json_filepath)
    print(f'   ✓ Loaded {len(df)} hurricane events from {len(df_per_year)} years')
    
    # ========================================================================
    # STEP 2: Fit Severity Model
    # ========================================================================
    print('\n2. Fitting severity model...')
    
    wind_model = dict()
    _, wind_model['shape'], wind_model['loc'], wind_model['scale'] = model_windspeed()
    print(f'   ✓ Lognormal severity model fitted')
    print(f'      Shape: {wind_model["shape"]:.4f}')
    print(f'      Scale: {wind_model["scale"]:.2f}')
    
    # ========================================================================
    # STEP 3: Set Up Simulation Parameters
    # ========================================================================
    print('\n3. Setting up simulation parameters...')
    
    exposure = 1e9  # $1 billion
    n_sim = 100000
    
    simulation_inputs = {
        'df_per_year': df_per_year,
        'df': df,
        'n_sim': n_sim,
        'exposure': exposure
    }
    
    print(f'   Exposure: ${exposure/1e9:.1f}B')
    print(f'   Simulations: {n_sim:,}')
    
    # Calculate base frequency parameters
    base_mean = np.mean(df_per_year['hurricane_count'])
    base_variance = np.var(df_per_year['hurricane_count'], ddof=1)
    
    print(f'   Historical mean frequency: {base_mean:.2f} hurricanes/year')
    print(f'   Historical variance: {base_variance:.2f}')
    
    # ========================================================================
    # STEP 4: Initialize Report Generator
    # ========================================================================
    print('\n4. Initializing report generator...')
    
    report = CatModelingReport(exposure)
    print('   ✓ Report generator ready')
    
    # ========================================================================
    # STEP 5: Run Scenarios
    # ========================================================================
    print('\n5. Running Monte Carlo scenarios...')
    print('   ' + '-'*76)
    
    # Define scenarios to run
    scenarios = [
        {
            'name': 'Original',
            'mean_factor': 1.0,
            'var_factor': 1.0,
            'description': 'Base historical parameters'
        },
        {
            'name': 'High_Mean',
            'mean_factor': 1.2,
            'var_factor': 1.0,
            'description': '20% increase in mean frequency'
        },
        {
            'name': 'Low_Mean',
            'mean_factor': 0.8,
            'var_factor': 1.0,
            'description': '20% decrease in mean frequency'
        },
        {
            'name': 'High_Variance',
            'mean_factor': 1.0,
            'var_factor': 2.0,
            'description': 'Double variance (higher uncertainty)'
        },
        {
            'name': 'Low_Variance',
            'mean_factor': 1.0,
            'var_factor': 0.5,
            'description': 'Half variance (lower uncertainty)'
        },
        {
            'name': 'High_Mean_High_Variance',
            'mean_factor': 1.2,
            'var_factor': 2.0,
            'description': 'Combined high mean and variance'
        }
    ]
    
    # Run each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f'\n   Scenario {i}/{len(scenarios)}: {scenario["name"]}')
        print(f'   {scenario["description"]}')
        
        # Set frequency parameters
        hurricane_model = {
            'mean': base_mean * scenario['mean_factor'],
            'variance': base_variance * scenario['var_factor']
        }
        
        print(f'   Mean: {hurricane_model["mean"]:.2f}, Variance: {hurricane_model["variance"]:.2f}')
        
        # Run simulation
        hurricanes, losses = monte_carlo_simulation(
            simulation_inputs, 
            wind_model, 
            hurricane_model
        )
        
        # Add to report
        report.add_scenario(
            scenario['name'],
            hurricanes,
            losses,
            hurricane_model
        )
        
        # Quick preview
        aal = np.mean(losses)
        print(f'   ✓ Simulation complete')
        print(f'      AAL: ${aal/1e6:.2f}M')
        print(f'      99th percentile: ${np.percentile(losses, 99)/1e6:.2f}M')
    
    # ========================================================================
    # STEP 6: Generate Report
    # ========================================================================
    print('\n' + '='*80)
    print('6. Generating comprehensive report...')
    print('='*80)
    
    output_dir = 'Results/Professional_Report'
    summary_df = report.generate_full_report(output_dir)
    
    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print(f'\nAll reports saved to: {output_dir}/')
    print('\nFiles generated:')
    print(f'  • {len(scenarios)} individual scenario reports (*.png)')
    print(f'  • 1 scenario comparison dashboard (scenario_comparison.png)')
    print(f'  • 1 summary table (scenario_summary.csv)')
    print('\nNext steps:')
    print('  1. Review individual scenario reports for detailed analysis')
    print('  2. Use comparison dashboard for executive summary')
    print('  3. Import summary CSV for further analysis or presentation')