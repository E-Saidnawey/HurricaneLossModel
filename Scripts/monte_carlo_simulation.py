import numpy as np
from scipy.stats import nbinom, lognorm 
from load_data import load_hurricane_data
import os
from Severity_Model import model_windspeed
import matplotlib.pyplot as plt


def damage_ratio(wind, V0=74, alpha=0.0005, beta=3):
    """
    Maps wind speed (mph) to fractional damage.

    **mph = knots since this is landfall of a hurricane, at sea level

    """
    return np.minimum(alpha * (wind / V0)**beta, 1.0)

def simulate_one_year(hurricane_model, wind_model, exposure) -> tuple[float, int]:
    """
    Simulate total hurricane loss for one year.
    """

    shape = wind_model['shape']
    loc   = wind_model['loc']
    scale = wind_model['scale']

    mu = np.log(scale)

    sample_mean = hurricane_model['mean']
    sample_var  = hurricane_model['variance']
    dispersion  = sample_var / sample_mean

    r_mom = sample_mean**2 / (sample_var - sample_mean)
    p_mom = r_mom / (r_mom + sample_mean)

    n_events = nbinom.rvs(r_mom, p_mom)

    if n_events == 0:
        return 0.0, 0

    winds = lognorm.rvs(s=shape, scale=np.exp(mu), size=n_events)
    losses = damage_ratio(winds) * exposure
    annual_loss = min(losses.sum(), exposure)

    return (annual_loss, n_events)

def monte_carlo_simulation(simulation_inputs, wind_model=None, hurricane_model=None):

    if hurricane_model is None:
        hurricane_model['mean'] = np.mean(simulation_inputs['df_per_year']['hurricane_count'])
        hurricane_model['variance'] = np.var(simulation_inputs['df_per_year']['hurricane_count'], ddof=1)

    if wind_model is None:
        wind_model = dict()
        _, wind_model['shape'], wind_model['loc'], wind_model['scale'] = model_windspeed()

    n_sim    = simulation_inputs['n_sim']  # number of simulated years
    exposure = simulation_inputs['exposure']
    simulation_results = np.array([simulate_one_year(hurricane_model, wind_model, exposure) for _ in range(n_sim)])

    simulated_losses = [simulation_results[i][0] for i in range(n_sim)]
    simulated_hurricanes = [simulation_results[i][1] for i in range(n_sim)]

    print(f"Expected Annual Loss (EAL): ${np.mean(simulated_losses) / 1e9:.2f}B")
    print(f"99th percentile loss: ${np.percentile(simulated_losses, 99) / 1e9:.2f}B")

    print(f"Expected Annual number of hurricanes: {np.mean(simulated_hurricanes):.2f}")
    print(f"99th percentile number of hurricanes: {np.percentile(simulated_hurricanes, 99):.2f}")


    return simulated_hurricanes, simulated_losses

def plot_results(simulated_hurricanes, simulated_losses, savepath):

    os.makedirs(savepath, exist_ok=True)

    _, sigma, loc, scale = model_windspeed()
    mu = np.log(scale) 

    sample_mean = np.mean(simulated_hurricanes)
    sample_var = np.var(simulated_hurricanes, ddof=1)  # Sample variance (n-1)
    dispersion = sample_var / sample_mean

    r_mom = sample_mean**2 / (sample_var - sample_mean)
    p_mom = r_mom / (r_mom + sample_mean)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- LEFT PLOT: Histogram with fitted PMFs ---
    ax1 = axes[0]

    # Histogram of actual data
    counts, bins, _ = ax1.hist(simulated_hurricanes, bins=15, density=True, alpha=0.6, 
                                color='steelblue', edgecolor='black', 
                                label='Observed Data')

    # Range for PMF
    x_range = np.arange(0, 1000, 1)

    # Negative Binomial PMF
    
    nb_pmf = nbinom.pmf(x_range, n=r_mom, p=p_mom)
    ax1.plot(x_range, nb_pmf, 'r-', linewidth=2, 
            label=f'Negative Binomial (r={r_mom:.2f})', alpha=0.8)

    ax2 = axes[1]

    # Sort observed data
    sorted_data = np.sort(simulated_hurricanes)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    theoretical_quantiles_nb = nbinom.ppf(empirical_cdf, n=r_mom, p=p_mom)
    ax2.scatter(theoretical_quantiles_nb, sorted_data, alpha=0.6, s=50,
                label='Negative Binomial', color='red')

    valid_quantiles = theoretical_quantiles_nb[np.isfinite(theoretical_quantiles_nb)]
    max_val = max(np.max(sorted_data), np.max(valid_quantiles))

    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect Fit')

    ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax2.set_ylabel('Observed Quantiles', fontsize=12)
    ax2.set_title('Q-Q Plot: Model Fit Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'simulation_results.png'), dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: simulation_results.png")
    plt.close()

    fig, axes = plt.subplots(figsize=(14, 5))
    running_mean = np.cumsum(simulated_losses) / np.arange(1, len(simulated_losses)+1)
    plt.plot(running_mean)
    plt.xlabel("Simulation count")
    plt.ylabel("EAL estimate")
    plt.title("Monte Carlo Convergence of EAL")
    plt.savefig(os.path.join(savepath, 'running_mean.png'), dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: running_mean.png")
    plt.close()

    losses_sorted = np.sort(simulated_losses)
    ep = 1 - np.arange(1, len(losses_sorted)+1) / len(losses_sorted)
    fig, axes = plt.subplots(figsize=(14, 5))
    plt.plot(losses_sorted, ep)
    plt.xlabel("Losses")
    plt.ylabel("EP")
    plt.title("Exceedance Probability Curve")
    plt.savefig(os.path.join(savepath, 'exceedance_probability.png'), dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: exceedance_probability.png")
    plt.close()


    return


if __name__ == '__main__':
    # Parse the file
    json_folder  = 'Data'
    json_name = 'hurdat2_all_events.csv'

    json_filepath = os.path.join(json_folder, json_name)

    df, df_per_year = load_hurricane_data(json_filepath)

    simulation_inputs = {
        'df_per_year': df_per_year,
        'df': df,
        'n_sim': 100000,
        'exposure': 1e9
    }

    n_sim    = simulation_inputs['n_sim']  # number of simulated years
    exposure = simulation_inputs['exposure']

    hurricanes, losses = monte_carlo_simulation(simulation_inputs, wind_model, hurricane_model)

    plot_results(hurricanes, losses)

    